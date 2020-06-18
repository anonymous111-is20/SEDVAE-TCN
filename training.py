import os
import sys
# sys.path.append('/home/manhlt/VisualProject/speenai-is20')
import numpy as np 
import torch
import shutil
from speenai.utils.utils import SignalToFrames, ToTensor
# from torch.utils.data import DataLoader
from speenai.dataset.wavdataset_loader import WavDataset, DataLoader
from speenai.models.nets.tcnn import TCNN
from speenai.models.nets.dense_tcnn import Dense_TCNN
from speenai.commands.eval import cal_scores
from speenai.evals.loss_function import stftm_loss
from speenai.evals.metrics import pytorch_si_sdr
from speenai.evals.si_snr import cal_loss
from speenai.models.nets.conv_tasnet import TasNet
import timeit
import argparse
from pathlib import Path
from tensorboardX import SummaryWriter
import datetime
import pandas as pd
from speenai.dataset.example_signal_ola import OLA,SignalToFrames
from speenai.models.nets.dvae_with_tcn import DVAE
from speenai.models.nets.dvae_with_filter import DVAE2
import math



def degrade_lr(logg):
    arr = logg[-3:-1]
    curr_loss = logg[-1]
    if len(logg) < 3:
        return False
    elif all(curr_loss >= x for x in arr):
        return True
    else:
        return False

class Checkpoint(object):

    def __init__(self,
                start_epoch=None,
                start_iter=None,
                train_loss=None,
                eval_loss=None,
                best_loss=np.inf,
                state_dict=None,
                optimizer=None):

        self.start_epoch = start_epoch
        self.start_iter = start_iter
        self.train_loss = train_loss
        self.eval_loss = eval_loss
        self.best_loss = best_loss
        self.state_dict = state_dict
        self.optimizer = optimizer

    def save(self, filename, best_model,is_best):
        print("Saving checkpoint at %s"%filename)
        torch.save({
            'start_epoch': self.start_epoch,
            'start_iter': self.start_iter,
            'train_loss': self.train_loss,
            'eval_loss': self.eval_loss,
            'best_loss': self.best_loss,
            'state_dict': self.state_dict,
            'optimizer': self.optimizer
        }, filename)
        if is_best:
            print('Saving the best model at %s'%best_model)
            shutil.copyfile(filename, best_model)
        print('-------------------------------------------')
    
    def load(self, filename):
        if os.path.isfile(filename):
            print('Loading checkpoint from %s' % filename)
            checkpoint = torch.load(filename, map_location='cpu')

            self.start_epoch = checkpoint['start_epoch']
            self.start_iter = checkpoint['start_iter']
            self.train_loss = checkpoint['train_loss']
            self.eval_loss = checkpoint['eval_loss']
            self.best_loss = checkpoint['best_loss']
            self.state_dict = checkpoint['state_dict']
            self.optimizer = checkpoint['optimizer']

        else:
            raise ValueError('No checkpoint found at %s' % filename)

class Model(object):

    def __init__(self, args):

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        log_dir = args.model_name +'_'+ datetime.datetime.now().strftime("%Y%m%D_H%M%S")
        self.model_name = args.model_name
        self.writer = SummaryWriter(args.model_name)
        csv_file_name = self.model_name + datetime.datetime.now().strftime("%Y%m%D_%H-%M-%S").replace('/', '-')+'.csv'
        self.log_file_path = os.path.join(Path.cwd().parent.parent, args.note +'_'+ csv_file_name)
        self.log_data = []
        self.ola = OLA(160)
        self.sigtoframe = SignalToFrames(frame_size=320, frame_shift=160)
        self.lr_schedule = args.lr_schedule
        self.alpha = args.alpha
        self.logg_loss = []
        print('Using device: ', self.device)

    def train(self, args):

        # torch.autograd.set_detect_anomaly(True)

        self.mixture_dir = args.mixture_dir
        self.clean_dir = args.clean_dir
        self.csv_train_file = args.csv_train_file
        self.csv_validation_file = args.csv_validation_file

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.max_epoch = args.max_epoch
        self.model_path = args.model_path
        self.log_path = args.log_path
        self.eval_steps = args.eval_steps
        self.resume_model = args.resume_model
        self.model_name = args.model_name
        

        

        trainset = WavDataset(mixture_dataset_dir=self.mixture_dir,
                             clean_dataset_dir=self.clean_dir,
                             csv_file=self.csv_train_file)

        validationset = WavDataset(mixture_dataset_dir=self.mixture_dir,
                             clean_dataset_dir=self.clean_dir,
                             csv_file=self.csv_train_file)

        testset = WavDataset(mixture_dataset_dir=self.mixture_dir,
                                   clean_dataset_dir=self.clean_dir,
                                   csv_file=self.csv_validation_file)

        validation_idx = int(trainset.__len__()*(8/9))
        trainloader = DataLoader(validation_idx, trainset)
        validationloader = DataLoader(dataset_size=trainset.__len__(), dataset=validationset,begin_idx=validation_idx)
        evalloader = DataLoader(testset.__len__(),testset)

        if self.model_name == 'tcnn':
            net = TCNN(self.batch_size, causal=False)
        if self.model_name=='dense_tcnn':
            net = Dense_TCNN(L=320, causal=False)
            self.stftm_loss_function = stftm_loss(frame_size=320, frame_shift=160, device=self.device)
        elif self.model_name=='conv_tasnet':
            net = TasNet()
        elif self.model_name == 'proposed':
            self.stftm_loss_function = stftm_loss(frame_size=320, frame_shift=160, device=self.device)

            net = DVAE(args)
            if torch.cuda.device_count() > 1:
                print('Lets use ', torch.cuda.device_count(),'GPUs')
                net = torch.nn.DataParallel(net)
        
        net.to(self.device)


        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)
        scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.2)

        if self.resume_model:
            print('Resume model from %s'%self.resume_model)

            checkpoint = Checkpoint()
            checkpoint.load(self.resume_model)
            start_epoch = checkpoint.start_epoch
            start_iter = checkpoint.start_iter
            best_loss = checkpoint.best_loss
            net.load_state_dict(checkpoint.state_dict)
            optimizer.load_state_dict(checkpoint.optimizer)
        else:
            print('Training from scratch')
            start_epoch = 0
            start_iter = 0
            best_loss = np.inf
        net.train()
        num_train_batches = validation_idx // self.batch_size
        ttime = 0
        cnt = 0

        for epoch in range(start_epoch, self.max_epoch):
######################################################################################3
            # if epoch==3 or epoch==10:
            #     scheduler1.step()
            # elif epoch==12:
            #     scheduler2.step()
############################################################################################3
            accu_train_loss = 0
            start = timeit.default_timer()
            for i in range(num_train_batches):

                mixture, clean = trainloader.get_batch(self.batch_size)
 
                mixture = mixture.to(self.device)
                clean = clean.to(self.device)
                optimizer.zero_grad()


                mixture = mixture.unsqueeze(1)
                

                if self.model_name == 'tcnn':
                    outputs = net(mixture)
                    loss = mse_loss(outputs, clean)
                    # source_length = [32000 for i in range(self.batch_size)]
                    # source_length = torch.cuda.FloatTensor(source_length)
                    # clean = clean.view(-1, 1, 32000)
                    # outputs = outputs.view(-1, 1, 32000)
                    # loss,_,_,_ = cal_loss(clean, outputs, source_length)
                if self.model_name == 'dense_tcnn':
                    ola_output, outputs = net(mixture)
                    ola_output = ola_output.squeeze(1)
                    # clean = self.ola(clean)
                    ola_output = self.sigtoframe(ola_output)[:,:-1,:]
                    # clean = self.ola(clean)
                    output = outputs.view(-1, 100, 320)
                    loss1 = mse_loss(clean, ola_output)
                    loss2 = self.stftm_loss_function(ola_output, clean) #input [batch, nframes, frame_length]
                    loss = 0.8* loss1 + 0.2*loss2
                elif self.model_name=='conv_tasnet':
                    outputs = net(mixture)
                    source_length = [32000 for i in range(self.batch_size)]
                    source_length = torch.cuda.FloatTensor(source_length)
                    clean = clean.view(-1, 1, 32000)
                    outputs = outputs.view(-1, 1, 32000)
                    loss,_,_,_ = cal_loss(clean, outputs, source_length) # input [B, C, T]
                elif self.model_name=='proposed':
                    source_length = [32000 for i in range(self.batch_size)]
                    source_length = torch.cuda.FloatTensor(source_length)
                    clean = clean.contiguous().unsqueeze(1)
                    mixture = mixture.view(-1,1, 32000)
                    outputs, mu, logsig, z = net(mixture)
                    loss = net.loss_function(outputs, clean, mu, logsig,source_length, self.alpha)
                    # loss1 = net.loss_function(outputs, clean, mu, logsig,source_length)
                    # outputs = outputs.contiguous().view(-1, 100, 320)
                    # loss2 = self.stftm_loss_function(outputs, clean)
                    # loss = 0.5*loss2 + 0.5*loss1

            
                self.writer.add_scalar('loss/train', loss, cnt)
                loss.backward()
                optimizer.step()

                running_loss = loss.item()
                accu_train_loss += running_loss

                cnt+=1

                del loss, outputs, mixture, clean
                end = timeit.default_timer()
                curr_time = start - end
                ttime += curr_time
                print('iter = {}/{}, epoch = {}/{}, loss = {:.5f}'.format(i+1, num_train_batches, epoch+1, self.max_epoch, running_loss))



                if i == num_train_batches-1:
                    start = timeit.default_timer()
                    avg_train_loss = accu_train_loss / cnt
                    self.logg_loss.append(avg_train_loss)
                    avg_eval_loss = self.validate(net, validationloader, epoch)

                    net.train()

                    print('Iteration/Epoch: {}/{}, average_loss: {}'.format(i, epoch, avg_eval_loss))
                    # print('Epoch [%d/%d], Iter [%d/%d]  ( TrainLoss: %.4f | EvalLoss: %.4f )' % (epoch+1,self.max_epoch,i+1,self.num_train_sentences//self.batch_size,avg_train_loss,avg_eval_loss))
                    is_best = True if avg_eval_loss < best_loss else False
                    best_loss = avg_eval_loss if is_best else best_loss
                    
                    checkpoint = Checkpoint(epoch, i, avg_train_loss, avg_eval_loss, best_loss, net.state_dict(), optimizer.state_dict())
                    

                    # model_path = os.path.join(Path.cwd(), 'checkpoint')
                    model_path = os.path.join('/home/ubuntu','checkpoint')
                    model_name = self.model_name + '_lastest_proposed.model'
                    best_model = self.model_name + '_best_proposed.model'

                    if not os.path.isdir(model_path):
                        print('create a checkpoint dir')
                        os.makedirs(model_path)

                    checkpoint.save(os.path.join(model_path, model_name), os.path.join(model_path, best_model), is_best)

                     
                    accu_train_loss = 0.0
                    
                    net.train()
                if (i+1)%num_train_batches == 0:
                    break

            trainloader.reset()
            if degrade_lr(self.logg_loss) and self.lr_schedule:
                print('decay learning a half')
                scheduler1.step()

        log_data = np.array(self.log_data)
        df = pd.DataFrame(columns=['epoch','pesq','stoi','si-sdr'], data=log_data)
        df.to_csv(self.log_file_path, index=False)

    def validate(self, net, validation_loader, epoch):
        print('**************** Starting evaluation model on validation set*******************')
        net.eval()
        stoi = []
        pesq = []
        sdr = []
        iteration = validation_loader.dataset_size -validation_loader.begin_idx
        with torch.no_grad():
            mtime = 0
            ttime = 0
            cnt = 0
            accu_eval_loss = 0.0
            for k in range(iteration):
                start = timeit.default_timer()

                mixture, clean = validation_loader.get_batch(1)
                mixture = mixture.to(self.device)
                clean = clean.to(self.device)

                mixture = mixture.unsqueeze(1)
                
                if self.model_name == 'tcnn':
                    output = net(mixture)
                    # source_length = [32000]
                    # source_length = torch.cuda.FloatTensor(source_length)
                    clean = clean.view(-1, 1, 32000)
                    output = output.view(-1, 1, 32000)
                    # loss,_,_,_ = cal_loss(clean, output, source_length) # input [B, C, T]
                    loss = mse_loss(output, clean)
                elif self.model_name == 'dense_tcnn':
                    ola_output, output = net(mixture)
                    ola_output = ola_output.squeeze(1)
                    # clean =self.ola(clean)
                    ola_output = self.sigtoframe(ola_output)[:,:-1,:]
                    output= output.view(-1, 100, 320)
                    loss = mse_loss(ola_output, clean)
                elif self.model_name=='conv_tasnet':
                    output = net(mixture)
                    source_length = [32000]
                    source_length = torch.cuda.FloatTensor(source_length)
                    clean = clean.view(-1, 1, 32000)
                    output = output.view(-1, 1, 32000)
                    loss,_,_,_ = cal_loss(clean, output, source_length) # input [B, C, T]
                elif self.model_name=='proposed':
                    source_length = [32000]
                    source_length = torch.cuda.FloatTensor(source_length)
                    clean = clean.unsqueeze(1)
                    mixture = mixture.view(-1,1, 32000)
                    output, mu, logsig, z = net(mixture)
                    loss = net.loss_function(output, clean, mu, logsig,source_length, self.alpha)
                    # loss1 = net.loss_function(output, clean, mu, logsig,source_length)
                    # output = output.contiguous().view(-1, 100, 320)
                    # loss2 = self.stftm_loss_function(output, clean)
                    # loss = 0.5*loss2 + 0.5*loss1
                    output = output.contiguous().view(1, 100, 320)

                accu_eval_loss += loss

                cnt+=1

                end = timeit.default_timer()
                curr_time = end - start
                ttime +=curr_time
                mtime = ttime/cnt
                curr_stoi, curr_pesq = cal_scores(clean.contiguous().view(-1).cpu().numpy(), output.contiguous().view(-1).cpu().numpy())
                curr_sdr = pytorch_si_sdr(output, clean, 2)
                stoi.append(curr_stoi)
                if not math.isnan(curr_pesq):
                    pesq.append(curr_pesq)
                sdr.append(curr_sdr)

        validation_loader.reset()
        avg_eval_loss = accu_eval_loss / cnt
        avg_stoi = sum(stoi) / len(stoi)
        avg_pesq = sum(pesq) / len(pesq)
        avg_sdr = sum(sdr) / len(sdr)
        
        self.writer.add_scalar('Pesq/Evaluation', avg_pesq, epoch)
        self.writer.add_scalar('STOI/Evaluation', avg_stoi, epoch)
        self.log_data.append([epoch, avg_pesq, avg_stoi, avg_sdr])

        net.train()
        print('avg_sdr: ', avg_sdr)
        print('##############################################################################')
        return avg_eval_loss




## loss function ##
def mse_loss(outputs, clean):
    loss = (outputs - clean)**2
    # loss.sum()
    return loss.mean()

# def stftm_loss():
#     return 3




############ TEST #########################################################
parser = argparse.ArgumentParser(description='argument for training tcnn')
parser.add_argument('--mixture-dir', type=str, default='/home/ubuntu/Speech_data/16k_mix')
parser.add_argument('--clean-dir', type=str, default='/home/ubuntu/Speech_data/16k_clean')
parser.add_argument('--csv-train-file', type=str, default='/home/ubuntu/Speech_data/train2.csv')
parser.add_argument('--csv-validation-file', type=str, default='/home/ubuntu/Speech_data/validation2.csv')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--model-path', type=str, default='/home/ubuntu/speeai-checkpoint')
parser.add_argument('--log-path', type=str, default='')
parser.add_argument('--resume-model', type=str, default=None)
parser.add_argument('--eval-steps', type=int, default=500)
parser.add_argument('--model-name', type=str, default='tcnn')
parser.add_argument('--z-dim', type=int, default=256)
parser.add_argument('--num-sam', type=int, default=2)
parser.add_argument('--note', type=str, default='')
parser.add_argument('--lr-schedule', type=bool, default=True)
parser.add_argument('--alpha', type=float, default=0.2)



args = parser.parse_args()
model = Model(args)
model.train(args)
model.writer.close()