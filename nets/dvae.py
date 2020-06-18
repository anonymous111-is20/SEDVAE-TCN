import utils, torch, time, os, pickle
import numpy as np 
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)



def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose1d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

class DVAE(nn.Module):

    def __init__(self, args):
        super(DVAE, self).__init__()
        #parameter
        self.batch_size = args.batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = args.model_name
        self.z_dim = args.z_dim
        self.num_sam = args.num_sam

        self.n_frames = 100
        self.frame_length = 320
        self.output_dim = 1
        self.input_dim = 1
        # network init
        self.encoder_init()
        self.decoder_init()

        self.reconstruction_function = nn.MSELoss().cuda()

        # fixed noise
        self.sample_z = Variable(torch.randn(self.batch_size, 1, self.z_dim).cuda(), volatile=True)

    def elbo(self, recon_x, x, mu, logsig):

        M, N, C, frame_length =  recon_x.shape
        x = x.contiguous().view(N*M, C, frame_length)
        recon_x = recon_x.contiguous().view(N *M, C, frame_length)

        MSE = self.reconstruction_function(recon_x, x) / N
        KLD_element = (logsig - mu**2 -torch.exp(logsig) + 1)
        KLD = -torch.mean(torch.sum(KLD_element*0.5), dim=-1)

        return MSE + KLD

    def loss_function(self, recon_x, x, mu, logsig):

        J_low  = self.elbo(recon_x, x, mu, logsig)
        return J_low
    
    def decoder_init(self):

        self.dec_layer1 = nn.Sequential(
            nn.Linear(self.z_dim, 128*(self.n_frames*self.frame_length // 4)),
            nn.BatchNorm1d(128 * (self.n_frames*self.frame_length//4)),
            nn.ReLU()
        )
        self.dec_layer2 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 2, 1),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.ConvTranspose1d(64, self.output_dim, 4, 2, 1),
            nn.PReLU(),
        )
        initialize_weights(self.dec_layer1)
        initialize_weights(self.dec_layer2)

    def encoder_init(self):

        self.enc_layer1 = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, 4, 2, 1),
            nn.PReLU(),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.PReLU(),
        )

        self.mu_fc = nn.Linear(128 * (self.n_frames*self.frame_length // 4), self.z_dim)
        self.sigma_fc = nn.Linear(128 * (self.n_frames*self.frame_length // 4), self.z_dim)

        initialize_weights(self.enc_layer1)
        initialize_weights(self.mu_fc)
        initialize_weights(self.sigma_fc)
    
    def encode(self, x):

        x = self.enc_layer1(x) # input (N, channels,n_frames*frame_length)
        print('x_shape1', x.shape)
        x = x.view(-1, 128*(self.n_frames*self.frame_length//4))
        
        # x = x.view(-1, 128*(self.n_frames//4)*(self.frame_length//4))
        # print('x_shape2', x.shape)

        mean = self.mu_fc(x)
        print('mean_shape', mean.shape)
        logsigma = self.sigma_fc(x)

        return mean, logsigma

    def decode(self, z):

        z = z.view(-1, self.z_dim)
        # print('z_shape: ', z.shape)
        x = self.dec_layer1(z)
        x = x.view(-1, 128, (self.n_frames//4)*(self.frame_length//4))
        x = self.dec_layer2(x)
        x = x.view(-1)[0:32000]
        # print('x_shape: ', x.shape)
        return x.view(-1, 1, self.n_frames*self.frame_length)

    def sample(self, mu, logsig):

        std = torch.exp(logsig*0.5)
        eps = torch.randn(std.size()).cuda()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def get_latent_sample(self, x):

        mu, logsig = self.encode(x)
        z = self.sample(mu, logsig)
        return z

    def forward(self, x, test=False):

        if self.model_name == 'DVAE' and not testF:
            eps = torch.randn(x.size()).cuda() * 0.025

        mu, logsig = self.encode(x)
        mu = mu.repeat(self.num_sam, 1)
        logsig = logsig.repeat(self.num_sam, 1)

        z = self.sample(mu, logsig)
        # print()
        res = self.decode(z)
        return res, mu, logsig, z

################ TEST ############################
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--batch-size', type=int, default=10)
# parser.add_argument('--model-name', type=str, default='test')
# parser.add_argument('--z-dim', type=int, default=256)
# parser.add_argument('--num-sam', type=int, default=2)

# args = parser.parse_args()

# dvae = DVAE(args).cuda()
# data = torch.randn(1,1,32000).cuda()
# output, mu, logsig, z = dvae(data)
# print(output)
# print(mu.shape)
# print(z.shape)