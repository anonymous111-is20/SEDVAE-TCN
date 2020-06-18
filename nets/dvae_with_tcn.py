#import utils, torch, time, os, pickle
import numpy as np 
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from speenai.models.layers.tcn_block import TCN
from speenai.models.layers.dense_block import SPConvTranspose1d
import torch
from speenai.evals.si_snr import cal_loss
from speenai.evals.loss_function import stftm_loss
from torchsummary import summary

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)



def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.002)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.002)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.002)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0, 0.002)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose1d):
            m.weight.data.normal_(0, 0.002)
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
        self.reconstruction_function = nn.MSELoss().to(self.device)
        self.input_length = self.n_frames*self.frame_length
        # network init
        self.encoder_init()
        self.decoder_init()
        self.out_conv = nn.Conv1d(1, 1, kernel_size=1)

        

        # fixed noise
        self.sample_z = Variable(torch.randn(self.batch_size, 1, self.z_dim).cuda(), volatile=True)

    def elbo(self, recon_x, x, mu, logsig):

        N, C, frame_length =  recon_x.shape
        x = x.contiguous().view(N*1, C, frame_length)
        recon_x = recon_x.contiguous().view(N *1, C, frame_length)

        MSE = self.reconstruction_function(recon_x, x) / N
        KLD_element = (logsig - mu**2 -torch.exp(logsig) + 1)
        # KLD = -torch.mean(torch.sum(KLD_element*0.5), dim=-1)

        return MSE + KLD

    def loss_function(self, recon_x, x, mu, logsig, source_length, alpha):
        

        
        recon_x = recon_x.view(-1, 1, 32000)
        x = x.view(-1,1,32000)
        J_low  = self.elbo(recon_x, x, mu, logsig)
        si_snr,_,_,_ = cal_loss(x, recon_x, source_length)

        # stftmloss = stftm_loss(recon_x, x, device=torch.device('cuda'))

        # return J_low
        return alpha*J_low + (1-alpha)*si_snr
        # return J_low
    
    def decoder_init(self):

        self.dec_fc1 = nn.Linear(self.z_dim, 1000)
        self.dec_norm6 = nn.BatchNorm1d(1000)
        self.dec_prelu6 = nn.PReLU()

        self.dec_upconv5 = SPConvTranspose1d(in_channels=4*2, out_channels=8, kernel_size=1, r=2)
        self.dec_norm5 = nn.BatchNorm1d(8)
        self.dec_prelu5 = nn.PReLU()

        self.dec_upconv4 = SPConvTranspose1d(in_channels=8*2, out_channels=16, kernel_size=1, r=2)
        self.dec_norm4 = nn.BatchNorm1d(16)
        self.dec_prelu4 = nn.PReLU()

        self.dec_upconv3 = SPConvTranspose1d(in_channels=16*2, out_channels=32, kernel_size=1, r=2)
        self.dec_norm3 = nn.BatchNorm1d(32)
        self.dec_prelu3 = nn.PReLU()

        self.dec_upconv2 = SPConvTranspose1d(in_channels=32*2, out_channels=64, kernel_size=1, r=2)
        self.dec_norm2 = nn.BatchNorm1d(64)
        self.dec_prelu2 = nn.PReLU()

        self.dec_upconv1 = SPConvTranspose1d(in_channels=64*2, out_channels=1, kernel_size=1, r=2)
        self.dec_norm1 = nn.BatchNorm1d(1)
        self.dec_prelu1 = nn.PReLU()

        initialize_weights(self)

    def encoder_init(self):

        self.enc_conv1 = nn.Conv1d(self.input_dim, 64,4,2,1)
        self.enc_prelu1 = nn.PReLU()
        self.enc_tcn1 = TCN(64, 64, 64, 64, layer=8, stack=3, kernel=3, skip=True, causal=False)
        
        self.enc_conv2 = nn.Conv1d(64, 32, 4, 2, 1)
        self.enc_norm2 = nn.BatchNorm1d(32)
        self.enc_prelu2 = nn.PReLU()
        self.enc_tcn2 = TCN(32, 32, 32, 32, layer=8, stack=3, kernel=3, skip=True, causal=False)

        self.enc_conv3 = nn.Conv1d(32, 16, 4, 2, 1)
        self.enc_norm3 = nn.BatchNorm1d(16)
        self.enc_prelu3 = nn.PReLU()
        self.enc_tcn3 = TCN(16, 16, 16, 16, layer=8, stack=3, kernel=3, skip=True, causal=False)

        self.enc_conv4 = nn.Conv1d(16, 8, 4, 2, 1)
        self.enc_norm4 = nn.BatchNorm1d(8)
        self.enc_prelu4 = nn.PReLU()
        self.enc_tcn4 = TCN(8, 8, 8, 8, layer=8, stack=3, kernel=3, skip=True, causal=False)

        self.enc_conv5 = nn.Conv1d(8, 4, 4, 2, 1)
        self.enc_norm5 = nn.BatchNorm1d(4)
        self.enc_prelu5 = nn.PReLU()
        self.enc_tcn5 = TCN(4, 4, 4, 4, layer=8, stack=3, kernel=3, skip=True, causal=False)

        self.mu_fc = nn.Linear(1000, self.z_dim)
        self.sigma_fc = nn.Linear(1000, self.z_dim)

        initialize_weights(self)
    

    def sample(self, mu, logsig):

        std = torch.exp(logsig*0.5)
        eps = torch.randn(std.size()).to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def get_latent_sample(self, x):

        mu, logsig = self.encode(x)
        z = self.sample(mu, logsig)
        # return z
        return mu

    def forward(self, x, test=False):

        ########################### Encoder #######################################
        enc_conv1 = self.enc_conv1(x)
        enc_prelu1 = self.enc_prelu1(enc_conv1)
        enc_tcn1 = self.enc_tcn1(enc_prelu1)

        enc_conv2 = self.enc_conv2(enc_prelu1)
        enc_norm2 = self.enc_norm2(enc_conv2)
        enc_prelu2 = self.enc_prelu2(enc_norm2)
        enc_tcn2 = self.enc_tcn2(enc_prelu2)

        enc_conv3 = self.enc_conv3(enc_prelu2)
        enc_norm3 = self.enc_norm3(enc_conv3)
        enc_prelu3 = self.enc_prelu3(enc_norm3)
        enc_tcn3 = self.enc_tcn3(enc_prelu3)

        enc_conv4 = self.enc_conv4(enc_prelu3)
        enc_norm4 = self.enc_norm4(enc_conv4)
        enc_prelu4 = self.enc_prelu4(enc_norm4)
        enc_tcn4 = self.enc_tcn4(enc_prelu4)

        enc_conv5 = self.enc_conv5(enc_prelu4)
        enc_norm5 = self.enc_norm5(enc_conv5)
        enc_prelu5 = self.enc_prelu5(enc_norm5)
        enc_tcn5 = self.enc_tcn5(enc_prelu5)

        enc_reshape = enc_prelu5.view(-1, 1000)

        mean = self.mu_fc(enc_reshape)
        logsig = self.sigma_fc(enc_reshape)
        z = self.sample(mean, logsig)
        ########################## Decoder ########################################
        # print('enc_reshape: ', enc_reshape.shape)
        # print('mean shape: ', mean.shape)
        # print('logsig shape: ', mean.shape)
        # print('latent shape: ', z.shape)

        dec_fc1 = self.dec_fc1(z)
        dec_norm6 = self.dec_norm6(dec_fc1)
        dec_prelu6 = self.dec_prelu6(dec_norm6)
        dec_reshape = dec_prelu6.view(-1, 4, 1000)


        input_dec_upconv5 = torch.cat([dec_reshape, enc_tcn5], dim=1)
        # input_dec_upconv5 = torch.cat([dec_reshape, enc_prelu5], dim=1)
        dec_upconv5 = self.dec_upconv5(input_dec_upconv5)
        dec_norm5 = self.dec_norm5(dec_upconv5)
        dec_prelu5 = self.dec_prelu5(dec_norm5)

        input_dec_upconv4 = torch.cat([dec_prelu5, enc_tcn4], dim=1)
        # input_dec_upconv4 = torch.cat([dec_prelu5, enc_prelu4], dim=1)
        dec_upconv4 = self.dec_upconv4(input_dec_upconv4)
        dec_norm4 = self.dec_norm4(dec_upconv4)
        dec_prelu4 = self.dec_prelu4(dec_norm4)

        input_dec_upconv3 = torch.cat([dec_prelu4, enc_tcn3], dim=1)
        # input_dec_upconv3 = torch.cat([dec_prelu4, enc_prelu3], dim=1)
        dec_upconv3 = self.dec_upconv3(input_dec_upconv3)
        dec_norm3 = self.dec_norm3(dec_upconv3)
        dec_prelu3 = self.dec_prelu3(dec_norm3)

        input_dec_upconv2 = torch.cat([dec_prelu3, enc_tcn2], dim=1)
        # input_dec_upconv2 = torch.cat([dec_prelu3, enc_prelu2], dim=1)
        dec_upconv2 = self.dec_upconv2(input_dec_upconv2)
        dec_norm2 = self.dec_norm2(dec_upconv2)
        dec_prelu2 = self.dec_prelu2(dec_norm2)

        input_dec_upconv1 = torch.cat([dec_prelu2, enc_tcn1], dim=1)
        # input_dec_upconv1 = torch.cat([dec_prelu2, enc_prelu1], dim=1)
        dec_upconv1 = self.dec_upconv1(input_dec_upconv1)
        dec_norm1 = self.dec_norm1(dec_upconv1)
        dec_prelu1 = self.dec_prelu1(dec_norm1)

        out = self.out_conv(dec_prelu1)

        return out, mean, logsig, z

        
################ TEST ############################
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--batch-size', type=int, default=10)
# parser.add_argument('--model-name', type=str, default='test')
# parser.add_argument('--z-dim', type=int, default=256)
# parser.add_argument('--num-sam', type=int, default=2)

# args = parser.parse_args()

# dvae = DVAE(args).cuda()
# print(summary(dvae, (1,32000)))
# data = torch.randn(10,1,32000).cuda()
# output, mu, logsig, z = dvae(data)
# print(output)
# print(mu.shape)
# print(z.shape)