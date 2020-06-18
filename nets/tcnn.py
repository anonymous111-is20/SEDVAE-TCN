import sys
# sys.path.append('/home/manhlt/VisualProject/speenai-is20')

import torch
import torch.nn as nn
from speenai.models.layers.residual_block import ResidualBlock, ResidualBlock2
from torchsummary import summary
from speenai.models.layers.tcn_block import TCN

# class DilationBlock(nn.Module):
#     def __init__(self):
#         self.block = nn.Sequential()


class TCNN(nn.Module):
    def __init__(self,
                batch_size,
                causal=True):
        super(TCNN, self).__init__()
        self.batch_size = batch_size
        # self.tcnn = nn.Module()

        self.padding = nn.ConstantPad2d((1,0, 0, 0), value=0) # left, right, top, bottom
        self.encoder_conv2d_1 = nn.Conv2d(1, 16, kernel_size=(2,5), stride=(1,1), padding=(1, 2))
        # self.padding_2 = nn.ConstantPad2d((2, 1, 1, 0), value=0) # left, right, top, bottom
        self.encoder_conv2d_2 = nn.Conv2d(16, 16, kernel_size=(2,5), stride=(1,2), padding=(1, 2))
        self.encoder_conv2d_3 = nn.Conv2d(16, 16, kernel_size=(2,5), stride=(1,2), padding=(1, 2))
        self.encoder_conv2d_4 = nn.Conv2d(16, 32, kernel_size=(2,5), stride=(1,2), padding=(1, 2))
        self.encoder_conv2d_5 = nn.Conv2d(32, 32, kernel_size=(2,5), stride=(1,2), padding=(1, 2))
        self.encoder_conv2d_6 = nn.Conv2d(32, 64, kernel_size=(2,5), stride=(1,2), padding=(1, 2))
        self.encoder_conv2d_7 = nn.Conv2d(64, 64, kernel_size=(2,5), stride=(1,2), padding=(1, 2))
        
        #TCM
        self.tcm = nn.Sequential(
            # first dilation block
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,1,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,2,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,4,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,8,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,16,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,32,1], causal=causal),
            # second dilation block
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,1,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,2,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,4,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,8,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,16,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,32,1], causal=causal),
            # third dilation block
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,1,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,2,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,4,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,8,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,16,1], causal=causal),
            ResidualBlock(in_channels=256, kernel=[1,3,1], dilation=[1,32,1], causal=causal),
        )
        # self.tcm = TCN(input_channel=1, hidden_channel=1)

        #decoder module
        self.decoder_deconv2d_7 = nn.ConvTranspose2d(128, 64, kernel_size=(2,5), stride=(1,1), padding=(0,0))
        self.decoder_deconv2d_6 = nn.ConvTranspose2d(128, 32, kernel_size=(2,5), stride=(1,2), padding=(0,0))
        self.decoder_deconv2d_5 = nn.ConvTranspose2d(64, 32, kernel_size=(2,5), stride=(1,2), padding=(0,0))
        self.decoder_deconv2d_4 = nn.ConvTranspose2d(64, 16, kernel_size=(2,5), stride=(1,2), padding=(0,0))
        self.decoder_deconv2d_3 = nn.ConvTranspose2d(32, 16, kernel_size=(2,5), stride=(1,2), padding=(0,0))
        self.decoder_deconv2d_2 = nn.ConvTranspose2d(32, 16, kernel_size=(2,5), stride=(1,2), padding=(0,0))
        self.decoder_deconv2d_1 = nn.ConvTranspose2d(16, 1, kernel_size=(2,5), stride=(1,1), padding=(0,0))

    def forward(self, x):
        # x = x.unsqueeze(0)
        # x = x.unsqueeze(1)

        encoder_1 = self.encoder_conv2d_1(x)[:,:,:-1]


        encoder_2 = self.encoder_conv2d_2(encoder_1)[:,:,:-1,:]
        encoder_3 = self.encoder_conv2d_3(encoder_2)[:,:,:-1,:-1]
        encoder_4 = self.encoder_conv2d_4(encoder_3)[:,:,:-1,:-1]
        encoder_5 = self.encoder_conv2d_5(encoder_4)[:,:,:-1,:-1]
        encoder_6 = self.encoder_conv2d_6(encoder_5)[:,:,:-1,:-1]
        encoder_7 = self.encoder_conv2d_7(encoder_6)[:,:,:-1,:-1]

        
        reshape = encoder_7.contiguous().view(256, -1)
        reshape = reshape.unsqueeze(0)

        tcm_output = self.tcm(reshape)
        # print("tcm_output shape: ", tcm_output.shape)
        tcm_output = tcm_output.reshape(-1, 64, 100, 4)
        # tcm_output = tcm_output.unsqueeze(0)
        
    
        # print('---------------------------------------------------------------------')
        zeros_padding = torch.zeros(1)

        # print('tcm output: ', tcm_output.shape)
        # print('decoder_7 shape: ', encoder_7.shape)
        decoder_7 = self.decoder_deconv2d_7(torch.cat((tcm_output, encoder_7),1))[:,:,:-1,:].contiguous()
        decoder_7 = self.padding(decoder_7)
        decoder_6 = self.decoder_deconv2d_6(torch.cat((decoder_7, encoder_6), 1))[:,:,:-1,:-2].contiguous()
        decoder_5 = self.decoder_deconv2d_5(torch.cat((decoder_6, encoder_5), 1))[:,:,:-1,:-2].contiguous()
        decoder_4 = self.decoder_deconv2d_4(torch.cat((decoder_5, encoder_4), 1))[:,:,:-1,:-2].contiguous()
        decoder_3 = self.decoder_deconv2d_3(torch.cat((decoder_4, encoder_3), 1))[:,:,:-1,:-1].contiguous()
        decoder_2 = self.decoder_deconv2d_2(torch.cat((decoder_3, encoder_2), 1))[:,:,:-1,:-3].contiguous()
        decoder_1 = self.decoder_deconv2d_1(decoder_2)[:,:,:-1,:-4].contiguous()

        output = decoder_1.squeeze(1)

        return output
    def set_bath_size(self, batch_size):
        self.batch_size = batch_size



model = TCNN(1, causal=False).cuda()
# model.eval()
print(summary(model, (1, 100,320)))

test_data = torch.randn(1, 1, 100, 320).cuda()
# test_data = test_data.unsqueeze(1)
output = model(test_data)

print(output.shape)
print(output)
# print(type(output))