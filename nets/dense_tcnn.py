import torch
import torch.nn as nn
from torchsummary import summary
from speenai.models.layers.dense_block import TorchOLA, SPConvTranspose2d, DenseBlock


class Dense_TCNN(nn.Module):

    def __init__(self, L=512, width=64, causal=True):
        super(Dense_TCNN, self).__init__()
        self.L = L
        self.frame_shift = self.L // 2
        # self.N = 256
        # self.B = 256
        # self.H = 512
        # self.P = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.in_channels = 1
        self.out_channels = 1
        self.kernel_size = (2, 3)
        self.elu = nn.SELU(inplace=True)
        self.pad1 = nn.ConstantPad2d((1,1,0,0), value=0.0)
        self.width = width
        self.causal = causal
        
        ############ Encoder Block ##############################################################
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1,1))
        self.inp_norm = nn.LayerNorm(self.L)
        self.inp_prelu = nn.PReLU(self.width)

        self.enc_dense1 = DenseBlock(self.L, 5, self.width, causal=self.causal)
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1,3), stride=(1,2))
        self.enc_norm1 = nn.LayerNorm(self.L//2)
        self.enc_prelu1 = nn.PReLU(self.width)

        self.enc_dense2 = DenseBlock(self.L//2, 5, self.width, causal=self.causal)
        self.enc_conv2 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1,3), stride=(1,2))
        self.enc_norm2 = nn.LayerNorm(self.L//4)
        self.enc_prelu2 = nn.PReLU(self.width)

        self.enc_dense3 = DenseBlock(self.L//4, 5, self.width, causal=self.causal)
        self.enc_conv3 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1,3), stride=(1,2))
        self.enc_norm3 = nn.LayerNorm(self.L//8)
        self.enc_prelu3 = nn.PReLU(self.width)

        self.enc_dense4 = DenseBlock(self.L//8, 5, self.width, causal=self.causal)
        self.enc_conv4 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1,3), stride=(1,2))
        self.enc_norm4 = nn.LayerNorm(self.L//16)
        self.enc_prelu4 = nn.PReLU(self.width)

        self.enc_dense5 = DenseBlock(self.L//16, 5, self.width, causal=self.causal)
        self.enc_conv5 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1,3), stride=(1,2))
        self.enc_norm5 = nn.LayerNorm(self.L//32)
        self.enc_prelu5 = nn.PReLU(self.width)

        self.enc_dense6 = DenseBlock(self.L//32, 5, self.width, causal=self.causal)
        self.enc_conv6 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1,3), stride=(1,2))
        self.enc_norm6 = nn.LayerNorm(self.L//64)
        self.enc_prelu6 = nn.PReLU(self.width)
        ################ End block ##############################################################################################

        ############### Decoder block ########################################################################################
        self.dec_dense6 = DenseBlock(self.L//64, 5, self.width)
        self.dec_conv6 = SPConvTranspose2d(in_channels=self.width*2, out_channels=self.width, kernel_size=(1,3), r=2)
        self.dec_norm6 = nn.LayerNorm(self.L//32)
        self.dec_prelu6 = nn.PReLU(self.width)

        self.dec_dense5 = DenseBlock(self.L//32, 5, self.width)
        self.dec_conv5 = SPConvTranspose2d(in_channels=self.width*2, out_channels=self.width, kernel_size=(1,3), r=2)
        self.dec_norm5 = nn.LayerNorm(self.L//16)
        self.dec_prelu5 = nn.PReLU(self.width)

        self.dec_dense4 = DenseBlock(self.L//16, 5, self.width)
        self.dec_conv4 = SPConvTranspose2d(in_channels=self.width*2, out_channels=self.width, kernel_size=(1,3), r=2)
        self.dec_norm4 = nn.LayerNorm(self.L//8)
        self.dec_prelu4 = nn.PReLU(self.width)

        self.dec_dense3 = DenseBlock(self.L//8, 5, self.width)
        self.dec_conv3 = SPConvTranspose2d(in_channels=self.width*2, out_channels=self.width, kernel_size=(1,3), r=2)
        self.dec_norm3 = nn.LayerNorm(self.L//4)
        self.dec_prelu3 = nn.PReLU(self.width)

        self.dec_dense2 = DenseBlock(self.L//4, 5, self.width)
        self.dec_conv2 = SPConvTranspose2d(in_channels=self.width*2,out_channels=self.width, kernel_size=(1,3), r=2)
        self.dec_norm2 = nn.LayerNorm(self.L//2)
        self.dec_prelu2 = nn.PReLU(self.width)

        self.dec_dense1 = DenseBlock(self.L//2, 5, self.width)
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width*2, out_channels=self.width, kernel_size=(1,3), r=2)
        self.dec_norm1 = nn.LayerNorm(self.L)
        self.dec_prelu1 = nn.PReLU(self.width)
        
        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1,1))
        self.ola = TorchOLA(self.frame_shift)

    def forward(self, x):

        enc_list = []
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))
        
        out = self.enc_dense1(out)
        out = self.enc_prelu1(self.enc_norm1(self.enc_conv1(self.pad1(out))))
        enc_list.append(out)
        
        out = self.enc_dense2(out)
        out = self.enc_prelu2(self.enc_norm2(self.enc_conv2(self.pad1(out))))
        enc_list.append(out)
        
        out = self.enc_dense3(out)
        out = self.enc_prelu3(self.enc_norm3(self.enc_conv3(self.pad1(out))))
        enc_list.append(out)
        
        out = self.enc_dense4(out)
        out = self.enc_prelu4(self.enc_norm4(self.enc_conv4(self.pad1(out))))
        enc_list.append(out)
        
        out = self.enc_dense5(out)
        out = self.enc_prelu5(self.enc_norm5(self.enc_conv5(self.pad1(out))))
        enc_list.append(out)
        
        out = self.enc_dense6(out)
        out = self.enc_prelu6(self.enc_norm6(self.enc_conv6(self.pad1(out))))
        enc_list.append(out)
        
        out = self.dec_dense6(out)
        out = torch.cat([out, enc_list[-1]], dim=1)
        out = self.dec_prelu6(self.dec_norm6(self.dec_conv6(self.pad1(out))))
        
        out = self.dec_dense5(out)
        out = torch.cat([out, enc_list[-2]], dim=1)
        out = self.dec_prelu5(self.dec_norm5(self.dec_conv5(self.pad1(out))))
        
        out = self.dec_dense4(out)
        out = torch.cat([out, enc_list[-3]], dim=1)
        out = self.dec_prelu4(self.dec_norm4(self.dec_conv4(self.pad1(out))))
        
        out = self.dec_dense3(out)
        out = torch.cat([out, enc_list[-4]], dim=1)
        out = self.dec_prelu3(self.dec_norm3(self.dec_conv3(self.pad1(out))))
        
        out = self.dec_dense2(out)
        out = torch.cat([out, enc_list[-5]], dim=1)
        out = self.dec_prelu2(self.dec_norm2(self.dec_conv2(self.pad1(out))))
        
        out = self.dec_dense1(out)
        out = torch.cat([out, enc_list[-6]], dim=1)
        out = self.dec_prelu1(self.dec_norm1(self.dec_conv1(self.pad1(out))))
        
        out_none_ola = self.out_conv(out)
        out = self.ola(out_none_ola)
        
        return out, out_none_ola


########### Test #######################################

# model = Dense_TCNN(L=320, causal=False).cuda()
# # # # print(model.parameters())
# # # # print(summary(model, (1,64,512)))
# data = torch.randn((1,1,100,320)).cuda()
# a, result = model(data)
# # # result = result.view(1,1,100, 320)
# # # print(result)
# # print(result.shape)
# print(a.shape)