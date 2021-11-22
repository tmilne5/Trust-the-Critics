import torch
from torch import nn

class Generator(nn.Module):#added bias = False to each module followed by a batchnorm, as batchnorm will send mean to 0, effectively cancelling any learned bias
    def __init__(self, DIM, num_chan, h, w):
        """super(Generator, self).__init__() 
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM, bias = False),#latent dimension is 128
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True)
        )

        block1 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(4 * DIM, 2 * DIM, kernel_size=3, stride = 1, bias = False, padding = 0),
            #nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2, bias = False),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            #nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2, bias = False),
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(2 * DIM, DIM, kernel_size = 3, stride = 1, bias = False, padding = 0),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(DIM, num_chan, kernel_size = 3, stride = 1, bias = False, padding = 0))"""

        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM, bias = False),#latent dimension is 128
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True)
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2, bias = False),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2, bias = False),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, num_chan, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()
        self.DIM = DIM
        self.num_chan = num_chan

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, self.num_chan, 32, 32)

class Discriminator(nn.Module):
    def __init__(self, DIM, num_chan, h, w):
        super(Discriminator, self).__init__()
       
        self.DIM = DIM
        self.final_h = h//8
        self.final_w = w//8
        main = nn.Sequential(nn.Conv2d(num_chan, DIM, 3, 2,padding = 1),
                nn.LeakyReLU(),
                nn.Conv2d(DIM, 2*DIM, 3, 2, padding = 1),
                nn.LeakyReLU(),
                nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding = 1),
                nn.LeakyReLU())
        self.main = main
        self.linear = nn.Linear(self.final_h*self.final_w*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self.final_h*self.final_w*4*self.DIM)
        output = self.linear(output)
        return output

