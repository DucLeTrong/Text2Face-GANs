import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf, t_in, t_out=16):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(ndf * 8+1, 1, 4, 1, 0, bias=False)
        self.linear = nn.Linear(t_in, t_out)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        self.bn3 = nn.BatchNorm2d(ndf * 8)

    def forward(self, input):
        imgs, t_input = input
        # print (t_input.shape)
        b_s = imgs.shape[0]
        out = self.lrelu(self.conv1(imgs))
        out = self.lrelu(self.bn1(self.conv2(out)))
        out = self.lrelu(self.bn2(self.conv3(out)))
        out = self.lrelu(self.bn3(self.conv4(out)))
        t_tmp = self.lrelu(self.linear(t_input))
        t_tmp = torch.unsqueeze(t_tmp, 2)
        t_tmp = torch.unsqueeze(t_tmp, 3)
        t_tmp = t_tmp.reshape(b_s, -1, 4, 4)
        # print (t_tmp.shape)

        img_text_concat = torch.cat([out, t_tmp], axis=1)
        out = self.conv5(img_text_concat)
        # print (out.shape)

        return nn.Sigmoid()(out)


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
