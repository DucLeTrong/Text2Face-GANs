from __future__ import print_function
#%matplotlib inline
import argparse
import os
import cv2
import shutil
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from dataset import FaceData
from dcgan import weights_init, Generator,Discriminator

# Root directory for dataset
dataroot = "/content/drive/MyDrive/Deep_Learning/projects/"
workers = 2
batch_size = 256
image_size = 64
nc = 3
nz = 100
t_in = 4800
ngf = 64
ndf = 64
num_epochs = 150
start_epoch = 101
lr = 0.0002
beta1 = 0.5
ngpu = 1

path_samples = '/content/drive/MyDrive/Deep_Learning/projects/dcgan/continue_train/samples/'
if os.path.exists(path_samples):
  shutil.rmtree(path_samples)
  os.makedirs(path_samples)
else:
  os.makedirs(path_samples)
dataset = FaceData(data_dir=dataroot,
                   transform=transforms.Compose([
                       transforms.Resize(image_size),
                       transforms.CenterCrop(image_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print (device)
# custom weights initialization called on netG and netD
# Create the generator
netG = Generator(ngpu, nz+t_in, ngf, nc).to(device)
netD = Discriminator(ngpu, nc, ndf, t_in).to(device)

net_g_path = '/content/drive/MyDrive/Deep_Learning/projects/dcgan/continue_train/checkpoints/netG_latest.pth'
net_d_path = '/content/drive/MyDrive/Deep_Learning/projects/dcgan/continue_train/checkpoints/netD_latest.pth'
netG.load_state_dict(torch.load(net_g_path))
netD.load_state_dict(torch.load(net_d_path))
print ("Done loading model!")
# Handle multi-gpu if desired
# if (device.type == 'cuda') and (ngpu > 1):
#     netG = nn.DataParallel(netG, list(range(ngpu)))
#     netD = nn.DataParallel(netD, list(range(ngpu)))
# netG.apply(weights_init)
# netD.apply(weights_init)
netG = netG.cuda()
netD = netD.cuda()

# Initialize BCELoss function
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz+t_in, 1, 1, device=device)

real_label = 1.
fake_label = 0.
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(start_epoch, num_epochs):
    for i, (img, t_v) in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu = img.to(device)
        t_v = t_v.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD([real_cpu, t_v]).view(-1)
        # print (output.shape, label.shape)
        errD_real = criterion(output, label)
        errD_real.backward()
        # D_x = output.mean().item()
        noise = torch.cat([torch.randn(b_size, nz, 1, 1, device=device), torch.unsqueeze(torch.unsqueeze(t_v, 2), 3)], 1).cuda()
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD([fake.detach(), t_v]).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        # D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        netG.zero_grad()
        label.fill_(real_label) 
        output = netD([fake, t_v]).view(-1)
        errG = criterion(output, label)
        errG.backward()
        # D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item()))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        #     # with torch.no_grad():
        #         # fake = netG(fixed_noise).detach().cpu()
            img = vutils.make_grid(fake.detach().cpu(), padding=2, normalize=True)
            img_list.append(img)
            img = (np.transpose(img.numpy(),(1,2,0))*255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(path_samples, 'fake_epoch{}.png'.format(epoch)), img)
            torch.save(netG.state_dict(), '/content/drive/MyDrive/Deep_Learning/projects/dcgan/continue_train/checkpoints/netG_latest.pth')
            torch.save(netD.state_dict(), '/content/drive/MyDrive/Deep_Learning/projects/dcgan/continue_train/checkpoints/netD_latest.pth')
        iters += 1
    if epoch % 10 == 0:
      torch.save(netG.state_dict(), f'/content/drive/MyDrive/Deep_Learning/projects/dcgan/continue_train/checkpoints/netG_epoch_{epoch}.pth')
      torch.save(netD.state_dict(), f'/content/drive/MyDrive/Deep_Learning/projects/dcgan/continue_train/checkpoints/netD_epoch_{epoch}.pth')