# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision import datasets
from torchvision import utils as vutils
import matplotlib.pyplot as plt
import numpy as np

"""## 设定参数"""

PATH_DATASETS = "" # 预设路径
BATCH_SIZE = 128  # 批量
image_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

"""## 定义神经网路参数"""

nz = 100  # 生成神经网路杂讯维度
ngf = 64  # 生成神经网路滤波器个数
ndf = 64  # 判别神经网路滤波器个数
nc = 3    # 颜色通道

"""## 载入名人脸部资料集"""

# 转换
transform=transforms.Compose([
   transforms.Resize(image_size),
   transforms.CenterCrop(image_size),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 训练资料
dataset = datasets.ImageFolder(root='dataset', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 显示图档
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64]
                     , padding=2, normalize=True).cpu(),(1,2,0)));

"""## 神经网路权重初始值"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02) # 卷积层权重初始值
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02) # Batch Normalization 层权重初始值
        m.bias.data.fill_(0)

"""## 定义生成神经网路"""

class Generator(nn.Module):
    def __init__(self, nc=3, nz=100, ngf=ngf):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

netG = Generator().to(device)
netG.apply(weights_init)

"""## 定义判别神经网路"""

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

netD = Discriminator().to(device)
netD.apply(weights_init)

"""## 设定损失函数、优化器(optimizer)"""

# 设定损失函数
criterion = nn.BCELoss()

# 设定优化器(optimizer)
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

"""## 进行模型训练"""

# Commented out IPython magic to ensure Python compatibility.
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1.0
fake_label = 0.0
niter = 200
# 模型训练
for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        ########################################################
        # (1) 判别神经网路: maximize log(D(x)) + log(1 - D(G(z)))
        #######################################################
        # 训练真实资料
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # 训练假资料
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ########################################################
        # (2) 判别神经网路: maximize log(D(G(z)))
        #######################################################
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        if i % 200 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                   % (epoch+1, niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            vutils.save_image(real_cpu,'gan_fashion_output/real_samples.png' ,normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),'gan_fashion_output/fake_samples_epoch_%03d.png'
                              % (epoch), normalize=True)
    torch.save(netG.state_dict(), 'gan_fashion_weight/netG_epoch_%d.pth' % (epoch))
    torch.save(netD.state_dict(), 'gan_fashion_weight/netD_epoch_%d.pth' % (epoch))

"""## 新资料预测"""

batch_size = 25
latent_size = 100

fixed_noise = torch.randn(batch_size, latent_size, 1, 1).to(device)
# 产生图像，clamp 使像素值介于 [-1, 1] 之间
fake_images = netG(fixed_noise).clamp(min=-1, max=1)
fake_images_np = fake_images.cpu()
fake_images_np = fake_images_np.reshape(-1, 3, image_size, image_size)
fake_images_np = torch.permute(fake_images_np, (0, 2, 3, 1)).detach().numpy()
fake_images_np = (fake_images_np + 1) *.5  # 使像素值介于 [0, 1] 之间
R, C = 5, 5
plt.figure(figsize=(8, 8))
for i in range(batch_size):
    plt.subplot(R, C, i + 1)
    plt.axis('off')
    plt.imshow(fake_images_np[i])
plt.show();

"""## 显示 GIF 档"""

import imageio
import glob

# 产生 GIF 档
anim_file = './gan_fashion_output/dcgan.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('./gan_fashion_output/fake_samples*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

"""<img src="./gan_fashion_output/dcgan.gif" align="left">"""

# 载入权重
# netD.load_state_dict(torch.load('gan_pokemon_weights/netD_epoch_9.pth'))
# netG.load_state_dict(torch.load('gan_pokemon_weights/netG_epoch_9.pth'))