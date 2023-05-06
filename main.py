from __future__ import print_function

import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as vdataset
import torchvision.transforms as transfroms
import torchvision.utils as vutils


def get_dataloader(datapath,image_size,batch_size,workers=2):
    dataset=vdataset.ImageFolder(
        root=datapath,
        transform=transfroms.Compose([
            transfroms.Resize(image_size),
            transfroms.CenterCrop(image_size),
            transfroms.ToTensor(),
            transfroms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    )
    dataloader=torch.utils.data.DataLoader(
        dataset,batch_size=batch_size,shuffle=True,num_workers=workers
    )
    return dataloader



class Generator(nn.Module):
    def __init__(self,nz,nc,ngf,ngpu):
        super().__init__() 
        self.ngpu=ngpu
        self.main=nn.Sequential(
            nn.ConvTranspose2d(nz,ngf*8,4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf,nc,4,2,1,bias=False),
            nn.Tanh()
        )    
    def forward(self,x):
        return self.main(x)  

class Discriminator(nn.Module):
    def __init__(self,nc,ndf,ngpu):
        super().__init__()
        self.ngpu=ngpu
        self.main=nn.Sequential(
            nn.Conv2d(nc,ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(ndf*8,1,4,1,0,bias=False),
            nn.Sigmoid()
        )
    def forward(self,input):
        return self.main(input)

def train(num_epochs,dataloader,netD,netG,optimizerD,optimizerG,criterion,nz,fixed_noise,device):
    img_list=[]
    G_losses=[]
    D_losses=[]
    iters=0

    print('Starting Training Loop ...')
    for epoch in range(num_epochs):
        for i ,data in enumerate(dataloader,0):
            #training D network
            netD.zero_grad()
            real_cpu=data[0].to(device)
            b_size=real_cpu.size(0)
            label=torch.full((b_size,),1.,dtype=torch.float,device=device)
            output=netD(real_cpu).view(-1)
            errD_real=criterion(output,label)
            errD_real.backward()
            D_x=output.mean().item()

            noise=torch.randn(b_size,nz,1,1,device=device)
            fake=netG(noise)
            label.fill_(0.)
            output=netD(fake.detach()).view(-1)
            errD_fake=criterion(output,label)
            errD_fake.backward()
            D_G_z1=output.mean().item()
            
            errD=errD_real+errD_fake
            optimizerD.step()

            #Update G network
            netG.zero_grad()
            label.fill_(1.)
            output=netD(fake).view(-1)
            errG=criterion(output,label)
            errG.backward()
            D_G_z2=output.mean().item()

            optimizerG.step()

            if i%50==0:
                print('Epoch {}/{} ,{}/{}  Loss_D: {}  Loss_G :{} D(x): {}, D(G(z)):{}/{}'.format(
                    epoch,num_epochs,i,len(dataloader),errD.item(),errG.item(),D_x,D_G_z1,D_G_z2)
                )
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if iters%500 ==0 or ((epoch ==num_epochs-1) and (i==len(dataloader)-1)):
                with torch.no_grad():
                    fake=netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake,padding=2,normalization=True))
            iters+=1
    torch.save(netG.state_dict(),'/home/deeplearning/zxs/DCGAN/model/netG.pth')
    torch.save(netD.state_dict(),'/home/deeplearning/zxs/DCGAN/model/netD.pth')

    return img_list,D_losses,G_losses


def main():

    dataroot='../Data/celeba'
    workers=2
    batch_size=128
    image_size=64
    nc=3
    nz=100
    ngf=64
    ndf=64
    num_epochs=10
    lr=0.0002
    beta1=0.5
    ngpu=1

    device=torch.device('cuda:0' if (torch.cuda.is_available() and ngpu>0) else 'cpu')
    print('device:',device)
    dataloader=get_dataloader(dataroot,image_size,batch_size,workers)

    read_batch=next(iter(dataloader))
    print(read_batch[0].shape)
    # plt.figure(figsize=(8,8))
    # plt.axis('off')
    # plt.title('Training Images')
    # #显示需要维度改为 hwc才能用plt显示
    # plt.imshow(np.transpose(vutils.make_grid(read_batch[0].to(device)[:64],padding=2,normalize=True).cpu(),(1,2,0)))
    # plt.show()

    netG=Generator(nz,nc,ngf,ngpu).to(device)

    if (device.type=='cuda' and (ngpu>1)):
        netG=nn.DataParallel(netG,list(range(ngpu)))

    netG.apply(weights_init)
    # print(netG)

    netD=Discriminator(nc,ndf,ngpu).to(device)

    if (device.type=='cuda' and (ngpu>1)):
        netG=nn.DataParallel(netG,list(range(ngpu)))

    netD.apply(weights_init)
    # print(netD)

    criterion=nn.BCELoss()
    fixed_noise=torch.randn(64,nz,1,1,device=device)

    real_label=1.
    fake_label=0.
    optimizerD=optim.Adam(netD.parameters(),lr=lr,betas=(beta1,0.999))
    optimizerG=optim.Adam(netG.parameters(),lr=lr,betas=(beta1,0.999))

    img_list,D_loss,G_loss=train(num_epochs,dataloader,netD,netG,optimizerD,optimizerG,criterion,
        nz,fixed_noise,device)





if __name__=="__main__":
    main()