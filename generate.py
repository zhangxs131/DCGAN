import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pylab as plt
import numpy as np

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

def main():
    
    image_size=64
    nc=3
    nz=100
    ngf=64
    ndf=64
    num_epochs=10
    beta1=0.5
    ngpu=1
    batch_size=16

    device=torch.device('cuda:0' if (torch.cuda.is_available() and ngpu>0) else 'cpu')
    noise=torch.randn(batch_size,nz,1,1,device=device)

    netG=Generator(nz,nc,ngf,ngpu)
    netG.load_state_dict(torch.load('netG.pth',map_location=torch.device('cpu')))

    with torch.no_grad():
        fake=netG(noise).detach().cpu()
    print(fake.shape)
    img=vutils.make_grid(fake,padding=16,normalization=True)

    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title('Training Images')
    #显示需要维度改为 hwc才能用plt显示
    plt.imshow(np.transpose(vutils.make_grid(fake,padding=2,normalization=True).cpu(),(1,2,0)))
    plt.show()

if __name__=='__main__':
    main()


