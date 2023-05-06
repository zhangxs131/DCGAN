import torchvision.datasets as vdataset
import torchvision.transforms as transfroms
import torch

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


t=get_dataloader('cifar10',64,8)
print(t.shape)