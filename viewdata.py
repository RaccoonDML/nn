import  torch
import torchvision
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# get training and test data
def getdata(batch_size):
    data_dir='./data/char/'

    data_transform={x:transforms.Compose([transforms.Resize([32,32]),
                                          transforms.ToTensor()])
                    for x in ['train','test']}
    image_datasets={x:datasets.ImageFolder(root=os.path.join(data_dir,x),
                                           transform=data_transform[x])
                    for x in ['train','test']}
    data_loader={x:torch.utils.data.DataLoader(dataset=image_datasets[x],
                                               batch_size=batch_size,
                                               shuffle=True)
                 for x in ['train','test']}

    example_classes=image_datasets['train'].classes
    return data_loader,example_classes

# get training and test data
def getdata_mnist(batch_size):
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    data_train=datasets.MNIST("./data/mnist",transform=transform,train=True,download=True)
    data_test=datasets.MNIST("./data/mnist",transform=transform,train=False,download=True)

    data_loader_train=torch.utils.data.DataLoader(dataset=data_train,batch_size=batch_size,shuffle=True)
    data_loader_test=torch.utils.data.DataLoader(dataset=data_test,batch_size=batch_size,shuffle=True)

    return data_loader_train,data_loader_test

# preview data
def preview_data(data_loader):
    images,labels=next(iter(data_loader))
    img = torchvision.utils.make_grid(images)
    img=img.numpy().transpose(1,2,0)
    std=[0.1307,]
    mean=[0.3180,]
    img=img*std+mean
    print([(labels[i]) for i in range(bs)])
    plt.imshow(img)
    plt.show()
    noisy_img=img+0.4*np.random.randn(*img.shape)
    noisy_img=np.clip(noisy_img,0.,1.)
    plt.imshow(noisy_img)
    plt.show()

bs=16
# data_loader,example_classes=getdata(bs)
train,test=getdata_mnist(bs)
preview_data(train)