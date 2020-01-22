import  torch
import torchvision
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
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

# preview data
def preview_data(data_loader):
    images,labels=next(iter(data_loader))
    img = torchvision.utils.make_grid(images)
    img=img.numpy().transpose(1,2,0)
    std=[0.5,0.5,0.5]
    mean=[0.5,0.5,0.5]
    img=img*std+mean
    print([example_classes[(labels[i])] for i in range(bs)])
    plt.imshow(img)
    plt.show()

bs=64
data_loader,example_classes=getdata(bs)
preview_data(data_loader['train'])