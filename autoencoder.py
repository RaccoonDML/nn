import  torch
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# get training and test data
def getdata_mnist(batch_size):
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    data_train=datasets.MNIST("./data/mnist",transform=transform,train=True,download=True)
    data_test=datasets.MNIST("./data/mnist",transform=transform,train=False,download=True)

    data_loader_train=torch.utils.data.DataLoader(dataset=data_train,batch_size=batch_size,shuffle=True)
    data_loader_test=torch.utils.data.DataLoader(dataset=data_test,batch_size=batch_size,shuffle=True)

    return data_loader_train,data_loader_test

# build model
class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder=torch.nn.Sequential(
            torch.nn.Linear(28*28,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,32),
            torch.nn.ReLU()
        )
        self.decoder=torch.nn.Sequential(
            torch.nn.Linear(32,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,28*28)
        )

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

class AutoEncoderCnn(torch.nn.Module):
    def __init__(self):
        super(AutoEncoderCnn,self).__init__()
        self.encoder=torch.nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2,kernel_size=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.decoder=torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2,mode='nearest'),
            torch.nn.Conv2d(128,64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(64,1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

# if __name__ == '__main__':
#
#     # define hyper-parameters
#     batch_size = 4
#     epochs = 5
#     gpu = 0
#     device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and 0 else 'cpu')
#     print(device)
#
#     model = AutoEncoder().to(device)
#     cost = torch.nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters())
#
#     # get train and test data
#     data_loader_train,data_loader_test=getdata_mnist(batch_size)
#
#     # train process
#     for epoch in range(epochs):
#
#         for batch_idx, (data, target) in enumerate(data_loader_train):
#             data, target = data.to(device), target.to(device)
#             noisy_data=data+0.4*torch.randn(data.shape).to(device)
#             noisy_data = np.clip(noisy_data, 0., 1.)
#             data,noisy_data=Variable(data.view(-1,28*28)), Variable(noisy_data.view(-1,28*28))
#
#             train_pre=model(noisy_data)
#             loss=cost(train_pre,data)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             if batch_idx % 100 == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(
#                     epoch, batch_idx * len(data), len(data_loader_train.dataset),
#                     batch_idx / len(data_loader_train), loss.item()))

if __name__ == '__main__':

    # define hyper-parameters
    batch_size = 64
    epochs = 5
    gpu = 0
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    print(device)

    model = AutoEncoderCnn().to(device)
    cost = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # get train and test data
    data_loader_train,data_loader_test=getdata_mnist(batch_size)

    # train process
    for epoch in range(epochs):

        for batch_idx, (data, target) in enumerate(data_loader_train):
            # data, target = data.to(device), target.to(device)
            noisy_data=data+0.4*torch.randn(data.shape)
            noisy_data = np.clip(noisy_data, 0., 1.)
            data,noisy_data=Variable(data.to(device)), Variable(noisy_data.to(device))

            train_pre=model(noisy_data)
            loss=cost(train_pre,data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader_train.dataset),
                    batch_idx / len(data_loader_train), loss.item()))
        print('\n')