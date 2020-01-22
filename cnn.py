import  torch
import torchvision
from torchvision import datasets,transforms

# get training and test data
def getdata_mnist(batch_size):
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    data_train=datasets.MNIST("./data/mnist",transform=transform,train=True,download=True)
    data_test=datasets.MNIST("./data/mnist",transform=transform,train=False,download=True)

    data_loader_train=torch.utils.data.DataLoader(dataset=data_train,batch_size=batch_size,shuffle=True)
    data_loader_test=torch.utils.data.DataLoader(dataset=data_test,batch_size=batch_size,shuffle=True)

    return data_loader_train,data_loader_test

# torchvision 0.2.0 do not support emnist
def getdata_emnist(batch_size):
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_train=datasets.EMNIST("./data/emnist",split='balanced',transform=transform,train=True,download=True)
    data_test=datasets.EMNIST("./data/emnist",split='balanced',transform=transform,train=False,download=True)

    data_loader_train=torch.utils.data.DataLoader(dataset=data_train,batch_size=batch_size,shuffle=True)
    data_loader_test=torch.utils.data.DataLoader(dataset=data_test,batch_size=batch_size,shuffle=True)

    return data_loader_train,data_loader_test


# build model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1=torch.nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2,kernel_size=2)
        )
        self.dense=torch.nn.Sequential(
            torch.nn.Linear(14*14*128,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024,10)
        )

    def forward(self,x):
        x=self.conv1(x)
        x=x.view(-1,14*14*128)
        x=self.dense(x)
        return x

def test(net_g, data_loader):
    # testing
    net_g.eval()
    correct = 0
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        log_probs = net_g(data)
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    print('\nTest Accuracy: {}/{} ({:.2%})\n'.format(
        correct, len(data_loader.dataset),
        correct.double() / len(data_loader.dataset)))

    return correct

if __name__ == '__main__':

    # define hyper-parameters
    batch_size = 64
    epochs = 5
    gpu = 0
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Model().to(device)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # get train and test data
    data_loader_train,data_loader_test=getdata_mnist(batch_size)
    print('train samples:',len( data_loader_train))
    # train process
    for epoch in range(epochs):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(data_loader_train):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = cost(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader_train.dataset),
                    batch_idx / len(data_loader_train), loss.item()))
        _=test(model,data_loader_test)

    # model save and load
    torch.save(model.state_dict(), './log/cnn_mnist.pkl')
    # model.load_state_dict(torch.load('params.pkl'))

