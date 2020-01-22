import  torch
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable

# get training and test data
def getdata_mnist(batch_size):
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    data_train=datasets.MNIST("./data/mnist",transform=transform,train=True,download=True)
    data_test=datasets.MNIST("./data/mnist",transform=transform,train=False,download=True)

    data_loader_train=torch.utils.data.DataLoader(dataset=data_train,batch_size=batch_size,shuffle=True)
    data_loader_test=torch.utils.data.DataLoader(dataset=data_test,batch_size=batch_size,shuffle=True)

    return data_loader_train,data_loader_test

# build model
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=torch.nn.RNN(
            input_size=28,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.output=torch.nn.Linear(128,10)

    def forward(self,input):
        output,_=self.rnn(input,None)
        output=self.output(output[:,-1,:])
        return output

def test(net_g, data_loader):
    # testing
    net_g.eval()
    correct = 0
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, 28, 28)
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
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and 0 else 'cpu')
    print(device)
    model = RNN().to(device)
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
            data=data.view(-1,28,28)
            optimizer.zero_grad()
            output = model(data)
            loss = cost(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader_train.dataset),
                    batch_idx / len(data_loader_train), loss.item()))
        test(model,data_loader_test)

    # model save and load
    torch.save(model.state_dict(), './log/rnn.pkl')
    # model.load_state_dict(torch.load('params.pkl'))

