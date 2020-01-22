import  torch
import torchvision
from torchvision import datasets,transforms,models
import os
import matplotlib.pyplot as plt

# get training and test data
def getdata(batch_size):
    data_dir='./data/char/'

    # tsf learning
    # data_transform={x:transforms.Compose([transforms.Resize([224,224]),
    #                                       transforms.ToTensor()])
    #                 for x in ['train','test']}

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
    # x,y=next(iter(data_loader['train']))
    # print(len(x),len(y))
    example_classes=image_datasets['train'].classes

    return data_loader,example_classes


# build model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1=torch.nn.Sequential(
            torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2,kernel_size=2)
        )
        self.dense=torch.nn.Sequential(
            torch.nn.Linear(16*16*128,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024,73)
        )

    def forward(self,x):
        x=self.conv1(x)
        x=x.view(-1,16*16*128)
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
    batch_size = 256
    epochs = 50
    gpu = 1
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    print(device)

    # get train and test data
    data_loader,example_classes=getdata(batch_size)
    print('train samples:',len( data_loader['train'].dataset))
    print('test  samples:', len(data_loader['test'].dataset))

    # # normal cnn
    model=Model().to(device)

    # # transfer learning from VGG16
    # model = models.vgg16(pretrained=True)
    # for parma in model.parameters():
    #     parma.requires_grad=False
    # model.classifier=torch.nn.Sequential(
    #     torch.nn.Linear(25088,4096),
    #     torch.nn.ReLU(),
    #     torch.nn.Dropout(p=0.5),
    #     torch.nn.Linear(4096,4096),
    #     torch.nn.Dropout(p=0.5),
    #     torch.nn.Linear(4096,2)
    # )
    # model=model.to(device)

    # # transfer learning from ResNet50
    # model=models.resnet50(pretrained=True)
    # for parma in model.parameters():
    #     parma.requires_grad=False
    # model.fc=torch.nn.Linear(2048,2)
    # model = model.to(device)

    cost = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.fc.parameters(),lr=0.00001)# fc -> classifier
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # train process
    losses=[]
    testacc=[]
    for epoch in range(epochs):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(data_loader['train']):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = cost(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader['train'].dataset),
                    batch_idx / len(data_loader['train']), loss.item()))
                losses.append(loss.item())
        acc=test(model,data_loader['test'])
        testacc.append(acc.double() / len(data_loader['test'].dataset))

    # model save and load
    torch.save(model.state_dict(), './log/snh_cnn.pkl')
    # model.load_state_dict(torch.load('params.pkl'))

    # plot loss
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('iterations')
    plt.ylabel('train loss')
    plt.savefig('./log/snhloss.png')

    # plot test acc
    plt.figure()
    plt.plot(range(len(testacc)), testacc)
    plt.xlabel('iterations')
    plt.ylabel('train loss')
    plt.savefig('./log/snhtestacc.png')

