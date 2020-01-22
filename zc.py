import  torch
import torchvision
from torchvision import datasets,transforms,models
import os
import matplotlib.pyplot as plt
from PIL import Image

# losses=[0.786736,0.764545,0.654252,0.612342,0.523445]
# plt.figure()
# plt.plot(range(len(losses)), losses)
# plt.xlabel('iterations')
# plt.ylabel('train loss')
# plt.savefig('./log/snh.png')

def test(net_g, data_loader):
    # testing
    net_g.eval()
    correct = 0
    for idx, (data, target) in enumerate(data_loader):
        data, target = data, target
        log_probs = net_g(data)
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    print('\nTest Accuracy: {}/{} ({:.2%})\n'.format(
        correct, len(data_loader.dataset),
        correct.double() / len(data_loader.dataset)))

    return correct

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


model=Model()
model.load_state_dict(torch.load('snh_cnn50.pkl'))
model.eval()

path = ''
preprocess_transform = transforms.Compose([transforms.Resize([32,32]),transforms.ToTensor()])

data_loader,class_names=getdata(64)

image_PIL = Image.open(path)
image_tensor = preprocess_transform(image_PIL)
image_tensor.unsqueeze_(0)

out = model(image_tensor)
# 得到预测结果，并且从大到小排序
_, indices = torch.sort(out, descending=True)
# 返回每个预测值的百分数
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

print([(class_names[idx], percentage[idx].item()) for idx in indices[0][:5]])