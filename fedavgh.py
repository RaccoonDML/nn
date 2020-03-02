import numpy as np
import torch


def takeFirst(elem):
    return elem[0]
def calcNorm2(w):
    ss=[]
    # print('w[k]:',type(w[k]),'   len(w[k])   ',len(w[k]))
    for i in w.keys():
        # print('w[k][i]   ',type(w[k][i]),'   len(w[k][i])   ',len(w[k][i]))
        ss.append(torch.norm(w[i]))
    si=np.linalg.norm(ss)
    return si

# FedAvgH
def FedAvg_W(w):
    num=len(w)
    queue=[]
    for k in range(num):
        si=calcNorm2(w[k])
        queue.append((si,w[k]))
        print('***',si,' ',end=' ')
    print('----------------')

    while(len(queue)>1):
        queue.sort(key=takeFirst,reverse = True)
        print([i[0] for i in queue])
        w1 = queue.pop(0)[1]
        w2 = queue.pop(0)[1]
        # average w1 w2
        # print('w1:',w1)
        # print('w2:',w2)
        for k in w1.keys():
            w1[k]+=w2[k]
            w1[k] /= 2
        queue.append((calcNorm2(w1),w1))
        # print('1/2===',w1)

    w_avg=queue.pop(0)
    return w_avg[1]

a = torch.rand((2,3))
b=torch.rand((2,3))
c=torch.rand((2,3))
w1={'1':a,'2':a,'3':a}
w2={'1':b,'2':b,'3':b}
w3={'1':c,'2':c,'3':c}
ws=[w1,w2,w3]
print(w1,'\n',w2,'\n',w3)
print(FedAvg_W(ws))

