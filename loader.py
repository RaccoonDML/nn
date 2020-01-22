import os
import random,shutil

# balance the training set and build test set
path='E:\\AnotherC\\desktop\\Python\\nn\\data\\char\\train'
pathtest="E:\\AnotherC\\desktop\\Python\\nn\\data\\char\\test"

label_len={}
for label in os.listdir(path):
    newpath = os.path.join(path, label)
    length=len(os.listdir(newpath))
    label_len[label]=length
label_len = dict(sorted(label_len.items(), key=lambda x:x[1]))

tlabel_len={}
for label in os.listdir(pathtest):
    newpath = os.path.join(pathtest, label)
    length=len(os.listdir(newpath))
    tlabel_len[label]=length
tlabel_len = dict(sorted(tlabel_len.items(), key=lambda x:x[1]))

tottrain=0
tottest=0
print('*'*10+'training set'+'*'*10)
for i,j in label_len.items():
    print('{:15s}:{}'.format(i,j))
    tottrain+=j
print('\n'+'*'*10+'test set'+'*'*10)
for i,j in tlabel_len.items():
    print('{:15s}:{}'.format(i,j))
    tottest+=j
print('\ntotal label:{}'.format(len(label_len)))
print('train set:{}'.format(tottrain))
print('test  set:{}'.format(tottest))
# for i,j in label_len.items():
#     print('{:15s}:{}'.format(i,j))
#     if j<1000:
#         test=j//10
#     else:
#         test=100
#     newpath = os.path.join(path, i)
#     filelist=os.listdir(newpath)
#     movlist=random.sample(filelist,test)
#
#     dst_path=os.path.join("E:\\AnotherC\\desktop\\Python\\nn\\data\\char\\test", i)
#     for file in movlist:
#         f_src = os.path.join(newpath, file)
#         if not os.path.exists(dst_path):
#             os.mkdir(dst_path)
#         f_dst = os.path.join(dst_path, file)
#         print(f_src,f_dst)
        # shutil.move(f_src, f_dst)

    # delete to 5000
    # if(j<=5000):
    #     continue
    # else:
    #     newpath = os.path.join(path, i)
    #     filelist=os.listdir(newpath)
    #     dellist=random.sample(filelist,j-5000)
    #     print('*****del******:{}'.format(len(dellist)))
    #     for d in dellist:
    #         os.remove(os.path.join(newpath,d))








