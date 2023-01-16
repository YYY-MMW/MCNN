#! /usr/bin/python

import numpy as np
import torch
import cv2
from nets import *
from den_map import *
from Loader import *
from evaluator import *
import sys

#设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

channel_list_L = [16,32,16,8]
ker_list_L = [9,7,7,7]

channel_list_M = [20, 40, 20, 10]
ker_list_M = [7, 5, 5, 5]

channel_list_S = [24, 48, 24, 12]
ker_list_S = [5,3,3,3]

#初始化
#密集人群、稀疏人群
path = ['./images/ShanghaiTech/part_A/train_data/ground-truth/GT_IMG_',\
        './images/ShanghaiTech/part_B/train_data/ground-truth/GT_IMG_']

ipath = ['./images/ShanghaiTech/part_A/train_data/images/IMG_',\
        './images/ShanghaiTech/part_B/train_data/images/IMG_']


#训练函数
def trainer(epoch,net,name,lr,times=700,path=path,ipath=ipath):
        optimizer = torch.optim.Adam(net.parameters(),lr=lr)
        loss_func = torch.nn.MSELoss(reduce=True, size_average=False)
        loss_data = []


        for j in range(epoch):
                for i in range(times):
                        tmp = np.random.randint(1, 8, 1)
                        if tmp[0] <= 4:
                                type = 0
                        elif tmp[0] > 4:
                                type = 1
                        # 加载数据
                        img, data, n = Loader(type=type, path=path, ipath=ipath, g_size=15,num=[300,400])        #注意高斯模糊尺寸，对应网络填充
                        img = img.to(device)
                        data = (data).to(device)
                        out = net(img)
                        out = torch.mean(out,1).unsqueeze(0)
                        loss = loss_func(out, data)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss_data.append(loss.item())
                        sys.stdout.flush()
                        print('epoch:' + str(j) + '    times:' + str(i) + '    MSE loss:' + str(loss.item()) + \
                              '    pre:' + str(out.sum().item()/2) + '    real:' + str(n))

        torch.save(net,'./data/'+name+'.pkl')
        sys.stdout.flush()
        print('训练结束 数据文件已保存为'+name+'.pkl')

'''
#创建子列
#CNN_L = torch.load('./data/CNN_L.pkl').to(device)
CNN_L = CNN(channel_list=channel_list_L,ker_list=ker_list_L).to(device)
#CNN_M = torch.load('./data/CNN_M.pkl').to(device)
CNN_M = CNN(channel_list=channel_list_M,ker_list=ker_list_M).to(device)
#CNN_S = torch.load('./data/CNN_S.pkl').to(device)
CNN_S = CNN(channel_list=channel_list_S,ker_list=ker_list_S).to(device)

trainer(epoch=20,times=700,net=CNN_L,name='CNN_L',path=path,ipath=ipath,lr=0.0001)
print('CNN_L:')
Evaluator(CNN_L)
trainer(epoch=20,times=700,net=CNN_M,name='CNN_M',path=path,ipath=ipath,lr=0.001)
print('CNN_M:')
Evaluator(CNN_M)
trainer(epoch=20,times=700,net=CNN_S,name='CNN_S',path=path,ipath=ipath,lr=0.001)
print('CNN_S:')
Evaluator(CNN_S)


#创建MCNN
MCNN = Mcnn().to(device)
trainer(epoch=400,times=700,net=MCNN,name='MCNN',path=path,ipath=ipath,lr=0.0001)
Evaluator(MCNN)
'''

img,data,n = Loader(1,path,ipath,15,[300,400])

MCNN = torch.load('./data/MCNN.pkl').to(device)

