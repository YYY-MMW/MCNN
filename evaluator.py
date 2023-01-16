import torch
import numpy as np
import sys
from Loader import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_path = ['./images/ShanghaiTech/part_A/test_data/ground-truth/GT_IMG_',\
        './images/ShanghaiTech/part_B/test_data/ground-truth/GT_IMG_']

test_ipath = ['./images/ShanghaiTech/part_A/test_data/images/IMG_',\
        './images/ShanghaiTech/part_B/test_data/images/IMG_']


def Evaluator(net,path=test_path,ipath=test_ipath,model= 2):
    MAE = 0
    MSE = 0
    for i in range(5):
        if model==2:
            tmp = np.random.randint(1, 8, 1)
            if tmp[0] <= 4:
                type = 0
            elif tmp[0] > 4:
                type = 1
        elif model==0:
            type = 0
        elif model==1:
            type=1
        # 加载数据
        img, data, n = Loader(type=type, path=path, ipath=ipath, g_size=15,num=[182,316])  # 注意高斯模糊尺寸，对应网络填充
        img = img.to(device)
        data = data.to(device)
        out = net(img)
        out = out.squeeze()
        out = torch.mean(out, 1).unsqueeze(0)
        out_n = torch.sum(out)
        data = data.squeeze()
        MAE += abs(out_n-n)
        MSE += (out_n-n)**2
        sys.stdout.flush()
    print('====================================================')
    print('MAE: %.1f   MSE: %.1f '%((MAE.item() / (i + 1)),np.sqrt(MSE.item() / (i + 1))))
    print('====================================================')



