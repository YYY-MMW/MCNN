import numpy as np
from den_map import *
import torch

#数据加载函数,图片大小不同，批训练
def Loader(type,path,ipath,g_size,num):

    if type==0:
        tar = np.random.randint(1, num[0]+1, 1)
        now_path = path[0] + str(tar[0]) + '.mat'
        now_ipath = ipath[0] + str(tar[0]) + '.jpg'
    elif type==1:
        tar = np.random.randint(1, num[1]+1, 1)
        now_path = path[1] + str(tar[0]) + '.mat'
        now_ipath = ipath[1] + str(tar[0]) + '.jpg'

    data,num = Den_map(path=now_path,ipath=now_ipath,R=g_size)
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float() #float32类型tensor
    im = torch.from_numpy(cv2.imread(now_ipath)).unsqueeze(0).float()
    img = im.transpose(2, 3).transpose(1, 2)
    return img,data,num