import numpy as np
import scipy.io as sio
import cv2
import os

#高斯核
def gauss(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center

            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]

    kernel = kernel / sum_val

    return kernel

def Den_map(path,ipath,R):
    # 人头标记
    data = sio.loadmat(path)  # 为一个字典
    P = data['image_info']
    n = P[0][0][0][0][1][0][0]  # 人数
    P = P[0][0][0][0][0] / 4  # 坐标

    # 图像数据
    img = cv2.imread(ipath)  # 三通道图片数据
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);  # 将img图片格式转化为灰度图
    N = int(img.shape[0] / 4)
    M = int(img.shape[1] / 4)  # 向下取整，密度图填充时多加一个像素

    den_map = np.zeros([N, M, 3], dtype=np.float64)
    tot = np.zeros([N + R, M + R], dtype=np.float64)
    GAU = gauss(kernel_size=R, sigma=2.1)
    for j in range(n):
        x = int(P[j][1]) + (R - 1) // 2
        y = int(P[j][0]) + (R - 1) // 2
        tot[x - (R - 1) // 2:x + (R - 1) // 2 + 1, y - (R - 1) // 2:y + (R - 1) // 2 + 1] = \
            tot[x - (R - 1) // 2:x + (R - 1) // 2 + 1, y - (R - 1) // 2:y + (R - 1) // 2 + 1] + GAU
        #尺寸改变，卷积时做零填充
    return tot,n

#绘图函数
def DRAW(tot,person):
    # 密度图颜色
    mp = sio.loadmat('./color/map.mat')
    mp = mp['c']
    mp = mp[::-1]  # 对mp取逆序

    N = tot.shape[0]
    M = tot.shape[1]
    max_den = tot.max()
    den_map = np.zeros([N,M,3],dtype=np.float64)
    for X in range(N):
        for Y in range(M):
            pixel = 255 * tot[X][Y] / max_den
            den_map[X][Y] = mp[int(pixel)] * 255  # den_map三维array，每个位置存储array
            den_map[X][Y] = [int(ele) for ele in den_map[X][Y]]  # 将array解包
    #绘图
    title = '       pre:'+str(round(tot.sum()))+'       real:'+str(person)
    cv2.namedWindow(title, 0)
    cv2.resizeWindow(title , 350,250)
    cv2.imshow(title, den_map/255)  # 按RGB形式展示，但den_map为BGR
    cv2.waitKey()
    cv2.destroyAllWindows()
    return None

'''
img,data,n = Loader(1,path,ipath,15,[300,400])
img,data,n = Loader(0,test_path,test_ipath,15,[182,316])
plt.imshow(img.squeeze().transpose(0,1).transpose(1,2)/255)
cv2.namedWindow('img', 0)
cv2.resizeWindow('img' , 350, 250)
cv2.imshow('img',img.squeeze().transpose(0,1).transpose(1,2).data.numpy()/255)
out = MCNN(img)
DRAW(out.squeeze().data.numpy(),n)
DRAW(data.squeeze().data.numpy(),n)
'''