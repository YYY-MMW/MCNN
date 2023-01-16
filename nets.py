import torch
import torch.nn as nn
import cv2
from den_map import *

#单列卷积网络
class CNN(nn.Module):
    def __init__(self,channel_list,ker_list):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels= channel_list[0],
                kernel_size= ker_list[0],
                stride = 1,
                padding= ker_list[0]//2+int(15*4/2),    #保证形状不变
            ),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_list[0],
                out_channels= channel_list[1],
                kernel_size=ker_list[1],
                stride=1,
                padding=ker_list[1]//2
            ),
            nn.BatchNorm2d(channel_list[1]),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2,)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_list[1],
                out_channels=channel_list[2],
                kernel_size=ker_list[2],
                stride=1,
                padding=ker_list[2] // 2
            ),
            nn.BatchNorm2d(channel_list[2]),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, )
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_list[2],
                out_channels=channel_list[3],
                kernel_size=ker_list[3],
                stride=1,
                padding=ker_list[2] // 2
            ),
            nn.BatchNorm2d(channel_list[3]),
            nn.ReLU(),
        )
    def forward(self,input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        return conv4

#支持pretrain MCNN
class Mcnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.CNN_L = torch.load('./data/CNN_L.pkl')

        self.CNN_M = torch.load('./data/CNN_M.pkl')

        self.CNN_S = torch.load('./data/CNN_S.pkl')

        self.COV_ALL = nn.Sequential(
                nn.Conv2d(
                in_channels= 30,
                out_channels=1,
                kernel_size= 1,
                stride=1,
                padding=0,
            ),
            nn.ReLU()
        )


    def forward(self,input):
        cnn_L = self.CNN_L(input)
        cnn_M = self.CNN_M(input)
        cnn_S = self.CNN_S(input)
        al = torch.cat((cnn_L,cnn_M,cnn_S),1)
        cnn_all = self.COV_ALL(al)
        return cnn_all