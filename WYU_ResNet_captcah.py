import torch as t
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms as T
from torch.utils.data.dataloader import DataLoader
import time

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel, track_running_stats=True)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=62):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=True),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 512)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        return y1, y2, y3, y4

    def loadIfExist_cpu(self):
        fileList = os.listdir("./model/captcha/")
        # print(fileList)
        if "resNet_new.pth" in fileList:
            name = "./model/captcha/resNet_new.pth"
            self.load_state_dict(t.load(name, map_location=t.device('cpu')))

    def loadIfExist_gpu(self):
        fileList = os.listdir("./model/captcha/")
        # print(fileList)
        if "resNet_new.pth" in fileList:
            name = "./model/captcha/resNet_new.pth"
            self.load_state_dict(t.load(name))


def LabeltoStr(Label):
    Str = ""
    for i in Label:
        if i <= 9:
            Str += chr(ord('0') + i)
        elif i <= 35:
            Str += chr(ord('a') + i - 10)
        else:
            Str += chr(ord('A') + i - 36)
    return Str


def PreDeal(data):
    transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    return transform(data)


def userTest(model, input):
    t1 = time.time()
    y1, y2, y3, y4 = model(input)
    print(time.time() - t1)
    y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(1, 1), y2.topk(1, dim=1)[1].view(1, 1), \
                    y3.topk(1, dim=1)[1].view(1, 1), y4.topk(1, dim=1)[1].view(1, 1)
    y = t.cat((y1, y2, y3, y4), dim=1)
    return LabeltoStr([y[0][0], y[0][1], y[0][2], y[0][3]])
    

def UseCPU(img):
    input = PreDeal(img)[np.newaxis, :]
    model = ResNet(ResidualBlock)
    model.eval()
    model.loadIfExist_cpu()
    return userTest(model, input)


def UseGPU(img):
    input = PreDeal(img)[np.newaxis, :]
    model = ResNet(ResidualBlock)
    model.eval()
    model.loadIfExist_gpu()
    if t.cuda.is_available():
        model = model.cuda()
        input = input.cuda()
    return userTest(model, input)


if __name__ == '__main__':
    img_cpu_test = Image.open("E:/demo/CNN_captcha-master/data/test/2aap.jpg")
    img_gpu_test = Image.open("E:/demo/CNN_captcha-master/data/test/2aew.jpg")
    print(UseCPU(img_cpu_test))
    # print(UseGPU(img_gpu_test))
    


