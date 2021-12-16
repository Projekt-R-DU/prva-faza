# Kopija modela iz ./ResNet18 bilježnica za lakše učitavanje

import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1 #sto znaci ovaj expansion?

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #dimenzija jezgre odnosno matrice koja se pomice po ulaznoj i stvara mapu znacajki, 
        #padding nadopunjuje rubove, bias je false jer se koristi BatchNorm, stride je broj koraka(redaka/stupaca) koliko se pomice jezgra

        self.bn1 = nn.BatchNorm2d(planes)#normalizacija pomice vrijednosti u ovisnosti o srednjoj vrij.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()#kombinira module
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        #u ovaj if se ulazi kod svakog osim prvo bloka
        #TODO nadopuniti opis, sto znaci self.expansion? 

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        # CONV1 -> BN1 -> ReLu -> CONV2 -> BN2 = F(X)
        # F(x) + shorcut -> ReLu
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, cifar = True, num_classes=10):#koliko klasa imamo na kraju
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3 if cifar else 1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)#zbog grayscale inpanes je 1, za cifar 3
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)# flattening

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)#listu od dva elementa, prvi i drugi element su strideovi 
        layers = []
        for stride in strides:#svi u layeru imaju stride 1, osim prvog koji ima 2
            layers.append(block(self.in_planes, planes, stride))#appenda na listu blok
            self.in_planes = planes * block.expansion#pridruzivanje planesa in_planes, mnoezenjem s 1?
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)#out je jezgra, a 4 je stride, odnosno korak
        out = out.view(out.size(0), -1)#reshape tensora prije nego ide dalje, -1 znaci da ne znamo broj redaka/stupaca
        out = self.linear(out)#flattening prije fully connected layera
        return out
        # CONV1 -> BN1 -> Layer1(sa dva bloka) -> Layer2(sa dva bloka) -> Layer3(sa dva bloka) -> Layer4(sa dva bloka)
        # AVGPOOL -> reshape -> flattening (linear) ili downsample

def ResNet18(cifar):
    return ResNet(BasicBlock, [2, 2, 2, 2], cifar)#u svakom sloju koliko je blokova
