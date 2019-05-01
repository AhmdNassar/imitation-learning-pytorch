import torch
import torchvision
import torch.nn.functional as F

from torch import nn 
from torchvision import transforms

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.nSize=1 # init with 1 to avoid divided by azro error before feeding model with data
        self.branches = {} # we use it to brach model depend on input command 
        
        """---- Image Layers ----"""
        
        self.buildImageLayers()
        
        """---Speed (measurements) ---"""
        
        self.speed_fc1 = nn.Linear(1,128)
        self.speed_drop1 = nn.Dropout(0.5)
        self.speed_fc2 = nn.Linear(128,128)
        self.speed_drop2 = nn.Dropout(0.5)
        
        """ Joint sensory block"""
        self.joint_fc = nn.Linear(self.nSize,512)
        self.joint_drop = nn.Dropout(0.5)
        
        """start pranching"""
        
        """######################################################"""
        #  command = 0 or 2 -- > follow lane -- > brach idx = 0 #
        #  command = 3 -- > go left          -- > brach idx = 1 #
        #  command = 4-- > Right             -- > brach idx = 2 #
        #  command = 5-- > straight          -- > brach idx = 3 #
        """######################################################"""
        
        for i in range(4):
            self.branches[int(i)] = [nn.Linear(512,256)]
            self.branches[int(i)].append(nn.Dropout(0.5))
            self.branches[int(i)].append(nn.Linear(256,256))
            self.branches[int(i)].append(nn.Dropout(0.5)) 
            self.branches[int(i)].append(nn.Linear(256,3)) # model out is [steer,gas,brake] prediction
                
                
                
    def forward(self,x):
        img , speed , command = x
        """ --- image layers ---"""
        
        img = F.relu(self.drop1(self.bn1(self.conv1(img))))
        img = F.relu(self.drop2(self.bn2(self.conv2(img))))
        img = F.relu(self.drop3(self.bn3(self.conv3(img))))
        img = F.relu(self.drop4(self.bn4(self.conv4(img))))
        img = F.relu(self.drop5(self.bn5(self.conv5(img))))
        img = F.relu(self.drop6(self.bn6(self.conv6(img))))
        img = F.relu(self.drop7(self.bn7(self.conv7(img))))
        img = F.relu(self.drop8(self.bn8(self.conv8(img))))
        img = img.view(img.size(0),-1)  #Reshape 
        self.nSize = img.data.size(1)   # get input size for  first FC layer 
        img = F.relu(self.dropFc1(self.fc1(img)))
        img = F.relu(self.dropFc2(self.fc2(img)))
        
        """---Speed (measurements) ---"""
        
        speed = F.relu(self.speed_drop1(self.speed_fc1(speed)))
        speed = F.relu(self.speed_drop2(self.speed_fc2(speed)))
        
        """ Joint sensory """
        j = torch.cat((img,speed),1)
        self.nSize = img.data.size(1)   # get input size for joint FC layer
        j = self.relu(self.joint_drop(self.joint_fc(j)))
        
        """branching"""
        if int(command) == 0 or int(command) == 2 : # follow lane
            out = F.relu(self.branches[0][1](self.branches[0][0](j)))
            out = F.relu(self.branches[0][3](self.branches[0][2](out)))
            out = self.branches[0][4](out)
        
        if int(command) == 3 : # left
            out = F.relu(self.branches[1][1](self.branches[1][0](j)))
            out = F.relu(self.branches[1][3](self.branches[1][2](out)))
            out = self.branches[1][4](out)
        
        if int(command) == 4 : # right
            out = F.relu(self.branches[2][1](self.branches[2][0](j)))
            out = F.relu(self.branches[2][3](self.branches[2][2](out)))
            out = self.branches[2][4](out)
        
        if int(command) == 5 : # straight
            out = F.relu(self.branches[3][1](self.branches[3][0](j)))
            out = F.relu(self.branches[3][3](self.branches[3][2](out)))
            out = self.branches[3][4](out)
        return out
    
    def buildImageLayers(self):
        self.conv1 = nn.Conv2d(3,32,5,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32,32,3,padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(32,64,3,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.drop3 = nn.Dropout(0.2)
        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.drop4 = nn.Dropout(0.2)
        self.conv5 = nn.Conv2d(64,128,3,stride=2,padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.drop5 = nn.Dropout(0.2)
        self.conv6 = nn.Conv2d(128,128,3,padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.drop6 = nn.Dropout(0.2)
        self.conv7 = nn.Conv2d(128,256,3,padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.drop7 = nn.Dropout(0.2)
        self.conv8 = nn.Conv2d(256,256,3,padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.drop8 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.nSize,512)
        self.dropFc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,512)
        self.dropFc2 = nn.Dropout(0.5)
    
test = Net()
print(test)


    
