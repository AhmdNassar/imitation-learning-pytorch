import torch
from torch import nn 

class CarlaNet(nn.Module):
    def __init__(self):
        super(CarlaNet,self).__init__()
        
        """Conv Block """
        self.conv_block = nn.Sequential(
            nn.Conv2d(3,32,5,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(64,128,3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        
        """image FC layers"""
        self.img_fc = nn.Sequential(
            nn.Linear(70400,512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        
        """--- Speed (measurements) ---"""
        self.speed_fc = nn.Sequential(
            nn.Linear(1,128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        
        """--- Joint part ---"""
        self.joint_fc = nn.Sequential(
            nn.Linear(640,512),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        
        """branches"""
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 3),
            ) for i in range(4)
        ])
        
        #TODO understand speed branch and why we should use it && implement it.
    
    def forward(self,sample):
        img , speed , command = sample 
        
        img = self.conv_block(img)
        img = img.view(img.size(0),-1)  #Reshape 
        img = self.img_fc(img)
        
        speed = self.speed_fc(speed)
        
        j = torch.cat([img,speed],1)
        j = self.joint_fc(j)
        
        output = torch.stack([out(j) for out in self.branches],dim=0)
        return output

    
