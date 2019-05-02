import torch
import torchvision
import numpy as np
import os
from torchvision import transforms
import h5py as h5


class AgentData:
    """Carla self-driving car dataset.
    
    Args:
        dataDir: direction of folder which contain .h5 files

    
    Return:
          Tuble ---> (image, steer, gas, brake,speed,command ) 
          
    """

    def __init__(self,dataDir):

        self.dataDir = dataDir
        if( not os.path.isdir(self.dataDir)):
            print("Error: "+self.dataDir+" not exist!")
            return

        self.filesName = os.listdir(self.dataDir) # list of h5 files names And each file contain 200 images and there observation (1,28)


    def __len__(self):

        self.numberOfFiles = len(self.filesName)
        return self.numberOfFiles * 200
        

    def __getitem__(self,idx):
        
        self.data = h5.File(self.dataDir+'\\'+self.filesName[idx//200],'r')
        self.imgIndex= idx%200
        return (self.data['rgb'][self.imgIndex],self.data['targets'][self.imgIndex][0],self.data['targets'][self.imgIndex][1], \
                self.data['targets'][self.imgIndex][2],self.data['targets'][self.imgIndex][10],self.data['targets'][self.imgIndex][24]) 
        

class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img = sample
         
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))
        
        return img 