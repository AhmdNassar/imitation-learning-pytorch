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
          Nested Tuble ---> ((image, speed,command) ,(steer, gas, brake)) 
          
    """

    def __init__(self,dataDir,transforms=None):

        self.dataDir = dataDir
        self.transforms = transforms
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

        img, speed ,command= self.data['rgb'][self.imgIndex] , self.data['targets'][self.imgIndex][10] , self.data['targets'][self.imgIndex][24]

        steer, gas, brake = self.data['targets'][self.imgIndex][0],self.data['targets'][self.imgIndex][1],self.data['targets'][self.imgIndex][2]

        if(transforms):
            img = self.transforms(img)

        return ((img,speed,command),(steer,gas,brake))
        

class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img = sample
         
        # if image has no grayscale color channel, add one
        if(len(img.shape) == 2):
            # add that third color dim
            img = img.reshape(img.shape[0], img.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))
        
        return img 