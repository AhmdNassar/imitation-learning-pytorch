import matplotlib.pyplot as plt
import torch
import numpy as np
import os

class StopEarly:
    """ Stop training early and save model if no good improvment """

    def __init__(self,model_dir = 'saved_models/',patience=10):
        """ 
            Args:
                patience : How long to wait after last time (training/test/val) loss improved .
                            Default: 7
                model_dir : dir to save model when we git best loss
                            Default: saved_models/
                
                Hint: we can use this class to stop model depend on test loss or validation loss,
                      by just pass val/test loss instead of train loss
        """
        self.best_loss = None
        self.patience = patience
        self.counter = 0
        self.model_dir = model_dir
        if(not os.path.exists(self.model_dir)):
            os.makedirs(model_dir)
        
    def __call__(self,model,loss):
        # model : your traning model
        # loss : current loss
        if self.best_loss is None :
            self.best_loss = loss
            return False
        
        else:
            
            if self.counter >= self.patience:
                return True
            if loss >= self.best_loss :
                self.counter += 1
                print(f"no improvement for {self.counter} / {self.patience}./n------------")
            else:
                self.counter = 0
                self.best_loss = loss
                # save model 
                print(f'We get new best loss: {self.best_loss} , saving model...')
                torch.save(model.state_dict(),self.model_dir+"model.pt")
                print('model saved./n------------')
            return False


def visualizeImages(images,batch_size,GPU=False,trans=False,gt=None,gray=False):
    fig = plt.figure(figsize=(10,20))
    for i in range(batch_size):
        img  = images[i] 
        if(GPU):
            img = np.array(img.cpu(),dtype='int')

        else:
            img =np.array(img,dtype='int')

        if (trans):
            img = np.transpose(img,(1,2,0))
        ax = fig.add_subplot(batch_size/2,2,i+1)
        ax.imshow(img)
        plt.axis('off')

def loss_mask(controls):
    """
        Args
            controls
            the control values that have the following structure
            command flags: 2 - follow lane; 3 - turn left; 4 - turn right; 5 - go straight
            
        Returns
            a mask to have the loss function applied
            only on over the correct branch.
    """

    """ A vector with a mask for each of the control branches"""
    controls_masks = []
    number_targets = 3 # Steer", "Gas", "Brake
    # when command = 2, branch 1 (follow lane) is activated
    controls_b1 = (controls == 2)
    controls_b1 = torch.tensor(controls_b1, dtype=torch.float32).cuda()
    controls_b1 = torch.cat([controls_b1] * number_targets, 1)
    controls_masks.append(controls_b1)
    # when command = 3, branch 2 (turn left) is activated
    controls_b2 = (controls == 3)
    controls_b2 = torch.tensor(controls_b2, dtype=torch.float32).cuda()
    controls_b2 = torch.cat([controls_b2] * number_targets, 1)
    controls_masks.append(controls_b2)
    # when command = 4, branch 3 (turn right) is activated
    controls_b3 = (controls == 4)
    controls_b3 = torch.tensor(controls_b3, dtype=torch.float32).cuda()
    controls_b3 = torch.cat([controls_b3] * number_targets, 1)
    controls_masks.append(controls_b3)
    # when command = 5, branch 4 (go strange) is activated
    controls_b4 = (controls == 5)
    controls_b4 = torch.tensor(controls_b4, dtype=torch.float32).cuda()
    controls_b4 = torch.cat([controls_b4] * number_targets, 1)
    controls_masks.append(controls_b4)


    return controls_masks

def l2_loss(params):
    """
        Functional LOSS L2
        Args
            params dictionary that should include:
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls_mask: the masked already expliciting the branches tha are going to be used
                branches_weights: the weigths that each branch will have on the loss function
        Returns
            A vector with the loss function
    """

    """ It is a vec for each branch"""
    loss_branches_vec = []
    # TODO This is hardcoded but all our cases rigth now uses four branches
    for i in range(len(params['branches'])):
        loss_branches_vec.append(((params['branches'][i] - params['targets']) **2
                                           * params['controls_mask'][i])
                                 * params['branch_weights'][i])
    return loss_branches_vec


def l1_loss(params):
    """
        Functional LOSS L1
        Args
            params dictionary that should include:
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls_mask: the masked already expliciting the branches tha are going to be used
                branches weights: the weigths that each branch will have on the loss function
        Returns
            A vector with the loss function
    """

    """ It is a vec for each branch"""
    loss_branches_vec = []

    for i in range(len(params['branches']) ):
        loss_branches_vec.append(torch.abs((params['branches'][i] - params['targets'])
                                           * params['controls_mask'][i])
                                 * params['branch_weights'][i])
    
    return loss_branches_vec