import torch
import torchvision
import config
import torch.optim as optim
import torch.nn.functional as F
from dataset import SiameseDataset
from model import SiameseNetwork
from torch.utils.data import DataLoader
from utils import load_checkpoint
import numpy as np
import matplotlib.pyplot as plt

model = SiameseNetwork().to(config.DEVICE)
optimizer = optim.Adam(model.parameters(),lr=config.LR)
load_checkpoint("my_checkpoint.pth.tar",model,optimizer,config.LR)
val_dataset = SiameseDataset(root_dir=config.VAL_DIR,transform=config.TRANS,should_invert=False)

test_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)
dataiter = iter(test_loader)
x0,_,_ = next(dataiter)

def imshow(img,text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('output.png') 

for i in range(10):
    _,x1,label2 = next(dataiter)
    concatenated = torch.cat((x0,x1),0)
    
    output1,output2 = model(x0.to(config.DEVICE),x1.to(config.DEVICE))
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))