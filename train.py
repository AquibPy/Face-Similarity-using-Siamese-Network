import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import config
from dataset import SiameseDataset
from model import SiameseNetwork
from loss import ContrastiveLoss
from utils import save_checkpoint,load_checkpoint

def train_fn(model,optimizer,loader,loss):
    loop = tqdm(loader,leave=True)
    for idx ,data in enumerate(loop):
        img0, img1 , label = data
        img0, img1 , label = img0.to(config.DEVICE), img1.to(config.DEVICE) , label.to(config.DEVICE)
        optimizer.zero_grad()
        output_1,output_2 = model(img0,img1)
        loss_contrastive = loss(output_1,output_2,label)
        loss_contrastive.backward()
        optimizer.step()
        # if i %10 == 0 :
        #     print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
        if idx % 10 == 0:
            loop.set_postfix(
                LOSS=loss_contrastive.item(),
            )

def main():
    model = SiameseNetwork().to(config.DEVICE)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(),lr=config.LR)
    if config.LOAD_MODEL:
        load_checkpoint(model,optimizer,config.LR)
    train_dataset = SiameseDataset(root_dir=config.TRAIN_DIR,transform=config.TRANS,should_invert=False)
    train_loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True)
    for i in range(config.EPOCHS):
        train_fn(model,optimizer,train_loader,criterion)
        if config.SAVE_MODEL and i%10==0:
            save_checkpoint(model,optimizer)

if __name__=='__main__':
    main()