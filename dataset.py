import random
import torch
import numpy as np
import config
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from PIL import Image,ImageOps
from torchvision.utils import save_image


class SiameseDataset(Dataset):
    def __init__(self,root_dir,transform=False,should_invert = True):
        super().__init__()
        self.root_dir = root_dir
        self.ImageDataFolder = ImageFolder(self.root_dir)
        self.transform = transform
        self.should_invert = should_invert
    
    def __len__(self):
        return len(self.ImageDataFolder)
    
    def __getitem__(self, index):
        img0_tuple = random.choice(self.ImageDataFolder.imgs)
        get_same_class = random.randint(0,1)
        if get_same_class:
            while True:
                img1_tuple = random.choice(self.ImageDataFolder.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        
        else:
            while True:
                img1_tuple = random.choice(self.ImageDataFolder.imgs)
                if img0_tuple[1]!= img1_tuple[1]:
                    break
        
        img0 = Image.open(img0_tuple[0]).convert("L")
        img1 = Image.open(img1_tuple[0]).convert("L")

        if self.should_invert:
            img0 = ImageOps.invert(img0)
            img1 = ImageOps.invert(img1)
        
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0,img1, torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))

if __name__=='__main__':
    trans = tt.Compose(
        [
            tt.Resize((100,100)),
            tt.ToTensor()
        ]
    )
    dataset = SiameseDataset(root_dir=config.TRAIN_DIR,transform=trans,should_invert=False)
    loader = DataLoader(dataset,batch_size=8,shuffle=True,pin_memory=True)
    dataiter = iter(loader)
    data_batch = next(dataiter)
    concatenated = torch.cat((data_batch[0],data_batch[1]),0)
    save_image(concatenated,"data.png")
    print(data_batch[2].numpy())

