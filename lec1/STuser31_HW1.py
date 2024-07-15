import numpy as np
import cv2
from PIL import Image


# import Dataset & DataLoader 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
# import image process package
import torchvision
import torchvision.transforms as transforms

import os
import torch

padding = (50,50,50,50)
trans = transforms.Compose([
    # define your own image preprocessing
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    
    # convert to tensor
	transforms.ToTensor()
])

class txt_dataset(Dataset):
    # override the init function
	def __init__(self, data_dir, label_dir, file_num,  transform=None):
		self.data_dir = data_dir 
		self.label_dir = label_dir
		self.file_num = file_num 
        
    #override the getitem function
	def __getitem__(self, index):
		data_name = os.path.join(self.data_dir, str(index + 1) + '.jpg')
		label_name = os.path.join(self.label_dir, str(index + 1) + '.txt')
		image = Image.open(data_name)
		output = trans(image)
		label = np.loadtxt(label_name)
		return output, label
        
    #override the len function    
	def __len__(self):
		return self.file_num

# declare training dataset
train_dataset = txt_dataset(	data_dir = './train_data',
				label_dir = './train_label',
				file_num = 9)

# declare testing dataset
test_dataset = txt_dataset(	data_dir = './test_data',
				label_dir = './test_label',
				file_num = 3)

# declare training dataloader
trainloader = DataLoader(train_dataset, shuffle = True, batch_size = 3)

# declare testing dataloader
testloader = DataLoader(test_dataset, shuffle = True, batch_size = 3)

print('train_data & label')
for index, data_package in enumerate(trainloader):
	train_data, train_label = data_package
	print('Type: ', type(train_data))
	print('Index: ',index)
	print('Data: ',train_data)
	print('Label:', train_label)

print('test_data & label')
for test_data, test_label in testloader:
	print('Data: ',test_data)
	print('Label:', test_label)

