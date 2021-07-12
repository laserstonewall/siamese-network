from collections import defaultdict

import numpy as np
import pandas as pd
import PIL

import torch
from torch.utils.data import Dataset

class SiamesePairedDataset(Dataset):
    def __init__(self, 
                 data, 
                 path_col=None, 
                 label_col=None, 
                 sampling_strategy='uniform', 
                 class_prob=None, 
                 transform=None):
        
        #### Set important class variables
        self.path_col   = path_col
        self.label_col  = label_col
        self.classes    = data[label_col].unique()
        self.transform  = transform
        self.strategy   = sampling_strategy
        self.num_images = len(data)
        
        #### Create dictionary of key: value format class_label: [list of class image paths]
        
        # Each key starting with a blank list simplifies the dictionary's construction
        self.image_dict = defaultdict(list)
        
        # Iterate through rows, adding the image file path to the appropriate class in dictionary
        for image_path, image_class in data[[path_col, label_col]].values:
            self.image_dict[image_class].append(image_path)
        
        #### Based on the specified sampling strategy, establish individual class probabilities
        
        # Each class will have equal probability of being selected regardless of size
        if self.strategy=='uniform':
            self.class_prob = np.ones(len(self.classes))
            self.class_prob = self.class_prob / np.sum(self.class_prob)
        
        # Each class selection probability proportional to prevalence
        elif self.strategy=='proportional':
            # Create an array with individual class lengths
            class_lens = np.array([len(self.image_dict[image_class]) for image_class in self.classes])
            self.class_prob = class_lens / self.num_images
            
        # Each class selected according to user input of class probabilities
        elif self.strategy=='custom':
            if class_prob is not None:
                self.class_prob = class_prob
            else:
                raise Exception("For custom sampling strategy, class probabilities must be specified.")
            
        else:
            raise Exception("Invalid stratification strategy")
                
    def __len__(self):
        return self.num_images
        
    def __getitem__(self, idx):
                    
        # Get classes for both images, we'll overwrite class2 if it's a same-class example
        class1, class2 = np.random.choice(self.classes, p=self.class_prob, replace=False, size=2)
        
        #### Two cases: both drawn from same class, both drawn from different classes
        
        # Same class
        if idx % 2 == 1:
            label = 0 # Training label when images are same class
            
            # Same class, so overwrite random family2 from earlier
            class2 = class1
            
            # Select the two file path strings
            file1, file2 = np.random.choice(self.image_dict[class1], size=2, replace=False)
        
        # Different classes
        else:
            label = 1 # Training label when images are different classes
            
            # Select a random image from each dataset
            file1 = np.random.choice(self.image_dict[class1])
            file2 = np.random.choice(self.image_dict[class2])
             
        # Load the image pair
        img1 = PIL.Image.open(file1)
        img2 = PIL.Image.open(file2)
        
        if img1.mode != 'RBG':
            img1 = img1.convert('RGB')
        if img2.mode != 'RBG':
            img2 = img2.convert('RGB')
           
        # Apply any torch transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        if img1.size()[0] != 3:
            print(file1, img1.size()[0])
        if img2.size()[0] != 3:
            print(file2, img2.size()[0])
            
        # Concatenate them along the channel dimension (i.e 2 3-channel  
        # images as a single "6-channel" image. Keeping them as a tuple
        # breaks some of the fastai workflows when using them, so "cheating" 
        # in this way then separating within the network gets around this problem
        return torch.cat((img1, img2), dim=0), label