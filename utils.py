
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import models, datasets
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import random
from copy import deepcopy

random.seed(123)


def generate_dataloader(data, name, categories, transform, batch_size, unseen=False):
    if transform is None:
        dataset = SimilarityDataset(data, categories, transform=transforms.ToTensor(), unseen=unseen)
    else:
        dataset = SimilarityDataset(data, categories, transform=transform, unseen=unseen)

    kwargs = {}
    
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=False, num_workers=0,
                        **kwargs)
    return dataloader


class SimilarityDataset(Dataset):
    def __init__(self, data, categories, transform, unseen=False):
        super(SimilarityDataset, self).__init__()
        self.categories = categories
        self.unseen = unseen
        if transform is None:
            self.dataset = datasets.ImageFolder(data, transform=transforms.ToTensor())
        else:
            self.dataset = datasets.ImageFolder(data, transform=transform)
        self.group_examples()

    def group_examples(self):
        np_arr = np.array(deepcopy(self.dataset.targets))

        self.grouped_examples = {}
        if self.unseen == False:
            for i in range(0,self.categories):
                self.grouped_examples[i] = np.where((np_arr==i))[0]
        else:
            for i in range(self.categories,self.categories*2):
                self.grouped_examples[i] = np.where((np_arr==i))[0]

    def __len__(self):
        return int(self.categories*len(self.dataset)/200)

    def __getitem__(self, index):
        if self.unseen == False:
            selected_class = random.randint(0, self.categories-1)
        else:
            selected_class = random.randint(self.categories,self.categories*2-1)
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
        index_1 = self.grouped_examples[selected_class][random_index_1]
        image_1 = deepcopy(self.dataset[index_1][0])

        #pick image from same category for even index
        if index % 2 == 0:
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
            index_2 = self.grouped_examples[selected_class][random_index_2]
            image_2 = deepcopy(self.dataset[index_2][0])
            target = torch.tensor(1, dtype=torch.float)

        #pick image from same category for odd index
        else:
            if self.unseen == False:
                other_selected_class = random.randint(0, self.categories-1)
            else:
                other_selected_class = random.randint(self.categories,self.categories*2-1)

            while other_selected_class == selected_class:
                if self.unseen == False:
                    other_selected_class = random.randint(0, self.categories-1)
                else:
                    other_selected_class = random.randint(self.categories,self.categories*2-1)
            
            random_index_2 = random.randint(0, self.grouped_examples[other_selected_class].shape[0]-1)
            index_2 = self.grouped_examples[other_selected_class][random_index_2]
            image_2 = deepcopy(self.dataset[index_2][0])
            target = torch.tensor(0, dtype=torch.float)

        return image_1, image_2, target

