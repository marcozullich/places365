import os
from torch.utils.data import Dataset, DataLoader
import torch
from skimage import io
import torchvision.transforms as transforms
import numpy as np
from pandas import read_csv

def get_dataloaders(pre_root, batch_train, batch_test=None, root_train="data_256", root_test="val_256", train_txt="places365_train_standard.txt", test_txt="places365_val.txt", shuffle_train=True, **kwargs):
    '''
    Args:
        pre_root: folder where the folders for train and test are contained
        batch_train: batch size for training set.
        batch_test: batch size for test set. If None, no test set is built.
        root_train: folder where the training images are contained. Path must reflect
            what contained within the train_txt file.
        root_test: folder where the test images are contained. Path must
            reflect what contained within the test_txt file.
        train_txt, test_txt: composition of the folders root_train,
            root_test: each row is "path/to/img category"
        shuffle_train: whether to shuffle images for the trainloader
        **kwargs: any other arg passed on to DataLoader (e.g. num_workers)

    '''
    transforms_train = transforms.Compose(
        [transforms.ToTensor(),
        transforms.RandomCrop(256, padding=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(.1,.1))]
    )
    transforms_test = transforms.Compose(
        [transforms.ToTensor()]
    )



    trainset = Places365(os.path.join(pre_root, root_train), transform=transforms_train)
    testset = Places365(os.path.join(pre_root, root_test), transform=transforms_test, txt_file="places365_val.txt")

    trainloader = DataLoader(trainset, batch_size=batch_train, shuffle=shuffle_train, **kwargs)
    testloader = DataLoader(testset,  batch_size=batch_test, shuffle=False, **kwargs)

    return trainloader, testloader

class Places365(Dataset):
    def __init__(self, root, transform=None, txt_file="places365_train_standard.txt"):
        '''
        Args:
            root: base folder where the dataset is stored
            transform: torchvision.transform to modify the dataset
            txt_file: path of text file containing the pairs (imgpath, category) separated by a blank

        '''
        self.root = root
        self.transform = transform
        self.imlist, self.targets = self._get_imglist_target(root, txt_file=txt_file)

    def _get_imglist_target(self, root, txt_file, extension=".jpg"):

        imglist_file = read_csv(os.path.join(root, txt_file), delimiter=" ", header=None, names=("path", "id"))
        
        return [os.path.join(root,f) for f in imglist_file.path], list(imglist_file.id)

    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        ims = self._load_imgs(indices)

        if self.transform is not None:
            self.transform(ims)

        return ims, self.targets[indices]
    
    def _load_imgs(self, indices):
        '''
        Helper that loads images from a list of indices returning a np.ndarray
        '''
        imgnames = self.imlist[indices]
        if not isinstance(imgnames, list):
            imgnames = [imgnames]
        return self._imagelist_to_array(imgnames)

    def _imagelist_to_array(self, imglist, h=256, w=256, ch=3):
        '''
        Helper that loads images from a list of img paths returning a np.ndarray 
        '''
        arr = np.empty((len(imglist), h, w, ch))
        for i,img in enumerate(imglist):
            arr[i] = io.imread(img)
        
        return arr
