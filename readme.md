This helper module lets you load the dataset Places365 (http://places2.csail.mit.edu/download.html) in its 256x256 variant using PyTorch.

**Note: within the code, the hereby called validation set is referenced as the test set**. The real test set is downloadable from the site above, but the labels are not given since the classification on the dataset is an open challenge.

# Download the data

From the link above download the data (see the "Small images (256 * 256)" section), training and validation set. Please read the terms of use within the same link before downloading.

Read the instructions on https://github.com/metalbubble/places_devkit/download_data.sh on how to get the txt files containing the labels for the categories.

Untar the two datasets into two separate folders within the same root.

Untar the labels archive into the respective datasets folders.

# Usage

To create the torch Datasets from the data (e.g. training set), just run

```
import places365

trainset = places365.Places365(root=<train_root>, transform=<my_transforms>, txt_file=<name_of_the_file_containing_training_labels.txt>
```

to get directly the dataloaders for both train and validation, run

```
import places365

trainloader, testloader = places365.get_dataloaders(pre_root=<folder_containing_train_and_test_root>, batch_train=<batch_size_train>, batch_test=<batch_size_test/val>, root_train=<train_root>, root_test=<test/val_root>)
```

