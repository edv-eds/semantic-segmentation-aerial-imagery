import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import tifffile as tiff

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
import albumentations as A


from convert_labels import from_rgb_labels_to_int_labels



class PotsdamDataset(Dataset):
    def __init__(self, root, transform=None):
        """Constructor

        Args:
            root (Path/str): Filepath to the data root, e.g. './small_train_patches'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
    
        root = Path(root)
        if not (root.exists() and root.is_dir()):
            raise ValueError(f"Data root '{root}' is invalid")

        self.root = root
        self.transform = transform

        # Collect samples, both cat and dog and store pairs of (filepath, label) in a simple list.
        self._samples = self._collect_samples()




    def __getitem__(self, index):
        """Get sample by index

        Args:
            index (int)

        Returns:
             The index'th sample (Tensor, Tensor) 
             The first tensor is the image (3 x height x width).
             The second tensor is the label (height x width) with integer class IDs in the range 0-5.
        """
        # Access the stored path and label for the correct index
        img_path, label_path = self._samples[index]
        
        # Load the image into memory
        # in numpy array format, shape (height, width, channels)
        img = tiff.imread(img_path)
        label = tiff.imread(label_path)

        # Convert the label to integer format
        label = from_rgb_labels_to_int_labels(label)


        # Apply the transform if it is provided
        # Need to use albumentations package for the transform, since it can handle both image and label simultaneously
        if self.transform is not None:
            augmented = self.transform(image=img, mask=label)
        
            img = augmented["image"]
            label = augmented["mask"]


        # Convert the image to PyTorch tensors, 
        # rearrange the dimensions to (channels, height, width)
        # Normalize the pixel values to range [-1, 1]
        img_transform = Compose([ToTensor(), 
                                 Normalize((0.5,), (0.5,)),
                                 ])
        img = img_transform(img)

        # Convert the label to a long tensor for PyTorch
        label = torch.from_numpy(label).long() 

        return img, label



    def __len__(self):
        """Total number of samples"""
        return len(self._samples)


    def get_img_id(self, index):
        """Get the id of the image at the given index
            
            example image name: 
                small_val_patches/images/top_potsdam_2_10_256_1280.tif
            stem of image name: 
                top_potsdam_2_10_256_1280

        Args:
            index (int): The index of the sample in the dataset.

        Returns:
            str: The id of the image, e.g. "2_10_256_1280"
        """
        img_path, _ = self._samples[index]

        # Extract the id from the filename, which is in the format "top_potsdam_<id>.tif"
        img_id = img_path.stem.split("_", 2)[-1]  # Get the part after the first two underscores
        
        return img_id



    def _collect_samples(self):
        """Collect all paths and labels

        Helper method for the constructor
        """
        # Returns a sorted list of image filpaths
        img_paths = self._collect_tif_sub_dir(self.root / "images")

        # Returns a sorted list of label filpaths
        label_paths = self._collect_tif_sub_dir(self.root / "labels")

        # Zip the to lists into a single object
        # This will create a list of tuples, where each tuple contains the image path and the corresponding label path.
        pairs = list(zip(img_paths, label_paths))
        
        return pairs



    @staticmethod
    def _collect_tif_sub_dir(sub_dir: Path):
        """Collect paths of all .tif files in a given subdirectory
        Helper method for the constructor

        Returns:
            A sorted list of all paths to .tif files in the given subdirectory. 
            Raises an error if the directory does not exist.
        """
        if not sub_dir.exists():
            raise ValueError(
                f"Directory '{sub_dir}' does not exist. Are you sure you have the correct path?"
            )
        return sorted(sub_dir.glob("*.tif"))



    def get_tif_by_id(self, id_: str):
        """Get sample by image id. Convenience method for exploration.
    
        The image id is a string that uniquely identifies an image in the dataset.
        It is used to retrieve the corresponding image and label.

        Consider the following filename:
        top_potsdam_2_10_256_1280.tif

        The image id is the part of the filename after the first two underscores and before the file extension, i.e. "2_10_256_1280".

        Args:
            id_ (str): "2_10_256_1280"
        Returns:
            img (numpy.ndarray, shape (height, width, channels)): The image data
            label (numpy.ndarray, shape (height, width, channels)): The RGB label data
        """
        img_path = self.root / "images" / f"top_potsdam_{id_}.tif"
        label_path = self.root / "labels" / f"top_potsdam_{id_}.tif"

        try:
            index = self._samples.index((img_path, label_path))
        except ValueError:
            raise ValueError(f"Image with id '{id_}' not found.")

        img = tiff.imread(img_path)
        label = tiff.imread(label_path)
        return img, label



    def get_tensor_by_id(self, id_: str):
        """Get sample by image id. Convenience method for exploration.
    
        The image id is a string that uniquely identifies an image in the dataset.
        It is used to retrieve the corresponding image and label.

        Consider the following filename:
        top_potsdam_2_10_256_1280.tif

        The image id is the part of the filename after the first two underscores and before the file extension, i.e. "2_10_256_1280".

        Args:
            id_ (str): "2_10_256_1280"
        Returns:
            img (torch.Tensor, shape (channels, height, width)): The image data
            label (torch.Tensor, shape (height, width)): The integer label data
        """
        img_path = self.root / "images" / f"top_potsdam_{id_}.tif"
        label_path = self.root / "labels" / f"top_potsdam_{id_}.tif"

        try:
            index = self._samples.index((img_path, label_path))
        except ValueError:
            raise ValueError(f"Image with id '{id_}' not found.")

        return self[index]  # This will call the __getitem__ method and return the image and label as tensors
