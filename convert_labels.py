import numpy as np


def from_rgb_labels_to_int_labels(label):
    """ Convert an RGB TIFF label image to class IDs in the range 0-5.
        Label is a 3D numpy array with shape (height, width, channels) or (channels, height, width).   
    """
    
    if label.ndim != 3:
        raise ValueError("Expected a TIFF image with three color channels")

    # Support both height x width x channels and channels x height x width.
    if label.shape[-1] == 3:
        rgb = label
    elif label.shape[0] == 3:
        rgb = np.moveaxis(label, 0, -1)
    else:
        raise ValueError("Expected a TIFF image with three color channels")

    color_to_class = {
        (255, 255, 255): 0,  # Impervious surfaces
        (0, 0, 255): 1,      # Building
        (0, 255, 255): 2,    # Low vegetation
        (0, 255, 0): 3,      # Tree
        (255, 255, 0): 4,    # Car
        (255, 0, 0): 5,      # Clutter/background
    }

    # creates an array of the same height and width as the input image, filled with -1
    encoded = np.full(rgb.shape[:2], -1, dtype=np.int8)

    # Encode the RGB values to class IDs
    for color, class_id in color_to_class.items():
        # create a boolean mask where the RGB values match the current color
        # assign the corresponding class ID to the encoded array at those positions
        mask = np.all(rgb == color, axis=-1)
        encoded[mask] = class_id

    # Assign unmapped colors to class 5 (Clutter/background)
    encoded[encoded < 0] = 5  

    return encoded.astype(np.uint8)




def from_int_labels_to_rgb(label):
    """ Convert an integer encoded label matrix of class IDs in the range 0-5 to a 3D numpy RGB label.
      
      label is a 2D numpy array with shape (height, width).

      Returns a 3D numpy array with shape (height, width, 3) where the last dimension represents the RGB channels.
    """
    
    if label.ndim != 2:
        raise ValueError("Expected a 2D image with integer class IDs in the range 0-5")
    
    if label.min() < 0 or label.max() > 5:
        raise ValueError("Expected integer class IDs in the range 0-5")

    class_to_color = {
        0 : (255, 255, 255), # Impervious surfaces
        1: (0, 0, 255),      # Building
        2 : (0, 255, 255),   # Low vegetation
        3: (0, 255, 0),      # Tree
        4: (255, 255, 0),    # Car
        5: (255, 0, 0),      # Clutter/background
    }

    H, W = label.shape

    # creates a 3D matrix of the same height and width as the input image, filled with -1
    encoded = np.full((H, W, 3), -1)

    # Encode class IDs to RGB values
    for class_id, color in class_to_color.items():
        # create a boolean mask where the class IDs match the current class_id
        # assign the corresponding RGB color to the encoded array at those positions
        mask = (label == class_id)
        encoded[mask] = color

    if np.any(encoded == -1):
        raise ValueError("Some class IDs were not mapped to RGB colors. Check the input label matrix.")

    return encoded