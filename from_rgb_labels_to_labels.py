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