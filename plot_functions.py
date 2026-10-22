import torch
import matplotlib.pyplot as plt
import numpy as np

from convert_labels import from_int_labels_to_rgb

from training_loop import output_to_label


def display_image_tensor(axis, image_tensor):
    """Display a tensor as an image

    Args:
        axis (pyplot axis)
        image_tensor (torch.Tensor): tensor with shape (num_channels=3, heigth, width)
        or 
        image_tensor (torch.Tensor): tensor with shape (num_channels=1, heigth, width)
    """

    # See hint above
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(
            "The `display_image_tensor` function expects a `torch.Tensor` "
            + "use the `ToTensor` transformation to convert the images to tensors."
        )

    image_tensor = image_tensor.detach().cpu()


    # integer label tensor with shape (1, height, width) 
    if image_tensor.ndim == 3 and image_tensor.shape[0] == 1:
        # If the image tensor has a single color channel, i.e. it is an integer label tensor 
        # squeeze the channel dimension and convert it to a numpy array.
        image_data = image_tensor.squeeze(0).numpy()
        # The imshow commands expects a `numpy array` with shape (height, width, num_channels=3).
        image_data = from_int_labels_to_rgb(image_data)


    # image tensor with shape (3, height, width)
    elif image_tensor.ndim == 3:
        # revert normalization of image tensor in range [-1, 1] to range [0, 1] for display
        image_tensor = (image_tensor + 1) / 2

        # The imshow commands expects a `numpy array` with shape (height, width, num_channels=3).
        # We rearrange the dimensions with `permute` and then convert it to `numpy`
        image_data = image_tensor.permute(1, 2, 0).numpy()


    # integer label tensor with shape (height, width)
    elif image_tensor.ndim == 2:
        # The imshow commands expects a `numpy array` with shape (height, width, num_channels=3).
        # We convert it to `numpy` and add a dimension for the color channel.
        image_data = image_tensor.numpy()
        image_data = from_int_labels_to_rgb(image_data)

    else:
        raise ValueError(
            "The `display_image_tensor` function expects a 2D or 3D tensor. "
            + f"Got {image_tensor.ndim}D tensor instead."
        )

    axis.imshow(image_data)

    axis.set_xticks([])
    axis.set_yticks([])


# --------------------------------------------------------------------

def display_prediction(model, dataset, index, fig_size=(14, 7), file_name=None):
    """Display Input Image, True label, and Prediction from model

    Args:
        model: A model trained to make semantic segmentation
        dataset: A dataset containing images and labels
        index: The index of the image and label to display
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    img, label = dataset[index]

    img_id = dataset.get_img_id(index)

    # img needs to be in format ([1, 3, 512, 512])
    if img.ndim == 4 and img.shape[0] == 1:
        pass
    elif img.ndim == 3 and img.shape[0] == 3:
        img = img.unsqueeze(0) #add a first dimension
    else:
        raise ValueError("Wrong format of image - it needs to be in PyTorch Tensor format.")

    logits = model.forward(img.to(device))
    
    hard_preds = output_to_label(logits)
    hard_preds = hard_preds.squeeze(0).cpu().numpy()
    
    color_preds = from_int_labels_to_rgb(hard_preds)


    fig, axis = plt.subplots(1, 3, figsize=fig_size)

    display_image_tensor(axis[0], img.squeeze(0))
    axis[0].set_title("Input Image")

    display_image_tensor(axis[1], label)
    axis[1].set_title("Ground Truth")
    
    axis[2].imshow(color_preds)
    axis[2].set_title('Prediction')
    axis[2].set_xticks([])
    axis[2].set_yticks([])

    fig.suptitle(f"Ground Truth vs Prediction, Image {img_id}", fontsize=16)

    if file_name is not None:
        fig.savefig(str(file_name)+".jpg")
    
    plt.show()


# --------------------------------------------------------------------
def plot_train_val_losses(train_losses, val_losses, fig_size=(8,5), file_name=None):

    """Plot training and validation losses
    Args:
        train_losses: list
        val_losses: list

    Length of train_losses equal to:
        number_of_batches_per_epoch*number_of_epochs
    Length of val_losses equal to:
        number_of_epochs
    """

    print(f"Length of train_losses: {len(train_losses)}, Length of val_losses: {len(val_losses)}")

    # positions to plot text
    x_pos = len(train_losses)//2 + len(train_losses)//5
    y_pos = np.array(train_losses).max() - 1

    # end  -->  number of mini-batches that has been run through during training
    # +1 since we want to plot from x==1 to x==end
    end = len(train_losses)+1

    total_batches_passed = np.arange(1, end)

    # since validation loss is recorded only after each epoch
    # the range of x-values needs to be adjusted for plotting these metrics.
    # start = first x-value  -->  will be corresponding to the number of mini-batches in an epoch
    # step  -->  will be corresponding to the number of mini-batches in an epoch
    start = len(train_losses)//len(val_losses)
    step = len(train_losses)//len(val_losses)

    epochs_to_total_batches = np.arange(start, end, step)

    plt.figure(figsize=fig_size)
    plt.plot(total_batches_passed, train_losses, label="Train loss")
    plt.plot(epochs_to_total_batches, val_losses, label="Val loss")
    plt.xlabel('# of mini-batches')
    plt.ylabel('Average loss')
    plt.title("Loss during training", fontsize=15)
    plt.legend()

    plt.text(x_pos, y_pos, f"# of mini-batches: {len(train_losses)} \n# of epochs: {len(val_losses)}")

    if file_name is not None:
        plt.savefig(str(file_name)+".jpg")
    
    plt.show()
# --------------------------------------------------------------------


# --------------------------------------------------------------------


def plot_train_val_metrics(train_metrics, val_metrics, fig_size=(12, 4.5), file_name=None):

    """Plot training and validation metrics
    Args:
        train_metrics: dict
        val_metrics: dict
        fig_size: tuple

    Length of train and validation mean IoU, f1-macro, and Accuracy are equal to:
        number_of_epochs
    """
    # train_metrics["miou"]
    # train_metrics["f1_macro"]
    # train_metrics["accuracy"]
    # val_metrics["miou"]
    # val_metrics["f1_macro"]
    # val_metrics["accuracy"]
    
    # end: number of epochs
    # +1 since we want to plot from x=1 to x=end
    end = len(val_metrics["miou"])+1

    total_epochs = np.arange(1, end)


    fig, axs = plt.subplots(1,3, figsize=fig_size)
    fig.suptitle('Train and Validation Metrics', fontsize=14)
    
    axs[0].plot(total_epochs, train_metrics["miou"], label="Training data")
    axs[0].plot(total_epochs, val_metrics["miou"], label="Validation data")
    axs[0].set_xlabel('# of epochs')
    #axs[0].set_ylabel('mean IoU')
    axs[0].set_title('mean IoU')
    axs[0].legend()

    axs[1].plot(total_epochs, train_metrics["f1_macro"], label="Training data")
    axs[1].plot(total_epochs, val_metrics["f1_macro"], label="Validation data")
    axs[1].set_xlabel('# of epochs')
    #axs[1].set_ylabel('f1-macro')
    axs[1].set_title('f1-macro')
    axs[1].legend()

    axs[2].plot(total_epochs, train_metrics["accuracy"], label="Training data")
    axs[2].plot(total_epochs, val_metrics["accuracy"], label="Validation data")
    axs[2].set_xlabel('# of epochs')
    #axs[2].set_ylabel('Accuracy')
    axs[2].set_title('Accuracy')
    axs[2].legend()

    if file_name is not None:
        fig.savefig(str(file_name)+".jpg")

    plt.show()
