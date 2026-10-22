import torch
import torch.nn as nn
import segmentation_models_pytorch as smp



class DiceFocalLoss(nn.Module):
    """A loss function that combines Dice and Focal losses.

    Combining loss function from the Segmentation Models PyTorch library (smp) 
    to create a custom loss function that combines the strengths of both Dice Loss and Focal Loss.
    
    The Dice Loss is particularly effective for segmentation tasks, especially when dealing with imbalanced datasets.
    It measures the overlap between the predicted segmentation and the ground truth, making it suitable for multi-class 
    segmentation problems.
    
    The Focal Loss is designed to address class imbalance by down-weighting easy examples and focusing training on hard negatives.
    The `gamma` parameter controls the focusing effect; a higher value means more focus on hard-to-classify examples.
    
    The weights `w1` and `w2` allow for adjusting the contribution of each loss component to the final loss value, 
    providing flexibility in training.
    """
    
    def __init__(self, w1=1.0, w2=1.0):
        super(DiceFocalLoss, self).__init__()

        # Define the Dice Loss component
        self.lossDice = smp.losses.DiceLoss(mode='multiclass', log_loss=True, from_logits=True, smooth=1e-5)
        
        # Define the Focal Loss component 
        self.lossFocal = smp.losses.FocalLoss(mode='multiclass', gamma=2.0)
        
        self.w1 = w1    # Weight for loss1
        self.w2 = w2    # Weight for loss2


    def forward(self, y_pred, y_true):
        """A forward pass that computes the combined loss.
        y_pred: [Batch, 6, Height, Width] (Raw logits from U-Net)
        y_true: [Batch, Height, Width] (Integer class labels 0 to 5)
        """
        # Calculate individual losses
        dice_loss = self.lossDice(y_pred, y_true)
        focal_loss = self.lossFocal(y_pred, y_true)
        
        # Return weighted sum
        return (self.w1 * dice_loss) + (self.w2 * focal_loss)




def output_to_label(logits):
    """Map network output logits to a hard labels {0, 1, 2, 3, 4, 5} for each pixel in a batch.

    Args:
        logits (Tensor, (batch_size, num_classes, H, W)): Probabilities for each sample in a batch.
    Returns:
        pred (Tensor, (batch_size, H, W)): Hard label {0, 1, 2, 3, 4, 5} for each pixel in a batch
    """
    # Use argmax to find the class with the highest logit for each pixel
    # dim=1 because the class dimension is the second dimension in the tensor
    # Convert to long type for compatibility with loss functions
    hard_preds = torch.argmax(logits, dim=1).long()
    return hard_preds





def training_loop(
    model, optimizer, loss_fn, train_loader, val_loader, num_epochs, scheduler=None, print_every=None):

    best_loss = float("inf")
    best_miou = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_metrics = {"loss": [], "miou": [] , "f1_macro": [], "accuracy": [], "lr": []}
    val_metrics = {"loss": [], "miou": [] , "f1_macro": [], "accuracy": []}


    print("Starting training")
    
    #  ---- TRAINING STARTS ----
    #      Loop over epochs
    for epoch in range(1, num_epochs + 1):

        # Train for one epoch
        model, train_loss, train_miou, train_f1_macro, train_accuracy = train_epoch(
            model, optimizer, loss_fn, train_loader, val_loader, device, print_every
        )

        print(
            f"Epoch {epoch}/{num_epochs}: \n"
            f"Avg. Train loss: {sum(train_loss)/len(train_loss):.3f},  "
            f"Train mean IoU: {train_miou:.3f}, "
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.5f}, ")


        # running the validate function will:
        # collect validation metrics,
        # print validation loss and mean IoU for this epoch,
        # print per-class IoU values
        val_loss, val_miou, val_f1_macro, val_accuracy = validate(model, loss_fn, val_loader, device)


        # train losses stored for each mini-batch
        train_metrics["loss"].extend(train_loss)

        # Remaining train metrics stored for each epoch
        train_metrics["miou"].append(train_miou)
        train_metrics["f1_macro"].append(train_f1_macro)
        train_metrics["accuracy"].append(train_accuracy)
        train_metrics["lr"].append(optimizer.param_groups[0]['lr'])

        # val losses/metrics stored for each epoch
        val_metrics["loss"].append(val_loss)
        val_metrics["miou"].append(val_miou)
        val_metrics["f1_macro"].append(val_f1_macro)
        val_metrics["accuracy"].append(val_accuracy)


        # Update the learning rate scheduler if provided
        # Step the scheduler based on validation loss
        if scheduler is not None:
            scheduler.step(val_loss)  


        # Save the model if it has the best validation loss or best validation mIoU
        if epoch > 10 and val_loss < best_loss and val_miou > best_miou:
            best_loss = val_loss
            best_miou = val_miou
            torch.save(model.state_dict(), f"{model.__class__.__name__}_best_model.pth")
            print(f"Best model saved after epoch {epoch}. \n")

    print("Training completed")
    return model, train_metrics, val_metrics



def train_epoch(
    model, optimizer, loss_fn, train_loader, val_loader, device, print_every):
    
    # Train:
    model.train()
    train_loss_batches = []
    tp_epoch, fp_epoch, fn_epoch, tn_epoch = [], [], [], []

    num_batches = len(train_loader)
    
    for batch_index, (x, y) in enumerate(train_loader, 1):

        inputs, labels = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        logits = model.forward(inputs)
        loss = loss_fn(logits, labels)
        
        loss.backward()
        optimizer.step()

        train_loss_batches.append(loss.item())

        hard_preds = output_to_label(logits)

        # Calculate the mean intersection over union (mIoU) for this batch
        # Compute pixel statistics
        tp, fp, fn, tn = smp.metrics.get_stats(
                hard_preds, labels, mode="multiclass", num_classes=model.n_class)
        
        tp_epoch.append(tp)
        fp_epoch.append(fp)
        fn_epoch.append(fn)
        tn_epoch.append(tn)

        # end of batch loop


        # print(f"Batch {batch_index}/{num_batches} completed")

        # insert "print every" logic here if wanted
        # code for this is commented out at the bottom of this file, 
        # since it slows down training significantly


    # Compute the mean IoU for the entire epoch
    tp_total = torch.cat(tp_epoch, dim=0)
    fp_total = torch.cat(fp_epoch, dim=0)
    fn_total = torch.cat(fn_epoch, dim=0)
    tn_total = torch.cat(tn_epoch, dim=0)

    train_miou = smp.metrics.iou_score(tp_total, fp_total, fn_total, tn_total, reduction="macro").item()

    train_f1_macro = smp.metrics.f1_score(tp_total, fp_total, fn_total, tn_total, reduction="macro").item()
    
    train_accuracy = smp.metrics.accuracy(tp_total, fp_total, fn_total, tn_total, reduction="micro").item()

    return model, train_loss_batches, train_miou, train_f1_macro, train_accuracy




def validate(model, loss_fn, val_loader, device):
    val_loss_cum = 0
    tp_val, fp_val, fn_val, tn_val = [], [], [], []
    
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)
            logits = model.forward(inputs)

            batch_loss = loss_fn(logits, labels)
            val_loss_cum += batch_loss.item()
            
            hard_preds = output_to_label(logits)

            # Calculate the mean intersection over union (mIoU) for this batch
            # Compute pixel statistics
            tp, fp, fn, tn = smp.metrics.get_stats(
                hard_preds, labels, mode="multiclass", num_classes=model.n_class)

            tp_val.append(tp)
            fp_val.append(fp)
            fn_val.append(fn)
            tn_val.append(tn)

            # end of batch loop

    # Compute the average validation loss over all batches
    val_loss = val_loss_cum / len(val_loader)

    # Compute the mean IoU for the entire epoch
    tp_total = torch.cat(tp_val, dim=0)
    fp_total = torch.cat(fp_val, dim=0)
    fn_total = torch.cat(fn_val, dim=0)
    tn_total = torch.cat(tn_val, dim=0)

    val_miou = smp.metrics.iou_score(tp_total, fp_total, fn_total, tn_total, reduction="macro").item()

    val_f1_macro = smp.metrics.f1_score(tp_total, fp_total, fn_total, tn_total, reduction="macro").item()

    val_accuracy = smp.metrics.accuracy(tp_total, fp_total, fn_total, tn_total, reduction="micro").item()

    
    print(f"Avg. Val. loss: {val_loss:.3f}, "
          f"Val. mean IoU: {val_miou:.3f}"
        )

    # Get per-class IoU values to see which specific class performace
    tp_sum = tp_total.sum(dim=0)
    fp_sum = fp_total.sum(dim=0)
    fn_sum = fn_total.sum(dim=0)
    per_class_iou = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-7)

    print("\nPer-class IoU of Validation Set:")
    for class_idx, score in enumerate(per_class_iou):
        print(f"  -> Class {class_idx} IoU: {score.item():.4f}")
    print()

    return val_loss, val_miou, val_f1_macro, val_accuracy



def confusion_matrix_6x6(hard_preds, labels):
    """Return a 6x6 confusion matrix for one batch of segmentation labels.

    Rows represent the true class and columns represent the predicted class.
    """
    hard_preds = hard_preds.reshape(-1).long()
    labels = labels.reshape(-1).long()
    idx = labels * 6 + hard_preds
    
    matrix = torch.bincount(idx, minlength=6**2).reshape(6, 6)
    
    return matrix



def test_score(model, loss_fn, test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    conf_matrix = torch.zeros((6, 6), dtype=torch.int64)

    tp_, fp_, fn_, tn_ = [], [], [], []
    
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(test_loader, 1):
            inputs, labels = x.to(device), y.to(device)
            logits = model.forward(inputs)

            # No need for loss of test data
            #batch_loss = loss_fn(logits, labels)

            hard_preds = output_to_label(logits)

            # Calculate the mean intersection over union (mIoU) for this batch
            # Compute pixel statistics
            tp, fp, fn, tn = smp.metrics.get_stats(
                hard_preds, labels, mode="multiclass", num_classes=model.n_class)

            tp_.append(tp)
            fp_.append(fp)
            fn_.append(fn)
            tn_.append(tn)

            conf_matrix += confusion_matrix_6x6(hard_preds, labels).cpu()

            # end of batch loop



    # Compute the mean IoU for the entire epoch
    tp_total = torch.cat(tp_, dim=0)
    fp_total = torch.cat(fp_, dim=0)
    fn_total = torch.cat(fn_, dim=0)
    tn_total = torch.cat(tn_, dim=0)

    test_miou = smp.metrics.iou_score(tp_total, fp_total, fn_total, tn_total, reduction="macro").item()

    test_f1_macro = smp.metrics.f1_score(tp_total, fp_total, fn_total, tn_total, reduction="macro").item()

    test_accuracy = smp.metrics.accuracy(tp_total, fp_total, fn_total, tn_total, reduction="micro").item()

    
    print(f"Test mean IoU: {test_miou:.3f}")
    print(f"Test F1 Score (macro): {test_f1_macro:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")

    # Get per-class IoU values to see which specific class performace
    tp_sum = tp_total.sum(dim=0)
    fp_sum = fp_total.sum(dim=0)
    fn_sum = fn_total.sum(dim=0)
    per_class_iou = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-7)

    print("\nPer-class IoU of Test set:")
    for class_idx, score in enumerate(per_class_iou):
        print(f"  -> Class {class_idx} IoU: {score.item():.4f}")
    print()


    return conf_matrix, test_miou, test_f1_macro, test_accuracy




# print every logic for training loop is commented out below.

# If you want to print your progress more often than every epoch you can
# set `print_every` to the number of batches you want between every status update.
# Note that the print out will trigger a full validation on the full val. set => slows down training

# if print_every is not None and batch_index % print_every == 0:
#     val_loss, val_miou = validate(model, loss_fn, val_loader, device)
#     model.train()
#     print(
#         f"\tBatch {batch_index}/{num_batches}: "
#         f"\tTrain loss: {sum(train_loss_batches[-print_every:])/print_every:.3f}, "
#         f"\tTrain mIoU: {train_miou:.3f}, "
#         f"\tVal. loss: {val_loss:.3f}, "
#         f"\tVal. mIoU: {val_miou:.3f}"
#     )