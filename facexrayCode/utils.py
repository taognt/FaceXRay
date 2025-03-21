import os
from matplotlib import pyplot as plt 

def get_accuracy(success, failure):
    return success / (success + failure)


def get_total_accuracy(real_success, real_failure, fake_success, fake_failure):
    success = real_success + fake_success
    failure = real_failure + fake_failure
    return success / (success + failure)


def get_metrics(fake_success, fake_failure, real_failure):
    if fake_success == 0 and real_failure == 0:
        return {
            "precision": 0,
            "recall": 0,
            "f1_score": 0
        }


    precision = get_precision(fake_success, real_failure)
    recall = get_recall(fake_success, fake_failure)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": get_f1_score(precision, recall)
    }


def get_precision(tp, fp):
    return tp / (tp + fp)


def get_recall(tp, fn):
    return tp / (tp + fn)


def get_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def visualize_and_save(image_tensor, mask_tensor, filename, output_dir='output_masks'):
    """
    Function to visualize and save the image with its predicted mask.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert tensors to CPU and detach
    image = image_tensor.cpu().detach()
    mask = mask_tensor.cpu().detach()

    # Convert image tensor to numpy array
    image = image.permute(1, 2, 0).numpy()  # Move channels to last dimension
    image = (image - image.min()) / (image.max() - image.min())  # Normalize for visualization

    # Convert mask tensor to numpy array
    mask = mask.squeeze().numpy()  # Remove batch dimension

    # Display and save the image and mask
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(mask, alpha=0.7)  # Apply colormap
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()