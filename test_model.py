import pickle
from torchvision import datasets, models, transforms
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# from torchmetrics.functional.classification import accuracy
import itertools
from pytorch_lightning.callbacks import ModelCheckpoint
import imgaug as ia
import imgaug.augmenters as iaa
import os
from pandas.core.common import flatten
from datasets import load_dataset as ds_load_dataset
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import statistics
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import cv2
from datasets import concatenate_datasets
from tqdm import tqdm
from collections import Counter

from facexray.classifier import Classifier
from facexray.dataset import transform_img, DatasetTest
from code.utils import visualize_and_save

# NVIDIA H100 has Tensor Cores which can improve performance, to enable:
torch.set_float32_matmul_precision('high')  # 'medium' or 'high' for best speed
num_workers = min(8, os.cpu_count() // 2)  # Limits workers to 8 max


# Threshold evaluation

def find_best_threshold(predicted_probs, true_labels):
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.0, 1.05, 0.05)  # Try thresholds from 0 to 1 with a step of 0.05

    # Convert predicted_probs and true_labels to tensors for comparison in PyTorch
    predicted_probs_tensor = torch.tensor(predicted_probs)
    true_labels_tensor = torch.tensor(true_labels)

    for threshold in thresholds:
        # Convert to binary predictions using threshold
        predicted_classes = (predicted_probs_tensor >= threshold).int()
        
        # Calculate the F1 score
        f1 = f1_score(true_labels_tensor.numpy(), predicted_classes.numpy())
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold

def most_represented_number(values):
    # Round each value to 2 decimal places
    rounded_values = [round(val, 2) for val in values]
    
    # Count the frequency of each rounded value
    count = Counter(rounded_values)
    
    # Find the value with the highest frequency
    most_common_value, _ = count.most_common(1)[0]
    
    return most_common_value

def visualize_and_save_contour(image_tensor, mask_tensor, filename, output_dir='output_masks'):
    """
    Function to visualize and save the image with a green contour indicating modified areas.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert tensors to CPU and detach
    image = image_tensor.cpu().detach().permute(1, 2, 0).numpy()  # Move channels to last dimension
    mask = mask_tensor.cpu().detach().squeeze().numpy()  # Remove batch dimension

    # Normalize image for visualization
    image = (image - image.min()) / (image.max() - image.min())  # Normalize between 0 and 1
    image = (image * 255).astype(np.uint8)  # Convert to uint8

    # Convert grayscale mask to binary (thresholding)
    mask = (mask > 0.5).astype(np.uint8)  # Binarize the mask

    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert image to BGR format for OpenCV (needed for color drawing)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw contours on the image (green color)
    cv2.drawContours(image_bgr, contours, -1, (0, 255, 0), thickness=2)

    # Convert back to RGB format
    image_final = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Save the image
    filepath = os.path.join(output_dir, filename)
    print(filepath)
    plt.imsave(filepath, image_final)

    print(f"Image saved at: {filepath}")

def eval(model, dataloader, device, output_path=None, model_to_test='e4s', data='celeba'):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    best_thresholds = []
    log_file_path = None

    nb_mask = 10 # nb mask to save per batch

    if output_path is not None:
        log_file_path = os.path.join(output_path, "training_log.txt")

    if log_file_path is not None:
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"\n\n========================== Starting Test for {data}\n")
            log_file.write(f"Generation model tested: {model_to_test}\n")

    # Output directory for fake masks
    fake_mask_output_dir = os.path.join(output_path, f"fake_masks_{data}_{model_to_test}")
    os.makedirs(fake_mask_output_dir, exist_ok=True)

    # Initialize tqdm progress bar
    with tqdm(enumerate(dataloader), desc="Evaluating", unit=" batch") as pbar:
        for batch_idx, (image, label) in pbar:
            mask_id = 0
            image = image.to(device)
            label = label.to(device)
            
            with torch.no_grad():
                output_mask, output_target = model(image)
                
                # Convert logits to probabilities
                # predicted_probs = torch.sigmoid(output_target)  
                predicted_probs = output_target

                # Find the best threshold based on the predicted probabilities and true labels
                best_threshold = find_best_threshold(predicted_probs.cpu().numpy(), label.cpu().numpy())

                best_thresholds.append(best_threshold)

                # Convert to binary predictions
                predicted_classes = (predicted_probs >= 0.5).int()

                # Save masks for fake images (label == 1)
                for i in range(image.shape[0]): 
                    if mask_id < nb_mask:
                        if predicted_classes[i].item() == 1:  # Only save masks for fake images
                            filename = f"fake_mask{model_to_test}{data}{batch_idx}{mask_id}.png"
                            visualize_and_save(image[i], output_mask[i], filename, fake_mask_output_dir)
                            mask_id += 1

                # Append predictions and true labels to the lists
                all_preds.extend(predicted_classes.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

            # Flatten the lists of predictions and labels for metrics
            all_preds_flat = np.array(all_preds).flatten()
            all_labels_flat = np.array(all_labels).flatten()

            # Calculate the evaluation metrics after every batch
            accuracy = accuracy_score(all_labels_flat, all_preds_flat)
            precision = precision_score(all_labels_flat, all_preds_flat)
            recall = recall_score(all_labels_flat, all_preds_flat)
            f1 = f1_score(all_labels_flat, all_preds_flat, zero_division=1)

            # Update the tqdm progress bar with the latest scores
            pbar.set_postfix(
                accuracy=f"{accuracy:.4f}", 
                precision=f"{precision:.4f}", 
                recall=f"{recall:.4f}", 
                f1=f"{f1:.4f}"
            )

    # Final metrics after the full evaluation
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    print(f"Final Precision: {precision:.4f}")
    print(f"Final Recall: {recall:.4f}")
    print(f"Final F1 Score: {f1:.4f}")
    print(f"Final best threshold: {statistics.mean(best_thresholds)}")
    print(f"Final best threshold: {most_represented_number(best_thresholds)}")

    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Final Accuracy: {accuracy:.4f}\n")
        log_file.write(f"Final Precision: {precision:.4f}\n")
        log_file.write(f"Final Recall: {recall:.4f}\n")
        log_file.write(f"Final F1 Score: {f1:.4f}\n")

    # Generate Confusion Matrix
    if output_path is not None:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {model_to_test} with {data} dataset')

        # Save confusion matrix to the same directory as the log file
        cm_filename = f"confusion_matrix_{model_to_test}_{data}.png"
        cm_filepath = os.path.join(output_path, cm_filename)
        plt.savefig(cm_filepath)
        plt.close()

        print(f"Confusion matrix saved at: {cm_filepath}")

        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Confusion matrix saved at: {cm_filepath}\n")


def main(args):
    output_path = Path(args.output_path)

    # Create output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Log file
    log_file_path = os.path.join(output_path, "training_log.txt")

    # Load the trained model checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Classifier.load_from_checkpoint("output_data/final_model.ckpt")

    ds = ds_load_dataset("florian-morel22/deepfake-celeba") # Used for test
    ds = ds['train']
    e4s_images = ds.filter(lambda x: x['model'] == 'e4s', load_from_cache_file=False) #e4s
    real_images = ds.filter(lambda x: x['model'] == 'Real', load_from_cache_file=False) #Real
    reface_images = ds.filter(lambda x: x['model'] == 'REFace') #REFace
    real_e4s_images = ds.filter(lambda x: x['model'] != 'REFace') #Real, e4s
    real_reface_images = ds.filter(lambda x: x['model'] != 'e4s') #Real, e4s

    list_ds_image = {'e4s':e4s_images, 'real':real_images, 'reface':reface_images, 'e4s_real':real_e4s_images, 'reface_real': real_reface_images, 'all':ds}

    for key, ds_images in list_ds_image.items():
        dataset = ds_images
        test_dataset = DatasetTest(dataset, transform=transform_img)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)

        eval(model, test_loader, device, output_path, model_to_test=key, data='celeba')

    ds_hq = ds_load_dataset("Supervache/deepfake_celebaHQ")
    e4s_images = ds_hq['train']
    filtered_ds = concatenate_datasets([real_images, e4s_images])

    # Shuffle the dataset
    filtered_ds = filtered_ds.shuffle(seed=42)  # Set seed for reproducibility
    list_ds_image = {'all':filtered_ds}

    for key, ds_images in list_ds_image.items():
        dataset = ds_images
        test_dataset = DatasetTest(dataset, transform=transform_img)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=8)

        eval(model, test_loader, device, output_path, model_to_test=key, data='celeba_HQ')

# -- Test   
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path",
        type=str,
        default="./output_data",
    )

    args = parser.parse_args()
    
    main(args)








