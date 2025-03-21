import torch

import sys
import os

from facexray.classifier import Classifier 
from facexray.dataset import transform_img
from datasets import load_dataset as ds_load_dataset
import os
from torchvision import transforms
from PIL import Image

# Define your model
model = Classifier()

# Load the model state dict from Hugging Face's repo (or a local path)
checkpoint_url = "https://huggingface.co/TaoGnt/FaceXRay/resolve/main/pytorch_model.bin"
checkpoint_path = "pytorch_model.bin"

# If you want to load the model directly from Hugging Face, you can use requests
import requests
response = requests.get(checkpoint_url)
with open(checkpoint_path, 'wb') as f:
    f.write(response.content)

# Now load the state_dict into your model
model.load_state_dict(torch.load(checkpoint_path))

# Move the model to the appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Test inference
def inference(image, model, device=None, transform=transform_img, output_path="output_data", save=False, verbose=True):

    if save:
        os.makedirs(output_path, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert image to RGB if it has RGBA
    if isinstance(image, Image.Image): 
        image = image.convert("RGB") 

    # Set model to evaluation mode (important for inference)
    model.eval()

    # Convert PIL image to tensor if necessary
    if isinstance(image, torch.Tensor):
        image_tensor = image.unsqueeze(0)  # Add batch dimension
    else:
        image_tensor = transform(image).unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output_mask, output_target = model(image_tensor)
        
    # Convert logits to probability
    predicted_prob = torch.sigmoid(output_target).item()
    
    # Convert probability to binary prediction (Threshold = 0.55)
    predicted_class = 1 if predicted_prob >= 0.55 else 0
    
    # Save mask
    if save:
        fake_mask_output_dir = os.path.join(output_path, 'inference_mask')
        os.makedirs(fake_mask_output_dir, exist_ok=True)
        filename = f"fake_mask_test.png"
        visualize_and_save(image_tensor.squeeze(0), output_mask.squeeze(0), filename, fake_mask_output_dir)

    mask_numpy = output_mask.squeeze().cpu().detach().numpy()

    # Print results
    if verbose:
        print(f"Actual label: {'Fake' if label == 1 else 'Real'}")
        print(f"Predicted class: {'Fake' if predicted_class == 1 else 'Real'}")
        print(f"Confidence Score: {predicted_prob:.4f}")

    return mask_numpy, predicted_class


# Test image
# k = 200
# ds = ds_load_dataset("Supervache/deepfake_celebaHQ") # Used for test
# e4s_images = ds.filter(lambda x: x['model'] == 'e4s', load_from_cache_file=False)
# ds = e4s_images['train']
# image_data = ds[k]['image']
# label = ds[k]['fake']

# inference(image_data, model, device=device)