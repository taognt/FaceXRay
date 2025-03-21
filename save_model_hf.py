import torch
from pytorch_lightning import Trainer
from .facexray.classifier import Classifier
from huggingface_hub import upload_file

# Load the checkpoint
checkpoint_path = "output_data/final_model.ckpt"
model = Classifier.load_from_checkpoint(checkpoint_path)

# Save the model's state_dict in PyTorch format
torch.save(model.state_dict(), "output_data/final_model.pt")

# Upload the PyTorch model (.pt file)
upload_file(
    path_or_fileobj="final_model.pt",
    path_in_repo="pytorch_model.bin",
    repo_id="TaoGnt/FaceXRay", 
    repo_type="model"  # Type of repository (model)
)
