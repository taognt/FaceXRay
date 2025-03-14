import pickle
import os
from torchvision import transforms
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
import imgaug as ia
import imgaug.augmenters as iaa
from pytorch_lightning.callbacks import ModelCheckpoint
from pandas.core.common import flatten

# Dataset loading and preprocessing
img_paths_train = []
mask_paths_train = []
img_labels_train = []

all_paths_dir = 'blend_dataset/'
all_paths_filenames = []

for root, dirs, files in os.walk(all_paths_dir):
    for file in files:
        if file.endswith(".pkl"):
            all_paths_filenames.append(os.path.join(root, file))

print("all path filename: ", all_paths_filenames)

# Loading dataset
img_paths_train = []
mask_paths_train = []
img_labels_train = []

print("all path filename: ", all_paths_filenames[:-1])
for train_path in all_paths_filenames[:-1]:
    with open(train_path, 'rb') as alp:
        print(f"Loading {train_path}")
        all_paths_splits = pickle.load(alp)
        img_paths_train.append(all_paths_splits['image paths'])
        mask_paths_train.append(all_paths_splits['mask paths'])
        img_labels_train.append(all_paths_splits['image labels'])

img_paths_train = list(flatten(img_paths_train))
mask_paths_train = list(flatten(mask_paths_train))
img_labels_train = list(flatten(img_labels_train))

print("all path filename valid: ", all_paths_filenames[:-1])
with open(all_paths_filenames[-1], 'rb') as alp:
    all_paths_splits = pickle.load(alp)
    img_paths_valid = all_paths_splits['image paths']
    mask_paths_valid = all_paths_splits['mask paths']
    img_labels_valid = all_paths_splits['image labels']

print("\n\n--------------\n\n")
print("Total training samples : ", len(img_paths_train))
print("Total validation samples : ", len(img_paths_valid))

# Dataset class definition
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, mask_paths, img_labels):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_labels = img_labels
        # Combining torch transforms with imgaug augmentations
        self.imgaug = iaa.Sequential([
            iaa.Fliplr(0.5),  # Horizontal flip
            iaa.Affine(rotate=(-20, 20)),  # Rotation
            iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))  # Noise
        ])
        self.transform_img = transforms.Compose([
            transforms.Resize((528, 528)),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((528, 528)),
            transforms.ToTensor(),
            transforms.Normalize([0.456], [0.224])
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        mask = Image.open(self.mask_paths[index]).convert('L')

        # Apply imgaug augmentation
        image = np.array(image)
        image = self.imgaug(image=image)  # Apply imgaug augmentation
        image = Image.fromarray(image)

        image = self.transform_img(image)
        mask = self.transform_mask(mask)

        value = 1 if self.img_labels[index] == 'fake' else 0

        return image, mask, value

# Create datasets and dataloaders
train_dataset = Dataset(img_paths_train, mask_paths_train, img_labels_train)
valid_dataset = Dataset(img_paths_valid, mask_paths_valid, img_labels_valid)

train = DataLoader(train_dataset, batch_size=6, num_workers=4, drop_last=True)
valid = DataLoader(valid_dataset, batch_size=6, num_workers=4, drop_last=True)


# Classifier definition using PyTorch Lightning
class Classifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        aux_params = dict(
            pooling='max',             
            dropout=0.5,               
            activation='sigmoid',      
            classes=1,                 
        )
        self.model = smp.DeepLabV3Plus(encoder_name="efficientnet-b6", encoder_weights="imagenet", in_channels=3, classes=1, aux_params=aux_params)
        # Metrics
        self.accuracy = Accuracy(task='binary')
        self.f1_score = F1Score(task='binary')

    def forward(self, image):
        output_mask, output_target = self.model(image)
        return output_mask, output_target

    def training_step(self, batch, batch_idx):
        image, mask, target = batch
        output_mask, output_target = self.model(image)
        mask_loss = F.mse_loss(output_mask, mask)
        class_loss = F.binary_cross_entropy_with_logits(output_target.squeeze(), target.type(torch.float32).cuda())
        loss = mask_loss + class_loss

        result_dict = {
            'class_loss': class_loss,
            'mask_loss': mask_loss,
            'predictions': output_target.squeeze(),
            'targets': target,
            'total_loss': loss
        }

        return {'loss': loss, 'result': result_dict}

    def validation_step(self, batch, batch_idx):
        image, mask, target = batch
        output_mask, output_target = self.model(image)
        mask_loss = F.mse_loss(output_mask, mask)
        class_loss = F.binary_cross_entropy_with_logits(output_target.squeeze(), target.type(torch.float32).cuda())
        loss = mask_loss + class_loss

        result_dict = {
            'class_loss': class_loss,
            'mask_loss': mask_loss,
            'predictions': output_target.squeeze(),
            'targets': target,
            'total_loss': loss
        }

        return result_dict

    def on_train_epoch_end(self, train_outputs):
        outputs = [x['result'] for x in train_outputs]

        avg_class_loss = torch.stack([x['class_loss'] for x in outputs]).mean()
        avg_mask_loss = torch.stack([x['mask_loss'] for x in outputs]).mean()
        avg_loss = torch.stack([x['total_loss'] for x in outputs]).mean()

        all_predictions = torch.stack([x['predictions'] for x in outputs]).flatten()
        all_targets = torch.stack([x['targets'] for x in outputs]).flatten()

        class_accuracy = self.accuracy(all_predictions, all_targets)
        class_f1 = self.f1_score(all_predictions, all_targets)

        self.log('train_class_loss', avg_class_loss)
        self.log('train_mask_loss', avg_mask_loss)
        self.log('train_loss', avg_loss)
        self.log('train_accuracy', class_accuracy, prog_bar=True)
        self.log('train_f1', class_f1)

    def on_validation_epoch_end(self, outputs):
        avg_class_loss = torch.stack([x['class_loss'] for x in outputs]).mean()
        avg_mask_loss = torch.stack([x['mask_loss'] for x in outputs]).mean()
        avg_loss = torch.stack([x['total_loss'] for x in outputs]).mean()

        all_predictions = torch.stack([x['predictions'] for x in outputs]).flatten()
        all_targets = torch.stack([x['targets'] for x in outputs]).flatten()

        class_accuracy = self.accuracy(all_predictions, all_targets)
        class_f1 = self.f1_score(all_predictions, all_targets)

        self.log('valid_class_loss', avg_class_loss)
        self.log('valid_mask_loss', avg_mask_loss)
        self.log('valid_loss', avg_loss)
        self.log('valid_accuracy', class_accuracy, prog_bar=True)
        self.log('valid_f1', class_f1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-4,
            max_lr=1e-3,
            step_size_up=3000,
            step_size_down=3000,
            mode='triangular2',
            cycle_momentum=False
        )
        return [optimizer], [scheduler]


# Ensure this is placed at the bottom of your script
if __name__ == "__main__":
    # Your model training code here

    model = Classifier()  # Initialize the model

    # checkpoint_callback = ModelCheckpoint(
    #     monitor='valid_loss', dirpath='checkpoints/', save_top_k=1, mode='min', verbose=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath="checkpoints/",
        save_top_k=1,
        mode="min",
        verbose=True
    )
    
    # trainer = pl.Trainer(
    #     accelerator='mps',  # Use MPS on MacOS (Apple M1/M2 chip)
    #     max_epochs=15,
    #     precision="16",
    #     callbacks=[checkpoint_callback],
    #     enable_progress_bar=True
    # )

    trainer = pl.Trainer(
        accelerator="mps",  
        devices=1,          
        precision="bf16-mixed",  # Use bf16 for better MPS compatibility
        max_epochs=15,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True
    )



    trainer.fit(model, train, valid)
    input("Continue ?")
    # Save model checkpoint
    trainer.save_checkpoint("final_model.ckpt")
