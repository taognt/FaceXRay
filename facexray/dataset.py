from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset

transform_img = transforms.Compose([
            transforms.Resize((528, 528)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class DatasetTest(Dataset):
    def __init__(self, dataset, transform=transform_img):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        label = sample["fake"]  # 0 (real), 1 (fake)

        # Convert RGBA to RGB if needed
        if image.mode == "RGBA":
            image = image.convert("RGB")
        elif image.mode == "L":  # If grayscale, convert to RGB
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
