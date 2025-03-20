PART=ENSTA-l40s #ENSTA-h100 #ENSTA-l40s
TIME=04:00:00


# Variables
FILE_ID=17VRoaavNg0JH4tZLOZC7yHZ9t3-tP1rR
FILE_ID_TRAIN=1iDJnoCobRXZh-qjSWubGO3Jz3WMQr67F
FILE_ID_VAL=1fwnBNdRtBcLdPmzc217NYOHGzDoAU7Yt
FILE_NAME=dataset.zip

# Phony target
.PHONY: download-dataset

# Target to download and extract the dataset
download-dataset:
	@echo "Downloading dataset from Google Drive..."
	gdown --id $(FILE_ID) -O $(FILE_NAME)
	@echo "Extracting dataset..."
	unzip $(FILE_NAME) -d .  # Extract directly to the root
	@echo "Cleaning up..."
	rm $(FILE_NAME)  # Remove the .zip file
	@echo "Dataset downloaded and extracted successfully!"

dowloads: download-dataset-train download-dataset-val

download-dataset-train:
	@echo "Downloading dataset from Google Drive..."
	gdown --id $(FILE_ID_TRAIN) -O $(FILE_NAME)
	@echo "Extracting dataset..."
	unzip $(FILE_NAME) -d .  # Extract directly to the root
	@echo "Cleaning up..."
	rm $(FILE_NAME)  # Remove the .zip file
	@echo "Dataset downloaded and extracted successfully!"

download-dataset-val:
	@echo "Downloading dataset from Google Drive..."
	gdown --id $(FILE_ID_VAL) -O $(FILE_NAME)
	@echo "Extracting dataset..."
	unzip $(FILE_NAME) -d .  # Extract directly to the root
	@echo "Cleaning up..."
	rm $(FILE_NAME)  # Remove the .zip file
	@echo "Dataset downloaded and extracted successfully!"

run:
	export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
	export TF_ENABLE_ONEDNN_OPTS=0
	srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python train_model.py

test:
	export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
	export TF_ENABLE_ONEDNN_OPTS=0
	srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python test_model.py

save-model:
	export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
	export TF_ENABLE_ONEDNN_OPTS=0
	srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python save_model_hf.py

test-inference:
	export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
	export TF_ENABLE_ONEDNN_OPTS=0
	srun --pty --time=$(TIME) --partition=$(PART) --gpus=1 python test_inference_hf.py