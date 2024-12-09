# Cervical Cancer Subclass Classifier
This project aims to classify images into 5 subcategories of Cervical Cancer using a deep learning model built with PyTorch. The model uses a pre-trained ResNet50, fine-tuned on a custom dataset, to predict cervical cancer subtypes.

This project is part of a scalable framework that can later include classifiers for additional cancer types and subclasses.

## Features
- Subclass classification for Cervical Cancer images: The model classifies cervical cancer images into 5 distinct subclasses.
- Scalable design: The framework is designed to be extended with classifiers for other types of cancer and their respective subclasses.
- Real-time training monitoring: Integration with Weights & Biases (wandb.ai) to monitor training and validation metrics in real time.
- Image preprocessing pipeline: The dataset undergoes transformations such as resizing, center cropping, and normalization, similar to ImageNet.
- Model evaluation: The model is evaluated on the test set using metrics like Accuracy, Precision, Recall, and F1-Score.

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/joandies/cervical-cancer-classifier.git
cd cervical-cancer-classifier
pip install -r requirements.txt
```
## Prerequisites
- Python 3.6 or higher
- PyTorch 1.9.0 or higher
- Weights & Biases account (for training monitoring)

## Prepare the dataset
To train and test the model, you need to prepare the cervical cancer dataset. Follow these steps to download and preprocess the data.
### 1. Download the dataset:
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/obulisainaren/multi-cancer/data)
### 2. Organize the data:
The original dataset is already structured, but make sure it's organized into the following directories:
```bash
train/
validation/
test/
```
This structure is required for the ```prepare_cervical_data.py``` script to work properly.
### 3. Run the data preparation script
Run the following script to organize the data into the appropriate folder structure. You can provide the `--input_dir` and `--output_dir` paths directly from the console, or it will use the paths from `config.yaml` by default.
```bash
python -m src.data.prepare_data --input_dir <input_dir> --output_dir <output_dir>
```
If you want to use the paths from the `config.yaml`, simply run:
```bash
python -m src.data.prepare_data
```
The script will create the following structure in the output directory:
```bash
output_dir
    └── train/
    └── validation/
    └── test/
```
Make sure to update the paths in the config.yaml file to point to your local dataset directory.

## How to use
### 1. Train the model
To train the model, use the following command:
```bash
python -m src/training/train
```
The training script will load the dataset, apply the necessary transformations, and begin training the model. You can track the training progress via [wandb.ai](wandb.ai).
### 2. Test the model
To test the model on a new image, use the following Jupyter notebook:
- Open notebooks/test_model.ipynb in a Jupyter environment.
- Follow the instructions in the notebook to load the model, make predictions, and visualize the results.
### 3. Evaluate the model
The model's performance can be evaluated using standard metrics such as Accuracy, Precision, Recall, and F1-Score. These metrics are automatically logged to wandb during training and can be accessed for review.

## Model structure
- ResNet50 Pretrained: The model architecture starts with a pre-trained ResNet50, which is fine-tuned for cervical cancer subclass classification.
- Custom Head: A custom fully connected layer is added on top of the ResNet50 to predict the 5 subclasses.

## Example workflow
### 1. Load the model
The model is loaded with the pre-trained weights from a checkpoint file:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CervicalModel(num_classes=5)
checkpoint_path = 'path_to_checkpoint/best_checkpoint.pth'
model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
model.to(device)
model.eval()
```
### 2. Make predictions
Once the model is loaded, you can use it to make predictions on the test dataset. Here's an example of predicting the class for one image:
```python
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Move image to device
images, labels = images.to(device), labels.to(device)

# Get the model's prediction
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Print the predicted class
print(f'Predicted class: {test_dataset.classes[predicted]}')
```
### 3. Visualize results
You can visualize the input image and the model's prediction using matplotlib:
```python
imshow(images[0])  # To show the image
```
## Contributing
Contributions are welcome! If you have any improvements or features you would like to suggest, feel free to fork this repository and submit a pull request.

## License
This project is licensed under the MIT License.