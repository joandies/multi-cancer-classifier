import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import yaml
from src.models.cervical_cancer_model import CervicalModel
import wandb
import argparse

def main(config_path):
    # Load configuration from config.yaml
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize wandb
    wandb.init(project='cervical-cancer-classifier', config=config)

    # Check if the train and validation dataset path exists
    train_dataset_path = config['train_dataset_path']
    if not os.path.exists(train_dataset_path):
        raise FileNotFoundError(f"The path {train_dataset_path} does not exist. Please check your config.yaml.")
    val_dataset_path = config['val_dataset_path']
    if not os.path.exists(val_dataset_path):
        raise FileNotFoundError(f"The path {val_dataset_path} does not exist. Please check your config.yaml.")


    # Number of classes (subclasses of cervical cancer)
    num_classes = config['num_classes']

    # Initialize the model
    model = CervicalModel(num_classes)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the transformations for the images
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image
        transforms.CenterCrop(224),  # Center crop to 224x224 to match model input
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet values
    ])

    # Load the training and validation dataset
    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dataset_path, transform=transform)
    


    # DataLoader to load images in batches
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define the loss function and optimizer
    lr = config['learning_rate']
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

    # Training loop
    epochs = config['epochs']  # Get the number of epochs

    # Early Stopping parameters
    early_stopping_patience = config['early_stopping_patience']  # Número de épocas sin mejora
    best_loss = float('inf')
    epochs_without_improvement = 0

    # Initialize model values
    checkpoint_path = config['checkpoints_path']
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_path, 'best_checkpoint.pth')


    print(f'Training started for {epochs} epochs')
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients to avoid accumulation from previous iterations
            optimizer.zero_grad()

            # Forward pass: Compute predicted outputs by passing inputs to the model
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass: Compute gradients
            loss.backward()

            # Optimize the model's parameters
            optimizer.step()

            # Track the loss and accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest probability
            _, predicted = torch.max(outputs, 1)

            # Update the total and correct counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        # Validation at the end of each epoch
        model.eval()  # Change model to eval mode
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        # Save checkpoint if validation improved
        if val_loss < best_loss:
            print(f"Validation loss improved ({best_loss:.4f} -> {val_loss:.4f}). Saving checkpoint...")
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_file)
        else:
            epochs_without_improvement += 1

        # Early stopping: If the validation loss does not improve after 'patience' epochs, training will stop early
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        # Log to wandb
        wandb.log({
        'epoch': epoch + 1,
        'Training loss': epoch_loss, 
        'Training accuracy': epoch_accuracy, 
        'Validation loss': val_loss, 
        'Validation accuracy': val_accuracy
    })

        # Print the results for this epoch
        print(f"Epoch {epoch+1}/{epochs}\nTraining Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}% \nValidation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    print("Training complete!")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a cervical cancer classifier.')
    parser.add_argument('--config', type=str, default='../../config.yaml', help='Path to the config file')
    parser.add_argument('--project_name', type=str, default='cervical-cancer-classifier', help='WandB project name')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    wandb.init(project=args.project_name)
    main(args.config)
