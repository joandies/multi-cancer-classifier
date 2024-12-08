import torch
import os
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.cervical_cancer_model import CervicalModel

def main(config_path):
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config.yaml'))
    # Load configuration from config.yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Number of classes (subclasses of cervical cancer)
    num_classes = config['num_classes']

    # Initialize the model
    model = CervicalModel(num_classes)
    
    # Load the best model checkpoint
    checkpoint_path = os.path.join(config['checkpoints_path'], 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Define the same image transformations as in training
    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])

    # Load the test dataset
    test_dataset_path = config['test_dataset_path']
    if not os.path.exists(test_dataset_path):
        raise FileNotFoundError(f"The path {test_dataset_path} does not exist. Please check your config.yaml.")
        
    test_dataset = datasets.ImageFolder(root=test_dataset_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    all_labels = []
    all_preds = []

    print(f"Evaluating the model on {len(test_dataset)} test images...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Generate the classification report
    target_names = test_dataset.classes  # Class names from ImageFolder
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print(report)

    # Generate the confusion matrix
    print("\nConfusion Matrix:")
    confusion = confusion_matrix(all_labels, all_preds)
    print(confusion)

    # Save the classification report to a file (optional)
    report_path = os.path.join(config['reports_path'], 'classification_report.txt')
    os.makedirs(config['reports_path'], exist_ok=True)
    with open(report_path, 'w') as file:
        file.write("Classification Report:\n")
        file.write(report)
        file.write("\n\nConfusion Matrix:\n")
        file.write(str(confusion))
    
    print(f"Classification report saved at: {report_path}")

if __name__ == '__main__':
    config_path = '../../config.yaml'
    main(config_path)
