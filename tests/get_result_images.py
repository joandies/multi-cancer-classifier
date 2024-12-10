from PIL import Image
import os

# Images path
images_path = '../results/figs/'

training_accuracy_path = images_path + 'Training Accuracy.png'
training_loss_path = images_path + 'Training Loss.png'
validation_accuracy_path = images_path + 'Validation Accuracy.png'
validation_loss_path = images_path + 'Validation Loss.png'

# Load images
training_accuracy_img = Image.open(training_accuracy_path)
training_loss_img = Image.open(training_loss_path)
validation_accuracy_img = Image.open(validation_accuracy_path)
validation_loss_img = Image.open(validation_loss_path)

# Merge images horizontally
def merge_images_horizontally(img1, img2, output_path):
    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)
    new_image = Image.new('RGB', (total_width, max_height))
    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (img1.width, 0))
    new_image.save(output_path)

# Save files
merge_images_horizontally(training_accuracy_img, training_loss_img, '../results/figs/training_curves.png')
merge_images_horizontally(validation_accuracy_img, validation_loss_img, '../results/figs/validation_curves.png')
