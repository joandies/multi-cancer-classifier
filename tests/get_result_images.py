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

# Merge images vertically
def merge_images_vertically(img1, img2, output_path):
    # Calculate the total height and maximum width
    total_height = img1.height + img2.height  # The height will be the sum of both images
    max_width = max(img1.width, img2.width)   # The width will be the widest image
    # Create a new image with the calculated size
    new_image = Image.new('RGB', (max_width, total_height))
    # Paste the first image at the top
    new_image.paste(img1, (0, 0))
    # Paste the second image below the first one
    new_image.paste(img2, (0, img1.height))  # The second image is placed below the first one
    # Save the resulting image
    new_image.save(output_path)

# Save files
merge_images_vertically(training_accuracy_img, training_loss_img, '../results/figs/training_curves_vertical.png')
merge_images_vertically(validation_accuracy_img, validation_loss_img, '../results/figs/validation_curves_vertical.png')
