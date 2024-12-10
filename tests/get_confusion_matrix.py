import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

confusion_matrix = np.array([[486, 13,  1,  0,  0],
                             [ 28, 455, 3,  1,  13],
                             [ 24,  5, 466, 0,  5],
                             [  0,  2,  0, 498, 0],
                             [  1,  5,  0,  0, 494]])

# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('../results/figs/confusion_matrix.png')
plt.show()
