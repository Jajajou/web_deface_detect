import os
import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
MODEL_PATH = 'models/web_defacement_model.keras'
MODEL_DIR = 'models'
REPORTS_DIR = 'reports'
NUM_CLASSES = 5
TARGET_NAMES = ['clean', 'defaced_type_1', 'defaced_type_2', 'defaced_type_3', 'defaced_type_4']

# Load processed data
def load_data(model_dir):
    with open(os.path.join(model_dir, 'X_test.pickle'), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(model_dir, 'y_test.pickle'), 'rb') as f:
        y_test = pickle.load(f)
    return X_test, y_test

# Load test data
X_test, y_test = load_data(MODEL_DIR)

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Predict results on the test set
y_pred = model.predict(X_test, batch_size=32)

# Convert predictions to class labels
y_pred_class = np.argmax(y_pred, axis=1)  # Get the index of the class with highest probability
y_test_class = y_test  # Since y_test was not one-hot encoded

# Generate classification report
report = classification_report(y_test_class, y_pred_class, target_names=TARGET_NAMES)

# Print and save the classification report
print("\nClassification Report:")
print(report)
with open(os.path.join(REPORTS_DIR, 'classification_report.txt'), 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)

# Generate and save confusion matrix
cm = confusion_matrix(y_test_class, y_pred_class)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'))
plt.show()

print("Classification report and confusion matrix have been saved to the 'reports' directory.")
