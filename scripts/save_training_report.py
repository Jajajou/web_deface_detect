import os
import pickle
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Constants
MODEL_DIR = 'models'
REPORTS_DIR = 'reports'
PDF_FILENAME = os.path.join(REPORTS_DIR, 'training_report.pdf')
PLOT_FILENAME = os.path.join(REPORTS_DIR, 'training_performance.png')

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load processed data
def load_data(model_dir):
    try:
        with open(os.path.join(model_dir, 'X_train.pickle'), 'rb') as f:
            X_train = pickle.load(f)
        with open(os.path.join(model_dir, 'y_train.pickle'), 'rb') as f:
            y_train = pickle.load(f)
        with open(os.path.join(model_dir, 'X_test.pickle'), 'rb') as f:
            X_test = pickle.load(f)
        with open(os.path.join(model_dir, 'y_test.pickle'), 'rb') as f:
            y_test = pickle.load(f)
        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        exit(1)

# Load the data
X_train, y_train, X_test, y_test = load_data(MODEL_DIR)

# Convert labels to one-hot encoding if not already done
if len(y_train.shape) == 1:
    y_train = to_categorical(y_train, num_classes=5)
if len(y_test.shape) == 1:
    y_test = to_categorical(y_test, num_classes=5)

# Build a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(299, 299, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 classes for classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model_save_path = os.path.join(MODEL_DIR, 'web_defacement_model.keras')
model.save(model_save_path)

# Save the training report into a PDF
c = canvas.Canvas(PDF_FILENAME, pagesize=letter)
c.setFont("Helvetica", 12)

# Title
c.drawString(100, 750, "Training Report")
c.setFont("Helvetica", 10)
c.drawString(100, 730, f"Model Path: {model_save_path}")
c.drawString(100, 710, f"Epochs: {10}")
c.drawString(100, 690, f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
c.drawString(100, 670, f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
c.drawString(100, 650, f"Final Training Loss: {history.history['loss'][-1]:.4f}")
c.drawString(100, 630, f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")

# Plot Accuracy and Loss Graphs
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

# Save plot to a file
plt.tight_layout()
plt.savefig(PLOT_FILENAME)

# Attach the plot to the PDF
c.drawImage(PLOT_FILENAME, 100, 300, width=400, height=200)

# Save PDF
c.save()

# Display the plot
plt.show()

print(f"Training complete. Model saved at: {model_save_path}")
print(f"Training report saved at: {PDF_FILENAME}")
