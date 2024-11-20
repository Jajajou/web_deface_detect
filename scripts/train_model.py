import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Constants
MODEL_SAVE_PATH = 'models/web_defacement_model.keras'
MODEL_DIR = 'models'
REPORTS_DIR = 'reports'
NUM_CLASSES = 5
INPUT_SHAPE = (299, 299, 3)
BATCH_SIZE = 32  # Increased batch size
EPOCHS = 10

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load processed data
def load_data(model_dir):
    with open(os.path.join(model_dir, 'X_train.pickle'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(model_dir, 'y_train.pickle'), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join(model_dir, 'X_test.pickle'), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(model_dir, 'y_test.pickle'), 'rb') as f:
        y_test = pickle.load(f)
    return X_train, y_train, X_test, y_test

# Load the data
X_train, y_train, X_test, y_test = load_data(MODEL_DIR)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
"""
# Build the complex CNN model
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(0.05)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.6))

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.05)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.6))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.05)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.7))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.05)))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Compile the model
model = build_model(INPUT_SHAPE, NUM_CLASSES)
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])  # Increased learning rate for faster convergence


# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)  # Increased patience
model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1)  # More aggressive learning rate reduction

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Save the final model
model.save(MODEL_SAVE_PATH)
"""


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

# Plot training history (accuracy and loss)
def plot_training_history(history, report_dir):
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

    # Save and display the plots
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'training_performance.png'))
    plt.show()

plot_training_history(history, REPORTS_DIR)

print("Training complete. The model and training report have been saved.")
