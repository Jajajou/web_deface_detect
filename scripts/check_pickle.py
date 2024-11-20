import pickle
import matplotlib.pyplot as plt
import os

# Constants
MODEL_DIR = 'models'
NUM_SAMPLES_TO_DISPLAY = 10

# Load training data from pickle files
def load_data(model_dir, data_type='train'):
    if data_type == 'train':
        with open(os.path.join(model_dir, 'X_train.pickle'), 'rb') as f:
            X_loaded = pickle.load(f)
        with open(os.path.join(model_dir, 'y_train.pickle'), 'rb') as f:
            y_loaded = pickle.load(f)
    elif data_type == 'test':
        with open(os.path.join(model_dir, 'X_test.pickle'), 'rb') as f:
            X_loaded = pickle.load(f)
        with open(os.path.join(model_dir, 'y_test.pickle'), 'rb') as f:
            y_loaded = pickle.load(f)
    else:
        raise ValueError("data_type must be either 'train' or 'test'.")
    
    return X_loaded, y_loaded

# Load the data (you can change 'train' to 'test' to load test data)
X_loaded, y_loaded = load_data(MODEL_DIR, data_type='train')

# Check if the data loaded correctly
print(f"Loaded X shape: {X_loaded.shape}")
print(f"Loaded y shape: {y_loaded.shape}")

# Display some sample images
for i in range(NUM_SAMPLES_TO_DISPLAY):
    plt.imshow(X_loaded[i])  # Display the ith image
    plt.title(f"Label: {y_loaded[i]}")  # Display the corresponding label
    plt.axis('off')  # Turn off the axis for better visualization
    plt.show()
