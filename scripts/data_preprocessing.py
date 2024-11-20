import os
import cv2
import numpy as np
import pickle
import random
import sys
from sklearn.model_selection import train_test_split

# Đảm bảo sys.stdout sử dụng mã hóa UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Constants
DATA_DIR = 'dataset/'
CATEGORIES = ['clean', 'defaced/defaced_type_1', 'defaced/defaced_type_2', 'defaced/defaced_type_3', 'defaced/defaced_type_4']
IMG_SIZE = 299  # Tăng kích thước ảnh từ 244 lên 299 để có thêm chi tiết
MODEL_DIR = 'models'

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Data augmentation function
def augment_image(image):
    # # Apply random augmentation
    # flip = random.choice([True, False])
    # if flip:
    #     image = cv2.flip(image, 1)  # Horizontal flip

    # # Random rotation
    # angle = random.randint(-20, 20)
    # (h, w) = image.shape[:2]
    # center = (w // 2, h // 2)
    # matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # image = cv2.warpAffine(image, matrix, (w, h))

    # # Random brightness adjustment
    # value = random.randint(-50, 50)
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    # image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image

def create_dataset(data_dir, categories, img_size):
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        if not os.path.exists(path):
            print(f"Warning: Directory '{category}' does not exist. Skipping this category.")
            continue

        class_label = categories.index(category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            try:
                img_array = cv2.imread(img_path)

                # Check if image is loaded correctly
                if img_array is None:
                    print(f"Warning: Failed to read image '{img_path}'. Skipping.")
                    continue

                # Resize the image to the defined size
                resized_img = cv2.resize(img_array, (img_size, img_size))

                # Apply augmentation to create multiple versions of the same image
                augmented_img = augment_image(resized_img)

                # Add both original and augmented images to the dataset
                data.append([resized_img, class_label])
                data.append([augmented_img, class_label])

            except Exception as e:
                print(f"Error processing image '{img_name}': {e}")
                continue

    return data

def split_and_save_data(dataset, model_dir):
    # Shuffle the dataset
    random.shuffle(dataset)

    # Split features and labels
    X, y = zip(*dataset)  # Unzip the list of tuples into two separate lists

    # Convert to numpy arrays and normalize the features
    X = np.array(X, dtype='float32').reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    y = np.array(y, dtype='int')

    # Split the dataset into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the datasets using pickle
    try:
        with open(os.path.join(model_dir, 'X_train.pickle'), 'wb') as f:
            pickle.dump(X_train, f)
        with open(os.path.join(model_dir, 'y_train.pickle'), 'wb') as f:
            pickle.dump(y_train, f)
        with open(os.path.join(model_dir, 'X_test.pickle'), 'wb') as f:
            pickle.dump(X_test, f)
        with open(os.path.join(model_dir, 'y_test.pickle'), 'wb') as f:
            pickle.dump(y_test, f)

        print("Data has been successfully saved to the model directory.")

    except Exception as e:
        print(f"Error saving processed data: {e}")

# Main function to orchestrate data creation, processing, and saving
if __name__ == "__main__":
    # Create dataset
    dataset = create_dataset(DATA_DIR, CATEGORIES, IMG_SIZE)

    # Check if dataset is empty
    if not dataset:
        print("Error: No data available for training. Exiting.")
        sys.exit(1)

    print(f"Total number of samples: {len(dataset)}")

    # Split the data and save them
    split_and_save_data(dataset, MODEL_DIR)
