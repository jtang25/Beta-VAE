import numpy as np
import pandas as pd
import os
from PIL import Image
import skimage.measure as ski
import cv2
import random
from sklearn.model_selection import train_test_split

def read_data_csv(path: str, verbose: bool):
    """
    Reads data from csv file and turns into a pandas Dataframe

    :param path:        path to image folders
    :param verbose:     prints debugging information
    :return             pd.Dataframe
    """

    if verbose:
        print(f"Read images from {path} into numpy array")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    if not path.endswith(".csv"):
        raise FileNotFoundError(f"{path} is not a csv file")
    
    return pd.read_csv(path, index_col=0)

def read_resize_images(path: str, verbose: bool, multiclass: bool = False):
    """
    Reads images from folders and stores each image as a numpy array with its label.

    :param path:        Path to image folders.
    :param verbose:     Prints debugging information.
    :return:            List of dictionaries containing 'image' and 'label'.
    """
    if verbose:
        print(f"Reading images from {path} into a structured format")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    
    

    if multiclass:
        data = []
        for label, folder in enumerate(["Testing", "Training"]):
            folder_path = os.path.join(path, folder)
            if not os.path.exists(folder_path):
                if verbose:
                    print(f"Folder {folder_path} does not exist, skipping...")
                continue
            for label, folder in enumerate(["glioma", "meningioma", "notumor", "pituitary"]):
                folder_path = os.path.join(folder_path, folder)
                if not os.path.exists(folder_path):
                    if verbose:
                        print(f"Folder {folder_path} does not exist, skipping...")
                    continue
            
                
    else:
        data = []
        for label, folder in enumerate(['no', 'yes']):
            folder_path = os.path.join(path, folder)
            if not os.path.exists(folder_path):
                if verbose:
                    print(f"Folder {folder_path} does not exist, skipping...")
                continue
            
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        img = Image.open(file_path).convert('L')
                        centered_img = center_brain(img)
                        img_resize = centered_img.resize((354, 386), Image.Resampling.LANCZOS)
                        img_array = np.array(img_resize)
                        data.append({'image': img_array, 'label': label})
                    except Exception as e:
                        print(f"Error loading image {file_name}: {e}")
    
    return data

def center_brain(img):
    """
    Detects the contours of the brain in the MRI scan and crops the image to fit the edges of the brain

    :param image:       PIL Image of brain scan
    :return:            PIL Image of cropped brain scan
    """

    image_array = np.array(img)
    _, binary_mask = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # Crop the image to the bounding box of the brain
        cropped_array = image_array[y:y+h, x:x+w]
    else:
        # If no contours are found, return the original image
        print("didnt")
        cropped_array = image_array
    
    return Image.fromarray(cropped_array)

def preprocess_data(raw_data: list[dict]):
    processed_data = []

    for entry in raw_data:
        processed_entry = entry.copy()
        # reduce
        processed_entry["image"] = ski.block_reduce(entry["image"], (8, 8), np.max)
        # zero center
        mean_pixel_value = np.mean(processed_entry["image"])
        processed_entry["image"] = processed_entry["image"] - mean_pixel_value
        # normalize
        processed_entry["image"] = (processed_entry["image"] / 255.0).astype(np.float32)
        processed_data.append(processed_entry)
    
    return processed_data

def shuffle_split(data: list[dict], split: float = 0.1, seed: int = None, verbose: bool = True):
    """
    Splits data into train, test, and validation sets.

    :param data:  List of dictionaries with 'image' (array) and 'label'.
    :param split: Proportion of data to use for test set.
    :param seed:  Random seed for reproducibility.
    :param verbose: If True, prints sizes of splits.
    :return: X_train, X_test, X_val, y_train, y_test, y_val
    """
    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    shuffled_data = random.sample(data, len(data))

    X = np.stack([entry["image"] for entry in shuffled_data], axis=0)
    y = np.stack([entry["label"] for entry in shuffled_data], axis=0)

    # split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=split, random_state=42, shuffle=True
    )   

    val_split = split / (1 - split)

    # split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_split, random_state=42, shuffle=True
    )  # 0.25 x 0.7 = 0.175 (17.5%) of the total data for validation

    if verbose:
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, X_test, X_val, y_train, y_test, y_val
