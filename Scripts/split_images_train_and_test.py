# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:29:04 2024

@author: Elijah

This script splits the annotated images for training a StarDist model into training and testing datasets. 
This script assumes you have your images stored in a directory and you want to randomly split them into 
training and testing sets with a specified ratio.

Make sure to replace "images_directory", "masks_directory", "train_directory", and "test_directory" with the 
actual paths to the images/masks directory and the directories where you want to save the train and test datasets, respectively.
"""

import os
import random
import shutil

# Set the directory containing your images
images_directory = "GTA_and_masks/images_and_masks/ground_truth/images"
masks_directory = "GTA_and_masks/images_and_masks/ground_truth/masks"

# Set the directory where you want to save the train and test datasets
train_images_directory = "second_training/train/images"
train_masks_directory = "second_training/train/masks"
test_images_directory = "second_training/test/images"
test_masks_directory = "second_training/test/masks"

# Set the ratio for splitting the dataset (e.g., 88 for 80% train, 12% test)
train_ratio = 0.88

# Get the list of image filenames
image_files = os.listdir(images_directory)
print("Looking through all images and masks...")

# Shuffle the list of filenames randomly
random.shuffle(image_files)
print("Shuffling images...")

# Calculate the number of images for the training set based on the ratio
num_train_images = round(len(image_files) * train_ratio)

# Split the image filenames into training and testing sets
print("Now, randomly splitting images into test and train")

train_images = image_files[:num_train_images]
test_images = image_files[num_train_images:]

# Create the train and test directories if they don't exist
os.makedirs(train_images_directory, exist_ok=True)
os.makedirs(train_masks_directory, exist_ok=True)
os.makedirs(test_images_directory, exist_ok=True)
os.makedirs(test_masks_directory, exist_ok=True)

# Move images and masks to the train directory
for filename in train_images:
    src_image = os.path.join(images_directory, filename)
    dst_image = os.path.join(train_images_directory, filename)
    src_mask = os.path.join(masks_directory, filename)
    dst_mask = os.path.join(train_masks_directory, filename)
    shutil.copy(src_image, dst_image)
    shutil.copy(src_mask, dst_mask)

# Move images and masks to the test directory
for filename in test_images:
    src_image = os.path.join(images_directory, filename)
    dst_image = os.path.join(test_images_directory, filename)
    src_mask = os.path.join(masks_directory, filename)
    dst_mask = os.path.join(test_masks_directory, filename)
    shutil.copy(src_image, dst_image)
    shutil.copy(src_mask, dst_mask)

print("Dataset split completed successfully!")
print("Number of training images:", len(train_images))
print("Number of testing images:", len(test_images))