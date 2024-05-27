# Model Training Script
The notebook for training and validation have been provided by [StarDist](https://nbviewer.org/github/stardist/stardist/blob/master/examples/2D). However, the code used below has been adpated for this particular use case, and therefore has some changes.

But, before running these code, the following steps must have been completed. Refer to the ```README.md``` file for details on how to do these:

1. Creating crops of images and masks from TMA cores.
2. Generating training data by annotating ROIs in the crops. 
3. Splitting the images and masks into training and test datasets.

Below are the Notebooks used in this project:

- [Data Preparation](https://colab.research.google.com/drive/1gw9ATdYw8QktLqLJT5pQ9wKxQ9zOvYir?usp=sharing)

- [Model Training](https://colab.research.google.com/drive/1KKI6eyBgBG-590ktFPLUk96BCevFO_il#scrollTo=yom1JrUuDlvM)

- [Model Predictions](https://colab.research.google.com/drive/1D697xfXbUkokDMrhDDkyOFIAeSU24sgC)


## Data Preparation
- Before the actual training is done, the ```Data Preparation``` script is used to confirm the fitting of the ground-truth labels with star-convex polygons, as this shows the optimal "number of rays" to be used in the training process. It also shows if the annotations in the image are ideal for training with the Stardist model.

- Open a Google Colab notebook and run the code blocks sequentially. 

```python
# First, install StarDist
pip install stardist
```

```python
# This block of code sets up an environment for working with image processing tasks using the StarDist model, and imports all the necessary libraries and functions to facilitate the training.

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path

from stardist import fill_label_holes, relabel_image_stardist, random_label_cmap
from stardist.matching import matching_dataset

np.random.seed(42)
lbl_cmap = random_label_cmap()
```

```python
# Read in the training data (images and their corresponding masks)
# Before running this code, make sure the folder containing the iomages and masks are saved as a zip file
# Run the code and select the zip file to upload the data
from google.colab import files # Import he Path object from the google.colab module
import os
import shutil

# Upload the zip file
uploaded = files.upload()

# Extract the uploaded zip file
for filename in uploaded.keys():
    # Create a directory with the same name as the zip file (without the .zip extension)
    folder_name = os.path.splitext(filename)[0]
    os.makedirs(folder_name, exist_ok=True)
    # Extract the contents of the zip file into the newly created directory
    shutil.unpack_archive(filename, folder_name)
    print(f"Uploaded folder '{folder_name}'")
```
```python
# Check the file structure
!find data_training -type d | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"
```

```python
# Check that the data are correctly read in and assign variable to the images and their corresponding masks
from pathlib import Path  # Import he Path object from the pathlib module

X = sorted(glob('data_training/data/train/images/*.tif'))
Y = sorted(glob('data_training/data/train/masks/*.tif'))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

# Print the list of files returned by the glob function call for X and Y
print("Files found in images folder:")
print(X)
print("\nFiles found in masks folder:")
print(Y)
``` 

```python
# Save the images in a list
X = list(map(imread,X))
Y = list(map(imread,Y))
```

```python
# Check the label, shapes, and dimensions of the images to ensure they are valid for use in the training
# This code selects a specific image and its label (at index i) for further processing, assuming that X and Y are lists or arrays containing image data and their corresponding labels

if len(X) > 0:    # Check if X has at least one element
    # Calculate the index i as the minimum value between 4 and one less than the length of X
    i = min(4, len(X) - 1)
    # Access elements from X and Y at index i
    img, lbl = X[i], fill_label_holes(Y[i])
    assert img.ndim in (2, 3)
    img = img if img.ndim == 2 else img[..., :3]
    # assumed axes ordering of img and lbl is: YX(C)
else:
    print("X has no elements. Please check the directory or the file naming.")
```

```python
# Use the example image selected above to comfirm the dimensions of images and masks
print("Image shape:", img.shape)
print("Label shape:", lbl.shape)
``` 

```python
# Preview the raw and ground truth label of the image
plt.figure(figsize=(16,10))  # Set plot size
plt.subplot(121); plt.imshow(img,cmap='gray'); plt.axis('off'); plt.title('Raw image')
plt.subplot(122); plt.imshow(lbl,cmap=lbl_cmap); plt.axis('off'); plt.title('GT labels')
None;
```

```python
# This evaluates the performance of the data with the StarDist model using different numbers of rays `(n_rays)
# It reconstructs the labels and calculates the mean Intersection over Union (IoU) scores for each configuration.

n_rays = [2**i for i in range(2,8)]
scores = []
for r in tqdm(n_rays):
    Y_reconstructed = [relabel_image_stardist(lbl, n_rays=r) for lbl in Y]
    mean_iou = matching_dataset(Y, Y_reconstructed, thresh=0, show_progress=False).mean_true_score
    scores.append(mean_iou)
```

### Fitting ground-truth labels with star-convex polygons
```python
# Create a plot showing the relationship between the number of rays used in the StarDist model and the reconstruction score (the mean IoU between the original labels and the reconstructed labels)
plt.figure(figsize=(8,5))
plt.plot(n_rays, scores, 'o-')
plt.xlabel('Number of rays for star-convex polygon')
plt.ylabel('Reconstruction score (mean intersection over union)')
plt.title("Accuracy of ground truth reconstruction (should be > 0.8 for a reasonable number of rays)")
None;
```

```python
# Reconstruct the example image with various number of rays
fig, ax = plt.subplots(2,3, figsize=(16,11))
for a,r in zip(ax.flat,n_rays):
    a.imshow(relabel_image_stardist(lbl, n_rays=r), cmap=lbl_cmap)
    a.set_title('Reconstructed (%d rays)' % r)
    a.axis('off')
plt.tight_layout();
```
The plot shows that the annotations are ideal for training a neural network using StarDist and the training can be performed using 32, 64, and 128 rays.



## Model Training 
- Open a Google Colab notebook and run each code block one after the other. See the model training script used for this project [here](https://colab.research.google.com/drive/1KKI6eyBgBG-590ktFPLUk96BCevFO_il?usp=sharing).

```python
pip install stardist
```

```python
import stardist
import tensorflow as tf  # TensorFlow is already installed in Google Colab

# Check installed versions
print("StarDist version:", stardist.__version__) 
print("TensorFlow version:", tf.__version__)
```

```python
# Read in the training data (images and their corresponding masks)
# Before running this code, make sure the folder containing the images and masks are saved as a zip file
# Run the code and select the zip file to upload the data
from google.colab import files # Import he Path object from the google.colab module
import os
import shutil

# Upload the zip file
uploaded = files.upload()

# Extract the uploaded zip file
for filename in uploaded.keys():
    # Create a directory with the same name as the zip file (without the .zip extension)
    folder_name = os.path.splitext(filename)[0]
    os.makedirs(folder_name, exist_ok=True)
    # Extract the contents of the zip file into the newly created directory
    shutil.unpack_archive(filename, folder_name)
    print(f"Uploaded folder '{folder_name}'")
```

```python
# Check the file structure
!find data_training -type d | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"
```

```python
# Set up an environment for training using StarDist and import all necessary dependencies
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

np.random.seed(42)
lbl_cmap = random_label_cmap()
```

```python
# Check that the data are correctly read in and assign variable to the images and their corresponding masks
from pathlib import Path  # Import the Path object from the pathlib module

X = sorted(glob('data_training/data/train/images/*.tif'))
Y = sorted(glob('data_training/data/train/masks/*.tif'))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))  # Checks whether the base names of corresponding files in X and Y are equal for all pairs to avoid errors in subsequent processing

# Print the list of files returned by the glob function call for X and Y
print("Files found in images folder:")
print(X)
print("\nFiles found in masks folder:")
print(Y)
```

```python
# Convert files specified in X and Y into lists, and then determine the number of channels in the images
# Since the raw images are DAB-stained, 3 channel images are used for the training
X = list(map(imread,X))
Y = list(map(imread,Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
```

```python
# Normalize images and fill small label holes.

axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]
``` 

```python
# Split into train and validation datasets

assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))
```

```python
# Training data consists of pairs of input image and label instances

# Visualize an image and its corresponding label side by side in a matplotlib plot
def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()
```

```python
i = min(9, len(X)-1)
img, lbl = X[i], Y[i]
assert img.ndim in (2,3)
img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
plot_img_label(img,lbl)
None;
```

```python
# A StarDist2D model is specified via a Config2D object
# Print a summary of the Config2D class to be used for the training, its parameters, attributes, methods, and their descriptions
print(Config2D.__doc__)
```

```python
# 32 is a good default choice (see Data_Preparation.ipynb)
n_rays = 32

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = False and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    use_gpu      = use_gpu,
    n_channel_in = n_channel,
)
print(conf)
vars(conf)
```

```python
if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.8)
    # alternatively, try this:
    # limit_gpu_memory(None, allow_growth=True)
```

```python
# Create an instance of a 2D StarDist model using a configuration specified by 'conf'

model = StarDist2D(conf, name='stardist_model', basedir='models')
```

```python
# Check if the neural network has large enough boundaries to see up to the boundaries of most objects

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")
```

### Data Augmentation

```python
# Data augmentation artificially creates more training data by transforming existing images and masks into different geometrical, yet plausible forms

# Here, the augmenter function applies random rotations, flips, and intensity changes to the images, which are typically sensible for (2D) microscopy images 
def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y
```

```python
# plot some augmented examples
img, lbl = X[0],Y[0]
plot_img_label(img, lbl)
for _ in range(3):
    img_aug, lbl_aug = augmenter(img,lbl)
    plot_img_label(img_aug, lbl_aug, img_title="image augmented", lbl_title="label augmented")
```

### Load Tensorboard and Imitialize Training
```python
# Load tensorboard for visualizing and monitoring the training process
%reload_ext tensorboard
%tensorboard --logdir=.
```

```python
# Initialize the training
model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter, epochs=100, steps_per_epoch= 15)
```
- The training process takes at least 1 hour 20 minutes (with the set parameters above) using 45 images (automatically split into 38 training and 7 validation images).
![](https://github.com/Elijah-Ugoh/Model-Training-For-Nuclei-Segmentation/blob/master/images/training_pogress.png)


- The visualization in tensorflow can be updated as the training progresses by clicking the refresh button any time during the process. 

![visualization of Model Training Progress in Tensorboard](https://github.com/Elijah-Ugoh/Model-Training-For-Nuclei-Segmentation/blob/master/images/visualization_in_tensorboard.png)

### Threshold Optimization
```python
# The optimized threshold values are saved to disk and will be automatically loaded with the model
model.optimize_thresholds(X_val, Y_val)
```

### Evaluation and Detection Performance of the Trained Model
```python
# First predict the labels for all validation images
Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_val)]
```

```python
# See definitions of the abbreviations used in the evaluation above
help(matching)
```

```python
# Plot a Ground truth/prediction example to see how the model performs
# Plot one or two more (change the indeces of X_val, Y_val, and Y_val_pred) to see how the precision of the predictions on other images 

plot_img_label(X_val[1],Y_val[1], lbl_title="label GT")
plot_img_label(X_val[1],Y_val_pred[1], lbl_title="label Pred")
```

```python
# Choose several IoU thresholds τ that might be of interest, and for each, compute matching statistics for the validation data

taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
```

```python
# Example: Print all available matching statistics for τ=0.5

stats[taus.index(0.5)]
```

```python
# Plot the matching statistics and the number of true/false positives/negatives as a function of the IoU threshold τ
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
    ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
ax1.set_xlabel(r'IoU threshold $\tau$')
ax1.set_ylabel('Metric value')
ax1.grid()
ax1.legend()

for m in ('fp', 'tp', 'fn'):
    ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
ax2.set_xlabel(r'IoU threshold $\tau$')
ax2.set_ylabel('Number #')
ax2.grid()
ax2.legend();
```

![Plot of Model Matching Statistics Against IoU Threshold](https://github.com/Elijah-Ugoh/Model-Training-For-Nuclei-Segmentation/blob/master/images/IoU_Threshold.png)

### Export the Model
```python
# Export the model
model.export_TF()
```

```python
# Download the saved model to a local directory (change the value of the "model_dir" variable to the path of the saved model)
import os

# Specify the directory containing the exported model files
model_dir = "/content/models/stardist_model"

# List the files in the directory
model_dir_files = os.listdir(model_dir)
print("Files in exported model directory:")
for file in model_dir_files:
    print(file)
```

```python
# Finally, export the directory containing the model and all its associated files from the Colab notebook to a local directory

from google.colab import files
!zip -r models.zip models

# Download the ZIP file
files.download('models.zip')

# Unzip the models.zip file to extract the model
```