# Model Training Script
The scripts for training and validation have already been provided by [StarDist](https://nbviewer.org/github/stardist/stardist/blob/master/examples/2D). However, the script used below has been adpated for this particular use case, and therefore has several changes. 

But, before running these scripts, the following steps must have been completed. Refer to the ```README.md``` file for details on how to do these:

1. De-arraying of the TMA cores
2. Create crops of images and tasks for training

## Data preparation
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

from stardist import fill_label_holes, relabel_image_stardist, random_label_cmap
from stardist.matching import matching_dataset

np.random.seed(42)
lbl_cmap = random_label_cmap()
```

```python
# Read in and prepare training data (images and their corresponding masks) for the machine learning model
from pathlib import Path  # Import he Path object from the pathlib module

X = sorted(glob('train/images/*.tif'))
Y = sorted(glob('train/masks/*.tif'))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

# Print the list of files returned by the glob function call for X and Y
print("Files found for X:")
print(X)
print("\nFiles found for Y:")
print(Y)
``` 

```python
# Save the images in a list
X = list(map(imread,X))
Y = list(map(imread,Y))
```

```python
# Check the label, shapes, and dimensions of the images to ensure they are valid for the use in the training

# Check if X has at least one element
if len(X) > 0:
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
# Comfirm the dimensions of images and masks
print("Image shape:", img.shape)
print("Label shape:", lbl.shape)
``` 

```python
# Preview the raw and ground truth label of one image
plt.figure(figsize=(16,10))
plt.subplot(121); plt.imshow(img,cmap='gray');   plt.axis('off'); plt.title('Raw image')
plt.subplot(122); plt.imshow(lbl,cmap=lbl_cmap); plt.axis('off'); plt.title('GT labels')
None;
```

```python
# This evaluates the performance of the StarDist model with different numbers of rays `(n_rays) by reconstructing the labels and calculating the mean Intersection over Union (IoU) scores for each configuration.

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

## Training Script

```python
# Check whether tensor flow and stardist are already installed
import stardist
import tensorflow as tf

print("StarDist version:", stardist.__version__)
print("TensorFlow version:", tf.__version__)

# If not, run
pip install stardist tensorflow

# Sets up an environment for traing using the model and imports all necessary dependencies
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
# Read in training data
X = sorted(glob('train/images/*.tif'))
Y = sorted(glob('train/masks/*.tif'))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))
```

```python
# Assign images and masks to lists X and Y

X = list(map(imread,X))
Y = list(map(imread,Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
``` 


```python
# Split into train and validation datasets.
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
# Training data consists of pairs of input image and label instances.
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
if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.8)
    # alternatively, try this:
    # limit_gpu_memory(None, allow_growth=True)
```

```python
# Save the model configuration in the working directory
model = StarDist2D(conf, name='stardist', basedir='models')
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
# Data augmentation artificially creates more training data by transforming exiting images and masks into different geometrical, yet plausible forms

```python
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

### Actual Training
```python
# Load tensorboard for visualizing and monitoring the training process
%reload_ext tensorboard
%tensorboard --logdir=.
```

```python
model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter, epochs=100, steps_per_epoch=10)
```

### Threshold optimization
```python
# The optimized threshold values are saved to disk and will be automatically loaded with the model
model.optimize_thresholds(X_val, Y_val)
```

### Evaluation and detection performance
```python
#First predict the labels for all validation images:
Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_val)]
```

```python
#Plot a GT/prediction example to see how the model performs
plot_img_label(X_val[0],Y_val[0], lbl_title="label GT")
plot_img_label(X_val[0],Y_val_pred[0], lbl_title="label Pred")
```
```python
# Choose several IoU thresholds τ that might be of interest, and for each, compute matching statistics for the validation data
taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
```

```python
# Print all available matching statistics for τ=0.5
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

### Export the model
```python
# Download the trained model as a zip file to a local directory (change the value of the "exported_model_dir" value to the path of the saved model)
import os

# Specify the directory containing the exported model files
exported_model_dir = "/content/models/stardist"

# List the files in the directory
files = os.listdir(exported_model_dir)
print("Files in exported model directory:")
for file in files:
    print(file)

# Finally, export the directory caontaining the model and all its associated files from colab to a local directory. 
!zip -r models.zip models
files.download('models.zip')

# Unzip the models.zip file to extract the model
```

C:\Users\Elijah\Documents\stardist_demo_proj\Training_data.ipynb