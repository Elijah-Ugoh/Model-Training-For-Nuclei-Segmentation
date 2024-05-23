# A 2D Nuclei Segmentation Model for the Detection and Visualization of PACCs in Breast Cancer TMA Images
This repository contains the code and step-by-step process for training a deep-learning model for 2D nuclei segmentation of poly-aneuploid cancer cells (PACCs) in breast cancer tissue microarray (TMA) images using the [StarDist deep-learning](https://github.com/stardist/stardist) package in Python.

## Overview
StarDist is optimized for handling round-shaped objects like cells and nuclei and has no limitation regarding normalization and bit depth of the data. The training data consists of the raw tissue image crops obtained from breast cancer patients and the corresponding ground truth annotatations or labels (as masks) for each image. The model is trained to detect and segment nuclei of all sizes, including PACCs, present in breast cancer tissue images.

Annotation was focused mainly on the nuclei and stromata in each tissue image, and PACCs are identifiable by their abnormally-large nuclei. The figure below shows a summarized workflow from a raw image crop, the ground truth annotation, and model prediction.

![](https://github.com/Elijah-Ugoh/Model-Training-For-Nuclei-Segmentation/blob/master/images/raw_image_ground_truth_prediction.png)

## Data Exploration and Installation of Required Tools
The data used in this project is brightfield tissue micro-array (TMA) images stained using EpCAM-DAB and obtained from 1190 pre-treatment breast cancer patients in the [SweBCG-RT91 cohort](https://www.sciencedirect.com/science/article/pii/S0959804916323620?via%3Dihub). 

### Installing QuPath
[QuPath](https://qupath.readthedocs.io/en/stable/docs/intro/installation.html#download-install) was used to label (ground truth annotation) the images. Once downloaded, it can be installed as below:

```bash
wget https://github.com/qupath/qupath/releases/download/v0.5.1/QuPath-v0.5.1-Linux.tar.xz
tar -xvf QuPath-v0.5.1-Linux.tar.xz # Extract the application
rm QuPath-v0.5.1-Linux.tar.xz # Remove the zip file
chmod u+x /path/to/QuPath/bin/QuPath # Make the launcher executable
```
QuPath requires some dependences, such as JavaFX, so there can be some issues running the application via Linux Terminal. 

Alternatively, install the software on Windows by directly downloading and running the [QuPath-v0.5.1-Windows.msi](https://github.com/qupath/qupath/releases/download/v0.5.1/QuPath-v0.5.1-Windows.msi) file.`

### Installing Fiji
[Fiji](https://imagej.net/software/fiji/downloads) is an alternative to QuPath, but, in this project, it was only used for running the model and comparing the output with QuPath. 

```bash
wget https://downloads.imagej.net/fiji/Life-Line/fiji-linux64-20170530.zip # ImageJ2
```
NB: Fiji is distributed as a portable application. So, no need to run an installer; just download, unpack and start the software.

### Installing StarDist and TensorFlow
The StarDist Python pipeline is used for training the model. It can be [downloaded](https://github.com/stardist/stardist#Installation) and installed as follows: 

```bash
# TensorFlow is a dependency used by StarDist
# Both installations can be done later in Google Colab

pip install tensorflow # First install TensorFlow (version 2.15.0)
pip install stardist # Then install stardist-(version 0.9.1)
```

## Labeling Images and Training the Model Using Stardist

### 1. Spilt the TMA cores into into individual images 
Each microarray contained between 90-130 different tissues cores embedded at defined array coordinates. Data from all patients are randomly distributed in the TMA, with one or two cores per patient. The cores are split to obtain training images. 

This is done using the ```new_dearrayer.groovy``` script. This script is found in the [Scripts](https://github.com/Elijah-Ugoh/Model-Training-For-Nuclei-Segmentation/tree/master/Scripts) directory. Below is the workflow for this:

- Create a new project directory, ```split_tma-cores```. 
- Launch QuPath from Windows Home button or the Linux terminal.
- Click "Create project" and select the folder created above. 
- Click "Add images" and load the TMA cores to be de-arrayed into QuPath. Select "Brightfield (H-DAB)" under "Set image type", then proceed to add images. 
- Go to Automate tab on QuPath and select "script editor"
- Load the ```new_dearrayer.groovy``` into the script editor. 
- Within the script, modify ```qupathProjectDir``` to the path where the TMA cores will be saved and ```tmaImagesDir``` to the TMA location. Also specify the name of the new folder that will hold the splitted TMA cores in ```line 18``` of the script. 
- Click "Run" and wait for the de-arraying to complete.
- A new folder with the specified name will be created inside the QuPath project directory created in step 1 above. 
- This process was repeated for 10 more TMAs containing PACCs. 


### 2. Randomly select 20 cores from the de-arrayed TMA images
This is done using the ```Ramdomize.groovy``` script. 
Below is the workflow for this:

- Create a new project directory, ```randomly_selected_TMA_cores```. 
- Launch QuPath from Windows Home button or the Linux terminal.
- Click "Create project" and select the folder created above. 
- Go to Automate tab on QuPath and select "script editor"
- Load the ```Randomize.groovy``` script into the script editor. 
- Within the script, modify ```sourceDir``` to the location of the TMA cores and ```destinationDir``` to the destination where the selcted cores will be saved in. The number of images to be randomly selected is specified in ```line 9``` of the script. 
- Click "Run" and wait for the sctript to run.
- The script shuffles all the cores, selcts the specified number, and saves them in the specified directory. 


### 3. Create crops for annotation from the selected TMA cores 
To generate the training data, crops that capture different morphologies and regions/time-points in each core must be created. Crops are sub-regions of the cores, measuring 300 x 300 pixels squared in dimension. For each core, 3 crops are created. StarDist recommends a minimum of 128 X 128 pixels squared. The workflow is explained below:

- Create a new project directory, ```annotations_for_training```. 
- Launch QuPath from Windows Home button or the Linux command line.
- Click "Create project" and select the directory created above. 
- Click "Add images'' and load the randomly selected TMA cores (20) from step (2) above into QuPath. Select "Brightfield (H-DAB)" under "Set image type", then proceed to add images.
- Double-click on the first image to open it in the viewer.
- With the image loaded in the viewer, go to ```Classify>Training images>Create region annotations```
- Set Width and Height to 300, each. Set the Size units to 'Pixels', Classification to 'Region*', and Location to 'Random'. 
- Click on 'Create region' as many times as necesary to create multiple sqaure crops of the image. 
- Save the best 3 crops: Double-click on each crop to select it, go to ```File>Export images>Original pixels```.
- Select 'TIFF(ImageJ)' as the Export Format. Leave Downsample factor as 1.0.
- Save the image in a new directory inside the ```annotations_for_training``` project directory. Also, save the other two crops to the same directory. It's best to retain the same folder name as the cores. 
- This process is repeated for the remaining 19 cores. 

NB: It is not possible to automate this process using a script, as a visual inspection of the selected cores is necesary before they are saved. This way, the best cores are selected for annotation.

### 4. Labelling the Images (Ground Truth Annotataions)
Here, two things are done to create the image labels:
- Create region of interests (ROI) object per crop/image
- Covert the ROI objects to label masks

The workflow is explained below:
- Create a new project directory, ```GTA_and_masks```. 
- Launch QuPath from Windows Home button or the Linux terminal.
- Click "Create project" and select the folder created above.
- Anotate the ROI (all cell nuclei in each crop). Press Ctrl+S to save the annotations. 
- Go to Automate tab on QuPath and select "script editor"
- All annotations are exported after. This is done by loading the ```save_and_export_annotations.groovy``` script into the script editor. 
- [This QuPath script](https://forum.image.sc/t/export-qupath-annotations-for-stardist-training/37391/3), which has been provided by Olivier Burri and Romain Guiet, will export the original images and the annotations as masks and save them in the specified directories. 
- Change lines 80 and 81 to the preferred directories for saving the ground truth images and masks. 
- Instead of running the script every time an annotation image is created, use "Run for project" from the script editor menu to export all the annotations for all images in the QuPath project ```GTA_and_masks```.
- The image below shows how to create image labels in QuPath:

![](https://github.com/Elijah-Ugoh/Model-Training-For-Nuclei-Segmentation/blob/master/images/Animation.gif)
Annotating ROIs in each crop using QuPath

NB: StarDist recommends at least 10 crops for training. In this case, 51 cores (6 test images, 7 validation images, and 38 training images) are used to provide sufficient training data.

### 5. Split Data into Test and Train Datasets
The python script ```split_images_train_and_test.py``` is used to randomly split the data into test and train datasets. This way, we ensure that the images for tests are as representative of the training images while also automating the process.

It is conventional to use 10-20% of the data for testing. Here, 12% is used considering the data isn't large. 

```bash
# From the terminal, run the script as:
python split_images_train_and_test.py

# NB: Update the directories for the images (in the script), as well as where the test and train data will be saved after splitting.  
```

### 6. Training the model
The [Model_Training_Script](https://github.com/Elijah-Ugoh/Model-Training-For-Nuclei-Segmentation/blob/master/Scripts/Model_Training_Script.md) is run in Google Colab. But, it can also be run in other IDEs or from the terminal. 

Colab has a simple interface and has TensorFlow pre-installed. The script contains the instructions for importing the data and training a model using the StarDist deep-learning package in Python.

The code blocks are run sequentially, starting from installing StarDist and TensorFlow, as well as importing all necessary python packages.

## Nuclei Detection Using the Trained Model
Unzip the models.zip file to extract the model files. 

```bash 
unzip models.zip
```

NB: Usually, the trained model will not work in the associated ImageJ/Fiji plugins (e.g. CSBDeep and StarDist) and QuPath Extension, especially if TensorFlow 2.0 and later was used in the training. 

For use in Fiji, the trained model must be exported using TensorFlow 1.x. For QuPath, the model must be converted to ONNX (Open Neural Network Exchange) format. The conversion allows for interoperability between different deep learning frameworks, in this case, enabling the model trained in TensorFlow to be run in other frameworks (that support ONNX) with minimal effort.

The guide below outlines how to achieve this. 

1.For Fiji, if TensorFlow 2.x is used, StarDist provides a simple [process to convert the model](https://gist.github.com/uschmidt83/4b747862fe307044c722d6d1009f6183) to a useable format in Fiji. 

```python
# Create a new Python environment and install TensorFlow 1.x, CSBDeep, and StarDist

conda create -y --name tf1_model_export python=3.7
conda activate tf1_model_export

# Note: GPU support is not necessary for tensorflow
pip install "tensorflow<2" # This isntall tensorflow v1.15.5
pip install "csbdeep[tf1]"

# Also install stardist in this environment
pip install "stardist[tf1]"
```

Export the model using the ```export_model.py``` python script. 

```bash
python export_model.py

# The script must be saved in the same directory where it is run from
```

2.For Qupath, a similar conversion process must be used. 
Within the same conda  enevironment, ```tf1_model_export```, install tf2onnx.

```bash
# First, isntall tf2onnx, if not already installed from the terminal 
pip install -U tf2onnx   # v1.16.1
```
Documentation for installing and using tf2onnx is provided [here](https://github.com/onnx/tensorflow-onnx). 

But, unlike for Fiji, the conversion is run on the raw model file.

```bash 
# Unzip the TF_SavedModel.zip obtained from step (1) above to extract the saved_model.pb model file
unzip saved_model.pb
```

Proceed to install tf2onnx, then run the following command to convert the model file to ONNX format. The correct path to the model (with a '.pb' suffix) must be provided. 

```bash 
# Once tf2onnx is installed, run this command to convert the TF model to ONNX format

python -m tf2onnx.convert --opset 11 --saved-model /mnt/c/Users/Elijah/Documents/BINP37_Research_Project/first_training/models/models/stardist/TF_SavedModel --output_frozen_graph /mnt/c/Users/Elijah/Downloads/dab_stained_nuclei.pb

# Stardist recommends opset value of 10, but that doesn't work due to update. 11 worked here. 
```

### Using the Trained Model in QuPath and Fiji
After successfully converting the model, the process below can be followed to use the model for cell nuclei segmentation in QuPath or Fiji. 

1. With the instructions on [this page](https://qupath.readthedocs.io/en/0.4/docs/deep/stardist.html), the trained and converted StarDist 2D model is incorporated directly within QuPath. 

   - Download the [QuPath extension for StarDist](https://github.com/qupath/qupath-extension-stardist). The [version used](https://github.com/Elijah-Ugoh/Model-Training-For-Nuclei-Segmentation/blob/master/qupath-extension-stardist-0.5.0.jar) for this project is included in this repository.  

   - To use the StarDist extension for QuPath, open QuPath, go to ```Extensions>StarDist```, and select "StarDist H&E nucleus detection script". The will open up a new QuPath script editor with a customizable script for use with a custom trained model. The path to the custom trained model can be iputed in the "def modelPath" variable.

   - For the purpose of this project, this script has been modified to ```detect_dab_stained_nuclei_.groovy``` for nuclei detection in EpCAM-DAB-stained images. This script is used for our trained purpose.

   - The model can now be used to detct and segment nuclei in any TMA image not seen by the model during training.  

   - The H&E model used for comparison can be downloaded from other [pretrained StarDist 2D models](https://github.com/qupath/models/tree/main/stardist).

   - Run the H&E model on the same set of images for comparison.

   - Adjust the normalization percentiles, pixel size, and threshold as necessary for optimal nuclei detection.

   ![](https://github.com/Elijah-Ugoh/Model-Training-For-Nuclei-Segmentation/blob/master/images/detection_example1.gif)

   ![](https://github.com/Elijah-Ugoh/Model-Training-For-Nuclei-Segmentation/blob/master/images/detection_example2.gif)

QuPath was primarily used in this project to run the trained model. However, Fiji can also be used. 

2. To use the trained model in Fiji, the instructions provided [here](https://imagej.net/plugins/stardist) must be followed to import the custom model into Fiji's StarDist 2D plugin. 

   - But first, the StarDist 2D plugin for Fiji must be installed. This [guide](https://github.com/stardist/stardist-imagej) details how to install the plugin in Fiji. 

   - Once the plugin is successfully installed, the custom trained models can be applied for nuclei detection in new images. 

   - To use the imported model, load an image into Fiji, go to ```Plugins>StarDist>StarDist2D```, make sure the model is selected under ```Advanced Options```, then click OK and wait for the prediction to run.

   - Adjust the normalization percentiles, pixel size, and threshold as necessary for optimal nuclei detection.

## Acknowledgements
This study is based on data from the [SweBCG-RT91 cohort](https://www.sciencedirect.com/science/article/pii/S0959804916323620?via%3Dihub). Special thanks to the contributors of this dataset and also to the [Tissue Development and Evolution (TiDE)](https://tide.blogg.lu.se/welcome-to-tide/) group at the Medical Faculty of Lund University, where this project was performed, for providing access to the dataset.

### Downloads
The converted model files can be downloaded from this repository:

- For use in QuPath, download the [dab_stained_nuclei2024.pb](https://github.com/Elijah-Ugoh/Model-Training-For-Nuclei-Segmentation/tree/master/Model/dab_stained_nuclei2024.pb) file - RGB (3-channel) images

- For use in Fiji, download the [TF_SavedModel](https://github.com/Elijah-Ugoh/Model-Training-For-Nuclei-Segmentation/tree/master/Model/TF_SavedModel) zip file - RGB (3-channel) images


### Credit & Reuse
This model can be used for free, but if you intend to train your own model using the included script, remember to [cite the original develelopers of StarDist](https://github.com/stardist/stardist).

### Limitations
This model was trained on brightfield tumor images stained with EpCAM-DAB only. While it is applicable for detecting and segmenting nuclei in breast cancer images, we cannot guarantee that it will produce the best results for other types of tissue images.
