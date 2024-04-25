# BINP37 Research Project Documentation
# Data Exploration and Installation of Required Tools

### Installing QuPath
QuPath is used for image analysis and Ground Truth annotation of stained slides

[QuPath Documentation](https://qupath.readthedocs.io/en/stable/docs/intro/installation.html#download-install)
```bash
# Install QuPath-v0.5.1-Linux.tar.xz 
# Process

wget https://github.com/qupath/qupath/releases/download/v0.5.1/QuPath-v0.5.1-Linux.tar.xz
tar -xvf QuPath-v0.5.1-Linux.tar.xz # Extract the application
rm QuPath-v0.5.1-Linux.tar.xz # Remove the zip file
chmod u+x /path/to/QuPath/bin/QuPath #Make the launcher executable
```

Alternatively, install the software on Windows directly. QuPath is an open-source unsigned software. It requires a some dependences, such as JavaFX, so ther can be some issues running the application via Linux Terminal

```bash
# Download and run the standard Windows (local) installer for QuPath
QuPath-v0.5.1-Windows.msi
```
### Installing StarDist
Follow the guide on the documenation
Installing and using [StarDist](https://github.com/stardist/stardist)

```bash
pip install tensorflow # First install TensorFlow 2 - 2.16.1
pip install stardist # then install stardist - 0.4.6 
```

### Installing Fiji
Follow the guide on the documenation
Installing and using [Fiji](https://imagej.net/software/fiji/downloads)

```bash
#On Linux
wget https://downloads.imagej.net/fiji/Life-Line/fiji-linux64-20170530.zip # ImageJ2
```
NB: Fiji is distributed as a portable application. So, no need to run an installer; just download, unpack and start the software.

## Training the Model using Stardist
### Nuclei segmentation using Stardist
- Biological analysis of 2D and 3D images - identify cell shapes, nuclei shape, segment individual cells from staining amd microscopy.

- Dense segmenatation and instamce segmentation has been very successful by training deep (convolutional) neural networks with annotated training data (ground truth images)

- Stardist is well adapted to nuclei segmentation becasue cell nuclei typically have round shapes.

- Stardist is ideal for round-shaped images and has no limitation regarding normalization and bit depth of the data. 
- When it comes to cell sizes, the training data must include all the sizes you expect it to be able to predict. Otherwise, it wil perform badly when you give it data that has varying sizes not found on the training data. 


### 1. Spilt the TMA cores into into individual images 
This is done using the ```new_dearrayer.groovy``` script. This script is found in the script directory. 
Below is the workflow for this:

- Create a new project directory, ```split_tma-cores```. 
- Launch QuPath from Windows Home button or the Linux command line.
- Click "Create project" and select the folder created above. 
- Click "Add images'' and load the TMA cores to be de-arrayed into QuPath. Select "Brightfield (H-DAB)" under "Set image type", then proceed to add images. 
- Go to Automate tab on QuPath and select "script editor"
- Load the ```new_dearrayer.groovy``` into the script editor. 
- Within the script, modify ```qupathProjectDir``` to the path where the TMA cores will be saved and ```tmaImagesDir``` to the TMA location. Also specify the name of the new folder that will hold the splitted TMA cores in ```line 18``` of the script. 
- Click "Run" and wait for the de-arraying to complete.
- A new folder with the specified name will be created inside the QuPath folder directory created in step 1 above. 
- Repeat this process for 10 more TMAs 


### 2. Randomly select 20 TMA cores from all  
This is done using the ```Ramdomize.groovy``` script. 
Below is the workflow for this:

- Create a new project directory, ```randomly_selected_TMA_cores```. 
- Launch QuPath from Windows Home button or the Linux command line.
- Click "Create project" and select the folder created above. 
- Go to Automate tab on QuPath and select "script editor"
- Load the ```Randomize.groovy``` script into the script editor. 
- Within the script, modify ```sourceDir``` to the location of the TMA cores and ```destinationDir``` to the destination you want to save them in. Also specify the number of images to be randomly selected from the shuffling in ```line 9``` of the script. 
- Click "Run" and wait for the sctript to run.
- The randomly selected cores will be saved in the specified destination directory. 

NB: StarDist recommends at least 10, but 20 cores are used to provide sufficiengt training data. 


### 3. Create crops for annotation from the TMA cores 
To generate the training data, crops that capture different morphologies and regions/time-points in each core must be created. The crops are essentially sub-regions of the cores, measuring 300 x 300 Pixels squared in dimension. For each core, 3 crops are created. Since the training is done using StarDist pluggin, the least recommended crop dimension is 128 X 128 Pixels squared. The workflow is explained below:

- Create a new project directory, ```annotations_for_training```. 
- Launch QuPath from Windows Home button or the Linux command line.
- Click "Create project" and select the folder created above. 
- Click "Add images'' and load the randomly selected TMA cores (20) from step 2 above into QuPath. Select "Brightfield (H-DAB)" under "Set image type", then proceed to add images.
- Double-click on the first image to select it.
- With th image loaded in the viewer, go to ```Classify>Training images>Create region annotations```
- Set Width and Height to 300 pixels, each, and set Location to Random. 
- Click on 'Create region' as many times as necesary to create sqaure crops of the image. 
- Save the best 3 crops: double-click on each crop to select it, go to ```File>Export images>Original pixels```.
- Select 'TIFF(ImageJ)' as the Export Format. Leave Downsample factor as 1.0.
- Save the image in a new folder inside the ```annotations_for_training``` project directory. Also, save the other two crops to the same folder. It's best to retain the same folder name as the cores. 
- Repeat this process for the remaining 19 cores. 

### 4. Ground Truth Annotataions
Here, two things need to be done to create the annotation images:
- Create ROI object per crop/image
- Covert the ROI objects to label masks

The workflow is explained below:
- Create a new project directory, ```GTA_and_masks```. 
- Launch QuPath from Windows Home button or the Linux command line.
- Click "Create project" and select the folder created above.
- Anotate the ROI (all cell nuclei in each crop) and press Ctrl+ S to save the annotations. 
- Go to Automate tab on QuPath and select "script editor"
- Load the ```save_and_export_annotations.groovy``` script into the script editor. 
- The giff below show a simple way to do this. 
- [This script](https://forum.image.sc/t/export-qupath-annotations-for-stardist-training/37391/3) has been provided by Olivier Burri and Romain Guiet. 
- Remember to change lines 82 and 83 with the preferred directories for saving the ground truth images and masks. 




### CLI upload if using V7 for annotation
darwin dataset push pacc-image-analysis/training_dat a -p [folder_with_images_and_folders]


### Nuclei Detection Using the Trained Model

NB: Usually, the trained model will not work in the associated ImageJ/Fiji plugins (e.g. CSBDeep and StarDist) and QuPath Extension, especially if TensorFlow 2.0 and later was used in the training. 

For use in Fiji, the trained model must be exported using TensorFlow 1.x. For QuPath, the model must be converted to ONNX (Open Neural Network Exchange) format. The conversion allows for interoperability between different deep learning frameworks, in this case, enabling the model trained in TensorFlow to be run in other frameworks (that support ONNX) with minimal effort.

The guide below outlines how to acheive this. 

1.For Fiji, if TensorFlow 2.x is used, StarDist provides a simple [process to convert the model](https://gist.github.com/uschmidt83/4b747862fe307044c722d6d1009f6183) to a useable format in Fiji. 

```python
# Create a new Python environment and install TensorFlow 1.x, CSBDeep and other necessary packages (e.g. StarDist)

conda create -y --name tf1_model_export python=3.7
conda activate tf1_model_export
# note: gpu support is not necessary for tensorflow
pip install "tensorflow<2"
pip install "csbdeep[tf1]"
# also install stardist in this example
pip install "stardist[tf1]"

#export the model using this python script:
from stardist.models import StarDist2D
model = StarDist2D(None, name='my_model', basedir='.')
model.export_TF()

# Replace value of the basedir parameter with the directory path where you want to save the exported TensorFlow model
# Also replace value in the name variable with the name of your trained model. See export_model.py for example. 
 ```

```bash
# Save it as export_model.py and run from the terminal as:
python export_model.py

# Make sure the script is saved in the same directory where it is run from
```

2.For Qupath, a similar conversion process must be used. 
Within the same conda ```tf1_model_export``` enevironment, install tf2onnx.

```bash
# First, isntall tf2onnx, if not already installed from the terminal 
pip install -U tf2onnx
```
Documentation for installing and using tf2onnx is provoded [here](https://github.com/onnx/tensorflow-onnx). 

Once the insatllation completes, run the following command to convert the model. Note that the correct path to the model (with a '.pb' suffix) must be provided. 

```bash 
# Once tf2onnx is installed, run this command to convert the TF model to ONNX format
python -m tf2onnx.convert --opset 11 --saved-model "/path/to/saved/model/" --output_frozen_graph "/path/to/output/model/file.pb"

# Example usage
python -m tf2onnx.convert --opset 11 --saved-model /mnt/c/Users/Elijah/Documents/BINP37_Research_Project/first_training/models/models/stardist/TF_SavedModel --output_frozen_graph /mnt/c/Users/Elijah/Downloads/dab_stained_nuclei.pb

# Stardist recommends opset value of 10, but that doesn't work due to update. 11 worked here. 
```
After successfully converting the model, the process below can be followed to use the model for cell nuclei segmentation in QuPath or Fiji. 

1. With the instructions on [this page](https://qupath.readthedocs.io/en/0.4/docs/deep/stardist.html), the trained StarDist 2D model is incorporated directly within QuPath. 

   - When the [QuPath extension for StarDist](https://github.com/qupath/qupath-extension-stardist) is downloaded, the path to the custom trained model must be changed within the script. 
   - To use the StarDist extension for QuPath, Open QuPath and go to ```Extensions>StarDist``` and select the script to use. Each script requires a trained model. 

   - For nuclei detection, the H&E nuclei dtection script provided by StarDist has been modified. 

   - Here are the [pretrained StarDist 2D models](https://github.com/qupath/models/tree/main/stardist) that can be downloaded and used for test-runs directly in QuPath. 

   - Remember to update the path for the custom trained or pretrained StarDist model in the script. 

2. To use the trained model in Fiji, the instructions provided [here](https://imagej.net/plugins/stardist) must be followed to import the custom model into Fiji's StarDist 2D plugin. 

   - But, first, the StarDist 2D plugin for Fiji must be installed. This [guide](https://github.com/stardist/stardist-imagej) details how to install the plugin in Fiji. 

   - Once the plugin is successfully installed, the custom trained models can be applied for nuclei detection in nnew images. 




 