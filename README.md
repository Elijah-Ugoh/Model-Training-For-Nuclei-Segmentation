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