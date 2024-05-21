from stardist.models import StarDist2D

# Define the paths
basedir = 'c:/Users/Elijah/Documents/BINP37_Research_Project/first_training/models/models/stardist'
exported_model_dir = 'c:/Users/Elijah/Downloads'

# Initialize the StarDist2D model
model = StarDist2D(None, name='dlmodel2', basedir=basedir)

# Export the model to TF format
model.export_TF(exported_model_dir)

# Extract the contents of the zip file
from zipfile import ZipFile
with ZipFile('c:/Users/Elijah/Documents/BINP37_Research_Project/first_training/models/models/stardist/stardistsavedmodel.zip', 'r') as zipObj:
    zipObj.extractall(exported_model_dir)

# Convert the exported model to ONNX format
import os
os.system(f'python -m tf2onnx.convert --opset 10 --saved-model "{exported_model_dir}" --output_frozen_graph "{exported_model_dir}/dlmodel2.pb"')
