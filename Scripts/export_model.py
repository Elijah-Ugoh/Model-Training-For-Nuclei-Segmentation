# Replace value of the basedir parameter with the directory path 
# where you want to save the exported TensorFlow model
# Also replace value in the name variable with the name of the trained model.

from stardist.models import StarDist2D
model = StarDist2D(None, name='stardist', basedir='.')
model.export_TF()