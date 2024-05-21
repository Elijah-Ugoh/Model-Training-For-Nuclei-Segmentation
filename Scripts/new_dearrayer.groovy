import javax.imageio.ImageIO
import qupath.lib.regions.RegionRequest

// Define resolution - 1.0 means full size
double downsample = 1.0

// Check if the TMA cores have already been dearrayed. If they haven't been dearrayed yet, the TMA dearraying operation is performed 
if (!isTMADearrayed()) {
	runPlugin('qupath.imagej.detect.dearray.TMADearrayerPluginIJ', '{"coreDiameterMM":1.3,"labelsHorizontal":"A-M","labelsVertical":"1-10","labelOrder":"Row first","densityThreshold":5,"boundsScale":105}')
	}

// Define the directory paths
// Replace qupathProjectDir with the path where the TMA cores will be saved and and tmaImagesDir with the TMA location
def qupathProjectDir = '/Users/Elijah/Documents/BINP37_Research_Project/split_tma-cores'
def tmaImagesDir = '/Users/Elijah/Documents/BINP37_Research_Project/TMA/EPCAM'

// Create output directory inside the QuPath project directory
def outputDir = buildFilePath(qupathProjectDir, 'EPCAM_20A_cores')
mkdirs(outputDir)

// Process TMA images
def server = getCurrentImageData().getServer()
def path = server.getPath()
getTMACoreList().parallelStream().forEach { core ->
    img = server.readRegion(RegionRequest.createInstance(path, downsample, core.getROI()))
    
    // Extract the core name from its path
    def coreName = core.getName().substring(core.getName().lastIndexOf('/') + 1)
    
    // Save the image to the output directory
    ImageIO.write(img, 'TIF', new File(outputDir, coreName + '.tif'))
}
print('De-arraying Completed!')
print("Images saved in: " + outputDir)