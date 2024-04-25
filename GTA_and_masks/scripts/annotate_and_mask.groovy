// 1 is full resolution. You may want something more like 20 or higher for small thumbnails
downsample = 1 

// Fetch annotations
def annotations = getAnnotationObjects().findAll{it.getPathClass() == null}

def imageName = GeneralTools.stripExtension(getCurrentImageData().getServer().getMetadata().getName())
def imageData = getCurrentImageData()

// Make sure the location you want to save the files to exists - requires a Project
def pathOutput = buildFilePath('/Users/Elijah/Documents/BINP37_Research_Project/TMA_and_masks', 'image_export')
mkdirs(pathOutput)

def cellLabelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
    .useCells()
    .useInstanceLabels()
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported    
    .multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
    .build()
def annotationLabelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
    .addLabel('Tumor',1) //Each class requires a name and a number
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported    
    .multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
    .build()

// If no annotations found, inform the user
if (annotations.isEmpty()) {
    println("No annotations found in the image")
} else {
    annotations.eachWithIndex{anno,x->
        roi = anno.getROI()
        def requestROI = RegionRequest.createInstance(getCurrentServer().getPath(), 1, roi)

        def regionOutput = buildFilePath(pathOutput, imageName+"_region_"+x)
        
        // Now to export one image of each type per annotation (in the default case, unclassified)
        // Objects with overlays as seen in the Viewer    
        writeRenderedImageRegion(getCurrentViewer(), requestROI, regionOutput+"_rendered.tif")
        // Labeled images, either cells or annotations
        writeImageRegion(annotationLabelServer, requestROI, regionOutput+"_annotationLabels.tif")
        writeImageRegion(cellLabelServer, requestROI, regionOutput+"_cellLabels.tif")
        // To get the image behind the objects, you would simply use writeImageRegion
        writeImageRegion(getCurrentServer(), requestROI, regionOutput+"_original.tif")
    }
}

