import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardCopyOption

// Define source directory containing all the TMA core images
def sourceDir = '/Users/Elijah/Documents/BINP37_Research_Project/split_tma-cores/TMA_Cores'

// Define destination directory where the randomly selected images will be copied
def destinationDir = '/Users/Elijah/Documents/BINP37_Research_Project/randomly_selected_TMA_cores'

// Create destination directory if it doesn't exist
new File(destinationDir).mkdirs()

// // List all images from all folders
def allImages = []

new File(sourceDir).eachDir { folder ->
    def images = folder.listFiles({ file -> file.isFile() })
    println "${images.size()} images found in folder: ${folder.name}"
    allImages.addAll(images)
}
print("Listing all images...")

// Shuffle the list of images
allImages.shuffle()
print("Shuffling Completed")

// Select any 20 images from the shuffled list
def selectedImages = allImages.take(Math.min(20, allImages.size()))
print("Selected 20 images from the shuffled list")

// Copy the selected images to the destination directory
selectedImages.eachWithIndex { image, index ->
    // Get the name of the original folder (core)
    def coreName = image.parentFile.name
    // Remove the "EPCAM_" prefix from the core name
    def coreNameWithoutPrefix = coreName - 'EPCAM_'
    // Create a new file name without the "EPCAM_" prefix
    def destFileName = "${coreNameWithoutPrefix}_${index + 1}.${image.name.split("\\.").last()}"
    Files.copy(Paths.get(image.path), Paths.get(destinationDir, destFileName), StandardCopyOption.REPLACE_EXISTING)
}

print("Process Completed")
print("20 randomly selected TMA cores from all source directories saved to " + destinationDir)