from vkmlib import *
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename

if __name__ == '__main__':

    options = sys.argv
    Tk().withdraw()
    filename = askopenfilename()
    actual_image = cv2.imread(filename, 1)  # actual image without any processing
    thresholded_image = display_and_return_double_threshold(filename)
    cleaned_image = clean_image(thresholded_image)
    connected_components_image = connected_components_using_flood_fill(cleaned_image)  # (output, num_components)
    connect_image = connected_components_image[0]

    stem_removed_image = remove_stem(cleaned_image)
    banana_regions = list(find_banana_intensities(connect_image)[0])
    centroids = find_banana_intensities(connect_image)[1]
    image = final_components(connect_image, banana_regions, centroids, stem_removed_image)  # final components
    classification(image, actual_image)

    if len(options) > 2:
        if options[1] == 'histogram_equalizer':  # provide image path
            histogram_equalizer(filename)
        elif options[1] == 'flood_fill':  # provide seed tuple as example: 100,100 and image path
            a = options[2].split(",")
            flood_fill((int(a[0]), int(a[1])), options[3])
        elif options[1] == 'double_thresholding':
            double_thresholding(options[2])
        elif options[1] == 'dilation':
            dilation(options[2])
        elif options[1] == 'erosion':
            erosion(options[2])
        else:
            print("Incorrect input")
            exit(0)

# floodfill command: python3 vkm.py flood_fill 200,200 /Users/vivekkumarmaheshwari/Downloads/image.ppm
# dilation / other functions command: python3 vkm.py dilation /Users/vivekkumarmaheshwari/Downloads/fruit1.bmp
# python3 file.py function name seed(optional) image path


# To run program 2:
#python3 vkm.py double_threshold /Users/vivekkumarmaheshwari/Downloads/fruit1.bmp