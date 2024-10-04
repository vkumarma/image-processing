from vkmlib import *
import sys

if __name__ == '__main__':
    options = sys.argv

    if options[1] == 'histogram_equalizer':  # provide image path
        histogram_equalizer(options[2])
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
