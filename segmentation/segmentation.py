from segmentation_lib import *
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename

if __name__ == '__main__':

    options = sys.argv

    Tk().withdraw()
    filename = askopenfilename()  # first option is for main file
    # template_file = askopenfilename()  # second option is for any template file
    # sigma = 1.0  # can use 0.6 or 1 or some other number
    actual_image = cv2.imread(filename, 1)
    watershed_segmentation(filename)   # normal watershed segmentation
    image = watershed_segmentation_using_markers(filename)  # watershed segmentation using markers
    draw(image, actual_image)

    exit(0)

    main_edge_image = canny_edge(filename, sigma)  # returns final edges
    main_chamfer = chamfer(main_edge_image)  # returns chamfer distance for that edge image

    if len(template_file) != 0:
        template_edge_image = canny_edge(template_file, sigma)  # returns final edges for template
        template_chamfer = chamfer(template_edge_image)  # returns chamfer distance for that edge image

        # out = ssd(main_chamfer, template_chamfer)