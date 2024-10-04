import cv2
import numpy as np


def histogram_equalizer(filename):
    image = cv2.imread(filename, 0)  # filename path to image and 0 for grayscale read
    height, width = image.shape  # gives height and width of image
    total_pixels = height * width
    new_image = np.zeros(shape=(height, width), dtype=np.uint8)

    histogram = [0] * 256
    for i in range(height):
        for j in range(width):
            histogram[image[i][j]] += 1

    norm_histogram = [freq / total_pixels for freq in histogram]  # normalized hist divided by total pixels and rounding

    cdf = [0] * 256
    cdf[0] = norm_histogram[0]
    for i in range(1, len(norm_histogram)):  # calculating cumulative distribution function
        cdf[i] = norm_histogram[i] + cdf[i - 1]

    for i in range(height):
        for j in range(width):
            new_image[i][j] = int((256 - 1) * cdf[image[i][j]])  # getting new adjusted intensities

    cv2.imshow("histogram_equalizer", new_image)
    cv2.waitKey(0)
    cv2.destroyWindow("histogram_equalizer")


def flood_fill(seed_point, filename):  # tuple, image path, new_color. manipulates input image
    new_color = 255  # color to fill with a constant, can use different color as argument
    image = cv2.imread(filename, 0)
    height, width = image.shape

    old_color = image[seed_point[0]][seed_point[1]]  # old_color at seed pixel location
    if old_color == new_color: return
    frontier = [seed_point]
    image[seed_point[0]][seed_point[1]] = new_color

    while len(frontier) != 0:  # while frontier is not empty
        q = frontier.pop()
        x, y = q  # unwrapping

        # checking neighbors of q
        if x + 1 <= height - 1:
            if image[x + 1][y] == old_color:
                frontier.append((x + 1, y))
                image[x + 1][y] = new_color

        if y + 1 <= width - 1:
            if image[x][y + 1] == old_color:
                frontier.append((x, y + 1))
                image[x][y + 1] = new_color

        if x - 1 >= 0:
            if image[x - 1][y] == old_color:
                frontier.append((x - 1, y))
                image[x - 1][y] = new_color

        if y - 1 >= 0:
            if image[x][y - 1] == old_color:
                frontier.append((x, y - 1))
                image[x][y - 1] = new_color

    cv2.imshow("floodfill", image)
    cv2.waitKey(0)
    cv2.destroyWindow("floodfill")
    return image


def flood_fill_separate(seed_point, filename, output):  # seed, filename(input), new output for double-thresholding
    if type(filename) == str:
        image = cv2.imread(filename, 0)  # returns numpy array of image
    else:
        image = filename  # else image is provided as numpy array

    new_color = 254
    height, width = image.shape

    old_color = image[seed_point[0]][seed_point[1]]  # old_color at seed pixel location
    if old_color == new_color: return
    frontier = [seed_point]
    output[seed_point[0]][seed_point[1]] = new_color

    while len(frontier) != 0:  # while frontier is not empty
        q = frontier.pop()
        x, y = q  # unwrapping

        # checking neighbors of q
        if x + 1 <= height - 1:
            if image[x + 1][y] == old_color and output[x + 1][y] != new_color:
                frontier.append((x + 1, y))
                output[x + 1][y] = new_color

        if y + 1 <= width - 1:
            if image[x][y + 1] == old_color and output[x][y + 1] != new_color:
                frontier.append((x, y + 1))
                output[x][y + 1] = new_color

        if x - 1 >= 0:
            if image[x - 1][y] == old_color and output[x - 1][y] != new_color:
                frontier.append((x - 1, y))
                output[x - 1][y] = new_color

        if y - 1 >= 0:
            if image[x][y - 1] == old_color and output[x][y - 1] != new_color:
                frontier.append((x, y - 1))
                output[x][y - 1] = new_color

    return output


def simple_thresh(image, t):  # returns thresholded image
    height, width = image.shape
    output = np.zeros(shape=(height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if image[i][j] > t:
                output[i][j] = 255
            else:
                output[i][j] = 0

    return output


def double_thresholding(filename):
    if type(filename) == str:
        image = cv2.imread(filename, 0)  # returns numpy array of image
    else:
        image = filename  # else image is provided as numpy array

    height, width = image.shape
    low_thresh = simple_thresh(image, 100)  # 100 low threshold value
    high_thresh = simple_thresh(image, 170)  # 170 high threshold value
    double_thresh = np.zeros(shape=(height, width), dtype=np.uint8)  # initialization

    for i in range(height):  # using high threshold image seed pixel and apply floodfill on low threshold image
        for j in range(width):
            if high_thresh[i][j] == 255:
                pixel = (i, j)
                double_thresh = flood_fill_separate(pixel, low_thresh, double_thresh)

    cv2.imshow("double_thresh_image", double_thresh)
    cv2.waitKey(0)
    cv2.destroyWindow("double_thresh_image")
    return double_thresh


def dilation(filename):
    if type(filename) == str:
        image = cv2.imread(filename, 0)  # returns numpy array of image
    else:
        image = filename  # else image is provided as numpy array

    image = double_thresholding(image)
    height, width = image.shape
    new_output = np.zeros(shape=(height, width), dtype=np.uint8)
    b4 = [[0, 254, 0], [254, 254, 254], [0, 254, 0]]
    x, y = (1, 1)  # center of B

    for i in range(height):  # if atleast one pixel of image intersects with B set ON otherwise OFF
        for j in range(width):
            if b4[x][y] == image[i][j]:
                new_output[i][j] = 254
            elif i + 1 <= height - 1:
                if b4[x + 1][y] == image[i + 1][j]:
                    new_output[i][j] = 254
            elif i - 1 >= 0:
                if b4[x - 1][y] == image[i - 1][j]:
                    new_output[i][j] = 254
            elif j + 1 <= width - 1:
                if b4[x][y + 1] == image[i][j + 1]:
                    new_output[i][j] = 254
            elif j - 1 >= 0:
                if b4[x][y - 1] == image[i][j - 1]:
                    new_output[i][j] = 254
            else:
                new_output[i][j] = 0

    cv2.imshow("dilated_image", new_output)
    cv2.waitKey(0)
    cv2.destroyWindow("dilated_image")
    return image


def erosion(filename):
    if type(filename) == str:
        image = cv2.imread(filename, 0)  # returns numpy array of image
    else:
        image = filename  # else image is provided as numpy array

    image = double_thresholding(image)
    height, width = image.shape
    new_output = np.zeros(shape=(height, width), dtype=np.uint8)
    b4 = [[0, 254, 0], [254, 254, 254], [0, 254, 0]]
    x, y = (1, 1)  # center of B

    for i in range(height):  # if all pixels of image intersects with B set ON
        for j in range(width):
            if b4[x][y] == image[i][j]:
                if ((i + 1 <= height - 1) and (i - 1 >= 0)) and ((j + 1 <= width - 1) and (j - 1 >= 0)):
                    if ((b4[x + 1][y] == image[i + 1][j]) and (b4[x - 1][y] == image[i - 1][j])) and (
                            b4[x][y + 1] == image[i][j + 1] and (b4[x][y - 1] == image[i][j - 1])):
                        new_output[i][j] = 254

    cv2.imshow("eroded_image", new_output)
    cv2.waitKey(0)
    cv2.destroyWindow("eroded_image")
    return image
