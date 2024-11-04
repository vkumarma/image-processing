import cv2
import numpy
import numpy as np
import math


def convolve(image, kernel):  # corresponding pixels of image and kernel are multiplied and then added together for every pixel
    if type(image) == str:
        image = cv2.imread(image, 0)  # returns numpy array of image
    else:
        image = image  # else image is provided as numpy array

    height, width = image.shape  # gives height and width of image
    # if len(kernel.shape) == 1:
    #     ker_h, ker_w = kernel.shape[0], 0
    # else:
    ker_h, ker_w = kernel.shape

    h = np.zeros(shape=(height, width), dtype=np.double)  # output image
    for i in range(height):
        for j in range(width):
            sum = 0
            for k in range(ker_h):
                for m in range(ker_w):
                    offset_i = -1 * math.floor(ker_h / 2) + k
                    offset_j = -1 * math.floor(ker_w / 2) + m
                    if (0 <= i + offset_i < image.shape[0]) and (0 <= j + offset_j < image.shape[1]):
                        sum += image[i+offset_i][j+offset_j] * kernel[k][m]

            h[i][j] = sum

    return h

def gaussian_kernel(sigma):  # returns 1D smoothing kernel. Central element is bigger than rest of the elements
    a = round(2.5 * sigma - 0.5)  # sigma refers to standard deviation
    w = 2 * a + 1
    sum = 0
    G = [0] * w
    for i in range(0, w):
        G[i] = math.exp((-1 * (i-a) * (i-a)) / (2 * sigma * sigma))
        sum = sum + G[i]

    for i in range(0, w):
        G[i] = G[i] / sum
    return np.array([G])  # returns horizontal kernel


def gaussian_derivative_kernel(sigma):  # used to identify edges, reduce noise, and extract features
    a = round(2.5 * sigma - 0.5)
    w = 2 * a + 1
    sum = 0
    G_prime = [0] * w
    for i in range(0, w):
        G_prime[i] = -1 * (i-a) * math.exp((-1 * (i-a) * (i-a)) / (2 * sigma * sigma))
        sum = sum - (i * G_prime[i])  # can change - to +

    for i in range(0, w):
        G_prime[i] = G_prime[i] / sum

    return np.array([G_prime])

def horizontal_gradient(image, g_kernel, g_prime_kernel):  # horizontal gradient
    temporary_horizontal = convolve(image, g_kernel.T)  # g_kernel.T refers to transpose of gaussian kernel (convolve image with vertical 1-D)
    horizontal = convolve(temporary_horizontal, g_prime_kernel)

    cv2.imshow("horizontal_gradient", np.uint8(horizontal))
    cv2.waitKey(0)
    cv2.destroyWindow("horizontal_gradient")
    return horizontal


def vertical_gradient(image, g_kernel, g_prime_kernel):  # vertical gradient
    temporary_vertical = convolve(image, g_kernel)
    vertical = convolve(temporary_vertical, g_prime_kernel.T) # g_prime_kernel.T refers to transpose of gaussian derivative kernel(convolve image with vertical 1-D)

    cv2.imshow("vertical_gradient", np.uint8(vertical))
    cv2.waitKey(0)
    cv2.destroyWindow("vertical_gradient")
    return vertical

def magnitude_direction(image, sigma):  # returns edges magnitude and edges direction
    if type(image) == str:
        image = cv2.imread(image, 0)  # returns numpy array of image
    else:
        image = image  # else image is provided as numpy array

    height, width = image.shape

    magnitude = np.zeros(shape=(height, width), dtype=np.double)
    direction = np.zeros(shape=(height, width), dtype=np.double)

    g_kernel = gaussian_kernel(sigma)
    g_prime_kernel = gaussian_derivative_kernel(sigma)

    horizontal = horizontal_gradient(image, g_kernel, g_prime_kernel)
    vertical = vertical_gradient(image, g_kernel, g_prime_kernel)

    for i in range(height):
       for j in range(width):
           magnitude[i][j] = np.sqrt((vertical[i][j] ** 2) + (horizontal[i][j] ** 2))
           direction[i][j] = np.arctan2(horizontal[i][j], vertical[i][j])

    cv2.imshow("magnitude", np.uint8(magnitude))
    cv2.waitKey(0)
    cv2.destroyWindow("magnitude")

    cv2.imshow("direction", direction)
    cv2.waitKey(0)
    cv2.destroyWindow("direction")

    return magnitude, direction


def non_maximal_suppression(mag, dir):  # for every pixel (i,j) in magnitude check if it is the local maximum if not then suppress that edge by setting it to zero
    height, width = mag.shape
    sup = np.zeros(shape=(height, width), dtype=np.double)  # suppression output
    for i in range(height):
       for j in range(width):
           sup[i][j] = mag[i][j]
           theta = dir[i][j]
           if theta < 0:  # easier if theta +ve
               theta = theta + math.pi
           theta = (180/math.pi) * theta  # radians to degree

           if theta < 22.5 or theta > 157.5:
               if i > 0 and i < height - 1:
                   if mag[i][j] < mag[i-1][j] or mag[i][j] < mag[i+1][j]:
                       sup[i][j] = 0

           if theta >= 22.5 and theta <= 67.5:
               if i > 0 and i < height - 1 and j > 0 and j < width - 1:
                   if mag[i][j] < mag[i-1][j-1] or mag[i][j] < mag[i+1][j+1]:
                       sup[i][j] = 0

           if theta > 67.5 and theta <= 112.5:
               if j > 0 and j < width - 1:
                   if mag[i][j] < mag[i][j-1] or mag[i][j] < mag[i][j+1]:
                       sup[i][j] = 0

           if theta > 112.5 and theta <= 157.5:
              if i > 0 and i < height - 1 and j > 0 and j < width - 1:
                   if mag[i][j] < mag[i-1][j+1] or mag[i][j] < mag[i+1][j-1]:
                       sup[i][j] = 0


    cv2.imshow("suppressed", np.uint8(sup))
    cv2.waitKey(0)
    cv2.destroyWindow("suppressed")
    return sup

def hysteresis(suppressed_image):  # thresholding
    sup_copy = suppressed_image.copy()
    flattened_pixels = sup_copy.flatten()  # array flattened
    sorted_pixels = np.sort(flattened_pixels)  # image pixels sorted

    percentile_90 = np.percentile(sorted_pixels, 90)  #t_hi
    percentile_20 = np.percentile(sorted_pixels, 20)  #t_lo

    hyster = suppressed_image.copy()
    height, width = hyster.shape
    for i in range(height):
        for j in range(width):
            if hyster[i][j] > percentile_90:  #t_hi
                hyster[i][j] = 255  # strong edge
            elif hyster[i][j] > percentile_20:  #t_lo
                hyster[i][j] = 125  # weak edge
            else:
                hyster[i][j] = 0  #non edge

    cv2.imshow("hysteresis", np.uint8(hyster))
    cv2.waitKey(0)
    cv2.destroyWindow("hysteresis")
    return hyster


def edge_linking(suppressed_image):
    hysteresis_output = hysteresis(suppressed_image)  # create a copy of hysteresis output
    edges = hysteresis_output.copy()
    height, width = edges.shape

    # 8-connected neighbor offsets
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    for i in range(height):
        for j in range(width):
            if hysteresis_output[i][j] == 125:  # weak pixel
                has_strong_connection = False   # Flag to check if there's a strong connection
                for dx, dy in neighbors:
                    new_i, new_j = i+dx, j+dy  # calculating new indices
                    if 0 <= new_i < height and 0 <= new_j < width:  # making sure new_indices are not out of bounds
                        if hysteresis_output[new_i][new_j] == 255:  # strong pixel connection
                            has_strong_connection = True
                            break  # no need to check further as found one strong pixel connection with the neighbor

                if has_strong_connection: # found atleast one strong connection with neighboring pixel
                    edges[i][j] = 255
                else:
                    edges[i][j] = 0  # no strong connection with any neighboring pixel therefore suppress

    cv2.imshow("Final_Edges", np.uint8(edges))
    cv2.waitKey(0)
    cv2.destroyWindow("Final_Edges")
    return edges


def chamfer(edge_image):  # distance of every pixel from edge
    edge = np.uint8(edge_image.copy())
    height, width = edge.shape
    chamfer_distance = np.zeros(shape=(height, width), dtype=np.uint8)
    for i in range(height):  # pass 1
        for j in range(width):
            if edge[i][j] > 0:  # if ON pixel
                chamfer_distance[i][j] = 0
            else:
                chamfer_distance[i][j] = min(float('inf'), 1+chamfer_distance[i-1][j], 1+chamfer_distance[i][j-1])


    for i in range(height-1,-1,-1): # pass 2
        for j in range(width-1,-1,-1):
            if edge[i][j] > 0:  # if ON pixel
                chamfer_distance[i][j] = 0
            else:
                if i < height - 1:  # Check the bottom pixel
                    chamfer_distance[i][j] = min(chamfer_distance[i][j], 1 + chamfer_distance[i + 1][j])
                if j < width - 1:  # Check the right pixel
                    chamfer_distance[i][j] = min(chamfer_distance[i][j], 1 + chamfer_distance[i][j + 1])

    cv2.imshow("Chamfer_distance", chamfer_distance)
    cv2.waitKey(0)
    cv2.destroyWindow("Chamfer_distance")
    return chamfer_distance


def canny_edge(filename, sigma):  # canny edge detector
    mag, dir = magnitude_direction(filename, sigma)
    suppressed = non_maximal_suppression(mag, dir)
    edge_image = edge_linking(suppressed)
    return edge_image


def ssd(image, kernel):  # Sum of Square Distance for template matching. Where pixel intensity if minimum that is location of template in an image

    height, width = image.shape
    ker_h, ker_w = kernel.shape

    h = np.zeros(shape=(height, width), dtype=np.uint64)  # output image
    for i in range(height):
        for j in range(width):
            sum = 0
            for k in range(ker_h):
                for m in range(ker_w):
                    offset_i = -1 * math.floor(ker_h // 2) + k
                    offset_j = -1 * math.floor(ker_w / 2) + m
                    if (0 <= i + offset_i < image.shape[0]) and (0 <= j + offset_j < image.shape[1]):
                        sum += ((image[i + offset_i][j + offset_j] - kernel[k][m]) ** 2)
            h[i][j] = sum

    cv2.imshow("ssd_map", np.uint8(h))
    cv2.waitKey(0)
    cv2.destroyWindow("ssd_map")
    return h


# kernel = np.full((3, 5), 1/15)
# kernel = np.array([[0, -1, 0], [-1, 5, -1],[0, -1, 0]])


