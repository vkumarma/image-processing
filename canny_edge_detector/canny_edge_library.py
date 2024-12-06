import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def histogram_equalizer(filename):
    if type(filename) == str:
        image = cv2.imread(filename, 0)  # returns numpy array of image
    else:
        image = filename  # else image is provided as numpy array

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
    return new_image


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

    cv2.imshow("magnitude", histogram_equalizer(np.uint8(magnitude)))
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


    cv2.imshow("suppressed", histogram_equalizer(np.uint8(sup)))
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
    edge = np.uint8(edge_image.copy())  # Ensure the image is in uint8 format
    height, width = edge.shape
    chamfer_distance = np.full((height, width), float('inf'))  # Initialize chamfer distance with infinity

    # First pass: top-left to bottom-right
    for i in range(height):  
        for j in range(width):
            if edge[i][j] > 0:  # if it's an edge pixel (ON pixel)
                chamfer_distance[i][j] = 0
            else:
                # Compare with the pixel above and the pixel to the left
                if i > 0:  # Check the pixel above
                    chamfer_distance[i][j] = min(chamfer_distance[i][j], 1 + chamfer_distance[i - 1][j])
                if j > 0:  # Check the pixel to the left
                    chamfer_distance[i][j] = min(chamfer_distance[i][j], 1 + chamfer_distance[i][j - 1])

    # Second pass: bottom-right to top-left
    for i in range(height - 1, -1, -1):  # Start from the bottom
        for j in range(width - 1, -1, -1):  # Start from the right
            if edge[i][j] > 0:  # if it's an edge pixel (ON pixel)
                chamfer_distance[i][j] = 0
            else:
                # Compare with the pixel below and the pixel to the right
                if i < height - 1:  # Check the pixel below
                    chamfer_distance[i][j] = min(chamfer_distance[i][j], 1 + chamfer_distance[i + 1][j])
                if j < width - 1:  # Check the pixel to the right
                    chamfer_distance[i][j] = min(chamfer_distance[i][j], 1 + chamfer_distance[i][j + 1])

    # Return the Chamfer distance image
    cv2.imshow("Final_Edges", histogram_equalizer(np.uint8(chamfer_distance)))
    cv2.waitKey(0)
    cv2.destroyWindow("Final_Edges")
    return chamfer_distance.astype(np.uint8)


def canny_edge(filename, sigma):  # canny edge detector
    mag, dir = magnitude_direction(filename, sigma)
    suppressed = non_maximal_suppression(mag, dir)
    edge_image = edge_linking(suppressed)
    return edge_image


def ssd(image, kernel, actual_image):  # Sum of Square Distance for template matching. Where pixel intensity if minimum that is location of template in an image
    template = kernel
    template_height, template_width = template.shape
    height, width = image.shape
    h = np.zeros(shape=(height, width), dtype=np.double)
    # Initialize a list to store SSD values for each location
    ssd_values = []

    # Loop over every possible position of the template in the image
    for y in range(image.shape[0] - template_height):
        for x in range(image.shape[1] - template_width):
            # Extract the subregion from the image where the template is being compared
            subregion = image[y:y + template_height, x:x + template_width]
            
            # Calculate the SSD (sum of squared differences) between the template and the subregion
            ssd = np.sum((subregion - template) ** 2)
            
            # Append the SSD value for this position
            ssd_values.append((x, y, ssd))
            h[y][x] = ssd

    # Find the position with the minimum SSD value (best match)
    best_match = min(ssd_values, key=lambda x: x[2])

    # Extract the coordinates of the best match
    best_x, best_y, min_ssd = best_match

    # Draw a rectangle around the best match (for visualization)
    top_left = (best_x-20, best_y+15)
    bottom_right = (best_x + template_width, best_y + template_height)
    cv2.rectangle(actual_image, top_left, bottom_right, (255, 0, 0), 2)

    cv2.imshow("ssd", np.uint8(h))
    cv2.waitKey(0)
    cv2.destroyWindow("ssd")

    # Display the result
    plt.imshow(actual_image, cmap='gray')
    plt.title(f"Best Match at ({best_x}, {best_y}) with SSD: {min_ssd}")
    plt.show()

    print(f"Best match found at ({best_x}, {best_y}) with SSD: {min_ssd}")
    return actual_image

# kernel = np.full((3, 5), 1/15)
# kernel = np.array([[0, -1, 0], [-1, 5, -1],[0, -1, 0]])


