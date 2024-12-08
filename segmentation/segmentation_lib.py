import cv2
import numpy
import numpy as np
import math

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

def dilation(filename):
    if type(filename) == str:
        image = cv2.imread(filename, 0)  # returns numpy array of image
    else:
        image = filename  # else image is provided as numpy array

    # image = double_thresholding(image)
    height, width = image.shape
    new_output = np.zeros(shape=(height, width), dtype=np.uint8)
    b4 = [[0, 255, 0], [255, 255, 255], [0, 255, 0]]
    x, y = (1, 1)  # center of B

    for i in range(height):  # if at least one pixel of image intersects with B set ON otherwise OFF
        for j in range(width):
            if b4[x][y] == image[i][j]:
                new_output[i][j] = 255
            elif i + 1 <= height - 1:
                if b4[x + 1][y] == image[i + 1][j]:
                    new_output[i][j] = 255
            elif i - 1 >= 0:
                if b4[x - 1][y] == image[i - 1][j]:
                    new_output[i][j] = 255
            elif j + 1 <= width - 1:
                if b4[x][y + 1] == image[i][j + 1]:
                    new_output[i][j] = 255
            elif j - 1 >= 0:
                if b4[x][y - 1] == image[i][j - 1]:
                    new_output[i][j] = 255
            else:
                new_output[i][j] = 0

    # cv2.imshow("dilated_image", new_output)
    # cv2.waitKey(0)
    # cv2.destroyWindow("dilated_image")
    return new_output


def erosion(filename):
    if type(filename) == str:
        image = cv2.imread(filename, 0)  # returns numpy array of image
    else:
        image = filename  # else image is provided as numpy array

    # image = double_thresholding(image)
    height, width = image.shape
    new_output = np.zeros(shape=(height, width), dtype=np.uint8)
    b4 = [[0, 255, 0], [255, 255, 255], [0, 255, 0]]
    x, y = (1, 1)  # center of B

    for i in range(height):  # if all pixels of image intersects with B set ON
        for j in range(width):
            if b4[x][y] and image[i][j] > 0:
                if ((i + 1 <= height - 1) and (i - 1 >= 0)) and ((j + 1 <= width - 1) and (j - 1 >= 0)):
                    if ((b4[x + 1][y] and image[i + 1][j] > 0) and (b4[x - 1][y] and image[i - 1][j] > 0)) and (
                            (b4[x][y + 1]  and image[i][j + 1] > 0) and (b4[x][y - 1] and image[i][j - 1] > 0)):
                        new_output[i][j] = 255

    # cv2.imshow("eroded_image", new_output)
    # cv2.waitKey(0)
    # cv2.destroyWindow("eroded_image")
    return new_output


def flood_fill_separate(seed_point, filename, output,
                        label=None):  # seed, filename(input), new output for double-thresholding
    if type(filename) == str:
        image = cv2.imread(filename, 0)  # returns numpy array of image
    else:
        image = filename  # else image is provided as numpy array

    if label:
        new_color = label
    else:
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



def simple_thresh(filename, t):  # returns thresholded image
    if type(filename) == str:
        image = cv2.imread(filename, 0)  # returns numpy array of image
    else:
        image = filename  # else image is provided as numpy array

    height, width = image.shape
    output = np.zeros(shape=(height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if image[i][j] < t:
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
    low_thresh = simple_thresh(image, 70)  # 100 low threshold value
    high_thresh = simple_thresh(image, 30)  # 180 high threshold value
    double_thresh = np.zeros(shape=(height, width), dtype=np.uint8)  # initialization

    for i in range(height):  # using high threshold image seed pixel and apply floodfill on low threshold image
        for j in range(width):
            if high_thresh[i][j] == 255:
                pixel = (i, j)
                double_thresh = flood_fill_separate(pixel, low_thresh, double_thresh)

    # cv2.imshow("double_thresh_image", double_thresh)
    # cv2.waitKey(0)
    # cv2.destroyWindow("double_thresh_image")
    new_height, new_width = double_thresh.shape
    for i in range(new_height):
        for j in range(new_width):
            if double_thresh[i][j] == 254:
                double_thresh[i][j] = 255
    return double_thresh



def clean_image(result):  # applying erosion and dilation to clean image
    i = erosion(result)
    a = dilation(i)
    b = erosion(a)
    c = dilation(b)
    d = dilation(c)
    
    return d


def ridler_calvard(filename):  # ridler calvard to find the best threshold value
    if type(filename) == str:
        image = cv2.imread(filename, 0)  # returns numpy array of image
    else:
        image = filename

    height, width = image.shape
    t = 50  # random starting threshold
    while True:
        mu_less = [0] * 256
        mu_greater = [0] * 256
        less_pixels_count = 0
        greater_pixels_count = 0

        for i in range(height):
            for j in range(width):
                if image[i][j] <= t:
                    mu_less[image[i][j]] += 1
                    less_pixels_count += 1
                else:
                    mu_greater[image[i][j]] += 1
                    greater_pixels_count += 1

        if less_pixels_count != 0:
            mu_l = round(sum([index * intensity for index, intensity in enumerate(mu_less)]) / less_pixels_count, 2)
        else:
            mu_l = 0

        if greater_pixels_count != 0:
            mu_g = round(sum([index * intensity for index, intensity in enumerate(mu_greater)]) / greater_pixels_count, 2)
        else:
            mu_g = 0

        new_t = (mu_l + mu_g) / 2
        if abs(t - new_t) < 0.01:
            return int(new_t)
        else:
            t = new_t




def connected_components_using_flood_fill(threshold_image):  # different components will have different labels
    # associated with them
    height, width = threshold_image.shape
    new_output = np.full(shape=(height,width), fill_value=-1, dtype=np.double)
    label = 1
    components = 0
    for i in range(height):  # if at least one pixel of image intersects with B set ON otherwise OFF
        for j in range(width):
            if new_output[i][j]==-1 and threshold_image[i][j] == 255:  # if threshold image ON that means a component
                new_output = flood_fill_separate((i, j), threshold_image, new_output, label)
                label = label + 1
                components += 1


    # cv2.imshow("connected", new_output)
    # cv2.waitKey(0)
    # cv2.destroyWindow("connected")
    return new_output, components


def isFrontON(I, label, p, d):
    r, c = p

    # Check boundaries for each direction (up, right, down, left)
    if d == 0:  # Up
        if r - 1 >= 0 and I[r - 1][c] == label:
            return True
    elif d == 1:  # Right
        if c + 1 < len(I[r]) and I[r][c + 1] == label:
            return True
    elif d == 2:  # Down
        if r + 1 < len(I) and I[r + 1][c] == label:
            return True
    elif d == 3:  # Left
        if c - 1 >= 0 and I[r][c - 1] == label:
            return True

    return False


def isLeftON(I, label, p, d):
    r, c = p
    d = d - 1
    if d < 0:
        d = 3

    return isFrontON(I, label, p, d)


def MoveForward(p, d):
    # if d < 0:
    # 	d = 3
    r, c = p
    if d == 0:
        return r - 1, c
    if d == 1:
        return r, c + 1
    if d == 2:
        return r + 1, c
    if d == 3:
        return r, c - 1


def wall_following(I, label, pixel):  # wall following for boundary
    height, width = I.shape
    p = pixel
    d = 0
    boundary = []

    p0 = p
    d0 = 0
    while isFrontON(I, label, p0, d0):
        d0 += 1

    d0 += 1
    while True:
        if p0 not in boundary:
            boundary.append(p0)
        if isLeftON(I, label, p0, d0):
            d0 -= 1  # turn left
            if d0 < 0:
                d0 = 3

            p0 = MoveForward(p0, d0)
        elif not isFrontON(I, label, p0, d0):
            d0 += 1
            if d0 > 3:
                d0 = 0

        else:
            p0 = MoveForward(p0, d0)


        if p0 == p and d0 == d: break
    return boundary


def draw_perimeter(image, boundary, color):  # draws the object boundary
    thickness = 2
    perimeter = boundary
    actual_image = image
    perimeter = [(b, a) for a, b in perimeter]
    perimeter = np.array(perimeter, np.int32).reshape((-1, 1, 2))
    actual_image = cv2.polylines(actual_image, perimeter, 1, color, thickness)
    return actual_image


def convolve(image, kernel):  # corresponding pixels of image and kernel are multiplied and then added together for every pixel
    if type(image) == str:
        image = cv2.imread(image, 0)  # returns numpy array of image
    else:
        image = image  # else image is provided as numpy array

    height, width = image.shape  # gives height and width of image
    ker_h, ker_w = kernel.shape

    h = np.zeros(shape=(height, width), dtype=np.double)  # output image
    for i in range(height):
        for j in range(width):
            sum = 0
            for k in range(ker_h):
                for m in range(ker_w):
                    offset_i = k - math.floor(ker_h / 2)
                    offset_j = m - math.floor(ker_w / 2)
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

    # cv2.imshow("horizontal_gradient", horizontal)
    # cv2.waitKey(0)
    # cv2.destroyWindow("horizontal_gradient")
    return horizontal


def vertical_gradient(image, g_kernel, g_prime_kernel):  # vertical gradient
    temporary_vertical = convolve(image, g_kernel)
    vertical = convolve(temporary_vertical, g_prime_kernel.T) # g_prime_kernel.T refers to transpose of gaussian derivative kernel(convolve image with vertical 1-D)

    # cv2.imshow("vertical_gradient", vertical)
    # cv2.waitKey(0)
    # cv2.destroyWindow("vertical_gradient")
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

    # cv2.imshow("magnitude", np.uint8(magnitude))
    # cv2.waitKey(0)
    # cv2.destroyWindow("magnitude")
    #
    # cv2.imshow("direction", direction)
    # cv2.waitKey(0)
    # cv2.destroyWindow("direction")

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


    # cv2.imshow("suppressed", np.uint8(sup))
    # cv2.waitKey(0)
    # cv2.destroyWindow("suppressed")
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

    # cv2.imshow("hysteresis", np.uint8(hyster))
    # cv2.waitKey(0)
    # cv2.destroyWindow("hysteresis")
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

    # cv2.imshow("Final_Edges", np.uint8(edges))
    # cv2.waitKey(0)
    # cv2.destroyWindow("Final_Edges")
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
    return chamfer_distance.astype(np.uint8)


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


def watershed(mag):
    height, width = mag.shape
    label = np.full(shape=(height, width),fill_value=-1, dtype=np.double)  # labeled output
    pixels = [[]] * 256  # precompute or map intensity levels to pixels
    global_label = 1
    frontier = []
    neighbors = [                           # offset
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    for i in range(height):
        for j in range(width):
            intensity = mag[i][j]
            pixels[intensity].append((i,j))

    for g in range(0,256):  # 0 - 255
        temp_label = label.copy()
        intensity_pixels = pixels[g]
        for p in intensity_pixels:  # grow existing catchment basins by one pixel, creating initial frontier
            if mag[p[0]][p[1]] == g:
                for dx, dy in neighbors:
                    new_i, new_j = p[0]+dx, p[1]+dy  # calculating new indices
                    if 0 <= new_i < height and 0 <= new_j < width:  # making sure new_indices are not out of bounds
                        if temp_label[new_i][new_j] >= 0:  # if already exists or part of some catchment basin
                            label[p[0]][p[1]] = label[new_i][new_j]  # label[p] = label[q]
                            frontier.append(p)  # frontier.pushback(p)


        while len(frontier) != 0:  # continue growing existing basins by one pixel
            p = frontier.pop(0)  # pop front
            for dx, dy in neighbors:
                new_i, new_j = p[0] + dx, p[1] + dy  # calculating new indices
                if 0 <= new_i < height and 0 <= new_j < width:  # making sure new_indices are not out of bounds
                    if mag[new_i][new_j] == g and label[new_i][new_j] == -1:
                        label[new_i][new_j] = label[p[0]][p[1]]
                        frontier.append((new_i,new_j))  # pushback(q) neighbor of p

        for p in intensity_pixels:  # create new catchment basin
            if mag[p[0]][p[1]] == g and label[p[0]][p[1]] == -1:  # unlabelled
                flood_fill_separate(p, mag, label, global_label)
                global_label += 1


    return label

def watershed_segmentation(image):  # provide the input filename
    if type(image) == str:
        image = cv2.imread(image, 0)  # returns numpy array of image
    else:
        image = image  # else image is provided as numpy array

    mag, dir = magnitude_direction(image, sigma=0.6)
    mag = np.uint8(mag)
    cv2.imshow("Normal Watershed: Magnitude", histogram_equalizer(mag))
    cv2.waitKey(0)
    cv2.destroyWindow("Normal Watershed: Magnitude")

    label = watershed(mag)# magnitude image is used for watershed
    label_show = (255 * label) / (np.max(label))
    cv2.imshow("Normal Watershed: Labels", np.uint8(label_show))
    cv2.waitKey(0)
    cv2.destroyWindow("Normal Watershed: Labels")
    return label


def marker_watershed(mag, marker):
    height, width = mag.shape
    label = np.full(shape=(height, width), fill_value=-1, dtype=np.double)  # labeled output
    pixels = [[]] * 256  # precompute or map intensity levels to pixels
    global_label = 1
    frontier = []
    neighbors = [
        (-1, -1),  # Top-left
        (-1, 0),   # Top
        (-1, 1),   # Top-right
        (0, 1),    # Right
        (1, 1),    # Bottom-right
        (1, 0),    # Bottom
        (1, -1),   # Bottom-left
        (0, -1)    # Left
    ]

    for i in range(height):
        for j in range(width):
            intensity = mag[i][j]
            pixels[intensity].append((i, j))


    label_image, num_labels = connected_components_using_flood_fill(marker)
    for label in range(1, num_labels):
        # Count pixels for each component
        component_size = np.sum(label_image == label)
        if component_size < 80:
            # Set small components to background (label 0)
            label_image[label_image == label] = 0

    label = label_image
   
    for g in range(0, 256):  # 0 - 255
        temp_label = label.copy()
        intensity_pixels = pixels[g]
        for p in intensity_pixels:  # grow existing catchment basins by one pixel, creating initial frontier
            if mag[p[0]][p[1]] == g:
                for dx, dy in neighbors:
                    new_i, new_j = p[0] + dx, p[1] + dy  # calculating new indices
                    if 0 <= new_i < height and 0 <= new_j < width:  # making sure new_indices are not out of bounds
                        if temp_label[new_i][new_j] >= 0:  # if already exists or part of some catchment basin
                            label[p[0]][p[1]] = label[new_i][new_j]  # label[p] = label[q]
                            frontier.append(p)  # frontier.pushback(p)

        while len(frontier) != 0:  # continue growing existing basins by one pixel
            p = frontier.pop(0)  # pop front
            for dx, dy in neighbors:
                new_i, new_j = p[0] + dx, p[1] + dy  # calculating new indices
                if 0 <= new_i < height and 0 <= new_j < width:  # making sure new_indices are not out of bounds
                    if mag[new_i][new_j] <= g and label[new_i][new_j] == -1:
                        label[new_i][new_j] = label[p[0]][p[1]]
                        frontier.append((new_i, new_j))  # pushback(q) neighbor of p


    
    label_show = (255 * label) / (np.max(label))
    cv2.imshow("Marker Watershed: Labels", np.uint8(label_show))
    cv2.waitKey(0)
    cv2.destroyWindow("Marker Watershed: Labels")
    return label

def marker(threshold_image, edge_image):  # performing OR operation between edge image and thresholded image for marker
    height, width = edge_image.shape
    marker = np.zeros(shape=(height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if threshold_image[i][j] > 0:
                marker[i][j] = threshold_image[i][j]
            if edge_image[i][j] > 0:
                marker[i][j] = edge_image[i][j]

    return marker


def watershed_segmentation_using_markers(image):
    if type(image) == str:
        image = cv2.imread(image, 0)  # returns numpy array of image
    else:
        image = image  # else image is provided as numpy array

    mag, dir = magnitude_direction(image, sigma=0.6)
    mag = np.uint8(mag)
    mag = histogram_equalizer(mag)
    cv2.imshow("Marker Watershed: Magnitude", mag)  # magnitude image
    cv2.waitKey(0)
    cv2.destroyWindow("Marker Watershed: Magnitude")

    print(ridler_calvard(image))
    g_kernel = gaussian_kernel(1.0)
    image = convolve(image, g_kernel)
    image = np.uint8(image)

    threshold_image = double_thresholding(image)  # thresholded image
    threshold_image = clean_image(threshold_image)
    cv2.imshow("Marker Watershed: Threshold", threshold_image)
    cv2.waitKey(0)
    cv2.destroyWindow("Marker Watershed: Threshold")


    chamfer_distance = chamfer(threshold_image)  # chamfer on thresholded image
    cv2.imshow("Marker Watershed: Chamfer", chamfer_distance)
    cv2.waitKey(0)
    cv2.destroyWindow("Marker Watershed: Chamfer")


    normal_watershed_chamfer = watershed(chamfer_distance)  # normal watershed on chamfer distance
    norm_cham_viz = (255 * normal_watershed_chamfer) / (np.max(normal_watershed_chamfer))
    cv2.imshow("Marker Watershed: Watershed of Chamfer", np.uint8(norm_cham_viz))
    cv2.waitKey(0)
    cv2.destroyWindow("Marker Watershed: Watershed of Chamfer")


    normal_watershed_edge_image = canny_edge(normal_watershed_chamfer, sigma=1.0)  # canny edge performed on watershed chamfer
    cv2.imshow("Marker Watershed: Edges separating objects", np.uint8(normal_watershed_edge_image))
    cv2.waitKey(0)
    cv2.destroyWindow("Marker Watershed: Edges separating objects")



    marker_image = marker(threshold_image,normal_watershed_edge_image)  # marker generated using thresholded image and edge image
    cv2.imshow("Marker Watershed: Watershed Marker", marker_image)
    cv2.waitKey(0)
    cv2.destroyWindow("Marker Watershed: Watershed Marker")

    label = marker_watershed(mag, marker_image)
    return label


def draw(image, actual_image):
    visited = set()
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            if image[i][j] != 0:
                label = image[i][j]
                if label not in visited:
                    visited.add(label)
                    perimeter = sorted(wall_following(image, label, (i, j)))
                    actual_image = draw_perimeter(actual_image, perimeter, (0, 255, 0))

    cv2.imshow("ridge lines", actual_image)
    cv2.waitKey(0)
    cv2.destroyWindow("ridge lines")