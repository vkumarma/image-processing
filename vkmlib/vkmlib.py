import cv2
import numpy
import numpy as np
import math


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
    new_color = 255
    # color to fill with a constant, can use different color as argument
    if type(filename) == str:
        image = cv2.imread(filename, 0)  # returns numpy array of image
    else:
        image = filename

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
    high_thresh = simple_thresh(image, 180)  # 180 high threshold value
    double_thresh = np.zeros(shape=(height, width), dtype=np.uint8)  # initialization

    for i in range(height):  # using high threshold image seed pixel and apply floodfill on low threshold image
        for j in range(width):
            if high_thresh[i][j] == 255:
                pixel = (i, j)
                double_thresh = flood_fill_separate(pixel, low_thresh, double_thresh)

    # cv2.imshow("double_thresh_image", double_thresh)
    # cv2.waitKey(0)
    # cv2.destroyWindow("double_thresh_image")
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

    for i in range(height):  # if at least one pixel of image intersects with B set ON otherwise OFF
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

    # cv2.imshow("dilated_image", new_output)
    # cv2.waitKey(0)
    # cv2.destroyWindow("dilated_image")
    return new_output


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
            if b4[x][y] and image[i][j] > 0:
                if ((i + 1 <= height - 1) and (i - 1 >= 0)) and ((j + 1 <= width - 1) and (j - 1 >= 0)):
                    if ((b4[x + 1][y] and image[i + 1][j] > 0) and (b4[x - 1][y] and image[i - 1][j] > 0)) and (
                            (b4[x][y + 1]  and image[i][j + 1] > 0) and (b4[x][y - 1] and image[i][j - 1] > 0)):
                        new_output[i][j] = 254

    # cv2.imshow("eroded_image", new_output)
    # cv2.waitKey(0)
    # cv2.destroyWindow("eroded_image")
    return new_output


def ridler_calvard(filename):  # ridler calvard to find the best threshold value
    if type(filename) == str:
        image = cv2.imread(filename, 0)  # returns numpy array of image
    else:
        image = filename

    height, width = image.shape
    t = 150  # random starting threshold
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

        mu_l = round(sum([index * intensity for index, intensity in enumerate(mu_less)]) / less_pixels_count, 2)
        mu_g = round(sum([index * intensity for index, intensity in enumerate(mu_greater)]) / greater_pixels_count, 2)
        new_t = (mu_l + mu_g) / 2
        if abs(t - new_t) < 0.01:
            return int(new_t)
        else:
            t = new_t


# print(ridler_calvard('/Users/vivekkumarmaheshwari/Downloads/fruit1.bmp'))  # 128
# # print(simple_thresh('/Users/vivekkumarmaheshwari/Downloads/fruit1.bmp', 150))


def display_and_return_double_threshold(filename):  # double thresholded image
    double_image = double_thresholding(filename)
    height,width = double_image.shape
    cv2.imshow("double_threshold_image", double_image)
    cv2.waitKey(0)
    cv2.destroyWindow("double_threshold_image")
    return double_image


def clean_image(result):  # applying erosion and dilation to clean image
    i = erosion(result)
    a = dilation(i)
    b = erosion(a)
    c = dilation(b)
    d = dilation(c)
    cv2.imshow("open image", d)
    cv2.waitKey(0)
    cv2.destroyWindow("open image")
    return d


def connected_components_using_flood_fill(threshold_image):  # different components will have different labels
    # associated with them
    labels = set()
    height, width = threshold_image.shape
    new_output = np.zeros(shape=(height, width), dtype=np.uint8)
    label = 40
    components = 0
    for i in range(height):  # if at least one pixel of image intersects with B set ON otherwise OFF
        for j in range(width):
            if new_output[i][j] == 0 and threshold_image[i][j] == 254:  # if threshold image ON that means a component
                new_output = flood_fill_separate((i, j), threshold_image, new_output, label)
                labels.add(label)
                label = label + 20
                components += 1


    cv2.imshow("connected", new_output)
    cv2.waitKey(0)
    cv2.destroyWindow("connected")
    return new_output, components, labels


def computations(components_image, region):  # moments, central moments, angle, area, eigen values, eccentricity calculation
    height, width = components_image.shape
    m00 = 0
    m01 = 0
    m10 = 0
    m20 = 0
    m02 = 0
    m11 = 0
    for i in range(height):  # if at least one pixel of image intersects with B set ON otherwise OFF
        for j in range(width):
            if components_image[i][j] == region:  # calculating moments for the region first
                m00 += 1
                m10 += i
                m01 += j
                m20 += (i ** 2)
                m02 += (j ** 2)
                m11 += (i * j)

    xc, yc = (m10 / m00, m01 / m00)  # calculating central moments
    mu00 = m00
    mu11 = m11 - (yc * m10)
    mu20 = m20 - (xc * m10)  # variance in x direction
    mu02 = m02 - (yc * m01)  # variance in y direction

    eigen_value1 = (1 / (2 * mu00)) * (mu20 + mu02 + math.sqrt((mu20 - mu02) ** 2 + (4 * (mu11 ** 2))))
    eigen_value2 = (1 / (2 * mu00)) * (mu20 + mu02 - math.sqrt((mu20 - mu02) ** 2 + (4 * (mu11 ** 2))))

    direction = 0.5 * float(np.arctan2(2 * mu11, mu20 - mu02))
    eccentricity = math.sqrt((eigen_value1 - eigen_value2) / eigen_value1)
    area = m00
    return area, direction, (eigen_value1, eigen_value2), eccentricity, (xc, yc)


def isFrontON(I, label, p, d):
    r, c = p
    if d == 0:
        if I[r - 1][c] == label:
            return True
    if d == 1:
        if I[r][c + 1] == label:
            return True
    if d == 2:
        if I[r + 1][c] == label:
            return True
    if d == 3:
        if I[r][c - 1] == label:
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


def remove_stem(cleaned_image):  # removing stem using erosion dilation
    e = erosion(cleaned_image)
    for i in range(5):
        e = erosion(e)

    d = e
    for j in range(2):
        d = dilation(d)

    # cv2.imshow("stem removed", d)
    # cv2.waitKey(0)
    # cv2.destroyWindow("stem removed")
    return d


def distance(centroid, pixel):  # euclidean distance
    xc, yc = centroid
    x1, y1 = pixel
    distance = math.sqrt(((xc - x1) ** 2) + ((yc - y1) ** 2))
    return distance


def find_banana_intensities(image):  # find banana labels
    height, width = image.shape
    regions = []
    visited = set()
    centroids = []
    for i in range(height):
        for j in range(width):
            if image[i][j] != 0:
                label = image[i][j]
                if label not in visited:
                    v = computations(image, label)
                    if v[3] > 0.5:  # since we know by analysis that bananas have eccentricity > 0.5
                        regions.append(int(label))
                        centroids.append(v[4])
                    visited.add(label)
    return regions, centroids


def final_components(image, banana_regions, centroids, stem_removed_image):  # final components with stems detected
    height, width = image.shape
    label1 = 200
    label2 = 240
    distances = []
    starting_pixels = []
    flag = True
    k = 0
    dis = 0

    while k < len(banana_regions):  # since two bananas therefore check max distances from between centroid and starting stem pixel
        for i in range(height):
            for j in range(width):

                if stem_removed_image[i][j] == 0 and image[i][j] == banana_regions[k]:
                    if int(distance(centroids[k], (i, j))) > dis:
                        dis = int(distance(centroids[k], (i, j)))

        distances.append(dis)
        dis = 0
        k += 1

    k = 0
    while k < len(banana_regions):  # get starting pixels for stems
        for i in range(height):
            for j in range(width):

                if stem_removed_image[i][j] == 0 and image[i][j] == banana_regions[k]:
                    if int(distance(centroids[k], (i, j))) == distances[k]:
                        if (i, j) not in starting_pixels:
                            starting_pixels.append((i, j))
                            flag = False
                            break

            if not flag:
                break
        k += 1
        flag = True

    for i in range(height):  # nearest pixels to the starting pixels form the sub component
        for j in range(width):

            if stem_removed_image[i][j] == 0 and image[i][j] == banana_regions[0]:
                if int(distance(starting_pixels[0], (i, j))) < 40:
                    image[i][j] = label1

            if stem_removed_image[i][j] == 0 and image[i][j] == banana_regions[1]:
                if int(distance(starting_pixels[1], (i, j))) < 40:
                    image[i][j] = label2

    cv2.imshow("stem detected", image)
    cv2.waitKey(0)
    cv2.destroyWindow("stem detected")
    return image

def draw_perimeter(image, boundary, color):  # draws the object boundary
    thickness = 2
    perimeter = boundary
    actual_image = image
    perimeter = [(b, a) for a, b in perimeter]
    perimeter = np.array(perimeter, np.int32).reshape((-1, 1, 2))
    actual_image = cv2.polylines(actual_image, perimeter, 1, color, thickness)
    return actual_image

def draw_axis(actual_image, centroid, angle, lengths):  # draws the axis's for classified objects
    xc, yc = centroid
    len1, len2 = lengths
    actual_image = cv2.line(actual_image, (yc, xc),
                            (yc + int(len1 * math.sin(angle)), xc + int(len1 * math.cos(angle))),
                            (0, 0, 0), 1)
    actual_image = cv2.line(actual_image, (yc, xc),
                            (yc - int(len1 * math.sin(angle)), xc - int(len1 * math.cos(angle))),
                            (0, 0, 0), 1)

    new_angle = math.pi - ((math.pi / 2) + angle)  # minor angle
    actual_image = cv2.line(actual_image, (yc, xc), (
        yc + int(len2 * math.sin(new_angle)), xc - int(len2 * math.cos(new_angle))), (0, 0, 0), 1)
    actual_image = cv2.line(actual_image, (yc, xc), (
        yc - int(len2 * math.sin(new_angle)), xc + int(len2 * math.cos(new_angle))), (0, 0, 0), 1)
    return actual_image


def classification(image, actual_image):  # object classified based on certain properties such as area and eccentricity
    visited = set()
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            if image[i][j] != 0:
                label = image[i][j]
                if label not in visited:
                    visited.add(label)
                    value = computations(image, label)
                    area, angle, eigens, eccentricity, centroid = value
                    eig1, eig2 = eigens[0], eigens[1]
                    len1, len2 = math.sqrt(eig1), math.sqrt(eig2)
                    xc, yc = int(centroid[0]),int(centroid[1])
                    centroid = (xc,yc)
                    lengths = (len1,len2)
                    if eccentricity > 0.5 and area > 3000:  # then it is banana
                        # banana
                        perimeter = sorted(wall_following(image, label, (i, j)))
                        actual_image = draw_axis(actual_image, centroid, angle,lengths)
                        actual_image = draw_perimeter(actual_image, perimeter, (0,255,255))

    #
                    elif area > 5000: # orange

                        perimeter = sorted(wall_following(image, label, (i, j)))
                        actual_image = draw_axis(actual_image, centroid, angle, lengths)
                        actual_image = draw_perimeter(actual_image, perimeter, (0,165,255))


                    elif area > 2500 and eccentricity < 0.5:  # apples

                        perimeter = sorted(wall_following(image, label, (i, j)))
                        actual_image = draw_axis(actual_image, centroid, angle, lengths)
                        actual_image = draw_perimeter(actual_image, perimeter, (0,0,255))

                    else:

                        perimeter = sorted(wall_following(image, label, (i, j)))
                        actual_image = draw_perimeter(actual_image, perimeter, (203,192,255))

    cv2.imshow("classified", actual_image)
    cv2.waitKey(0)
    cv2.destroyWindow("classified")


# filename = '/Users/vivekkumarmaheshwari/Downloads/fruit1.bmp'
# actual_image = cv2.imread(filename, 1)  # actual image without any processing
# thresholded_image = display_and_return_double_threshold(filename)
# cleaned_image = clean_image(thresholded_image)
# connected_components_image = connected_components_using_flood_fill(cleaned_image)  # (output, num_components)
# connect_image = connected_components_image[0]
#
# stem_removed_image = remove_stem(cleaned_image)
# banana_regions = list(find_banana_intensities(connect_image)[0])
# centroids = find_banana_intensities(connect_image)[1]
# image = final_components(connect_image, banana_regions, centroids)  # final components
# classification(image, actual_image)