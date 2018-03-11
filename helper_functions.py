'''---------------------------------------------------
        Reviewing Steps
            - Camera Calibration
            - Distortion Correction
            - Color and Gradient Threshold
            - Perspective Transform
-----------------------------------------------------'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from tracker import tracker
from scipy.signal import convolve2d

'''================================================
        Hyper Parameters
==================================================='''
smin = 3
smax = 255
bmin = 0
bmax = 209
dmin = 0.1
dmax = 0.9
mmin = 0
mmax = 300
d_kernal = 13
m_kernal = 5
picture = 5

# Read in the Saved ObjPoints and ImgPoints
dist_pickle = pickle.load( open( "calibration_pickle.p", "rb"))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

'''===================================
        Camera Calibration
======================================'''

'''-----------------------------------
    Finding Chess Board Corners
------------------------------------'''
def find_chessboard_corners(img, nx=9, ny=6):
    ret, corners = cv2.findChessboardCorners(img, (nx, ny), None)

    if ret == True:
        return corners
    else:
        return False


'''-----------------------------------
    Calibration Undistort
------------------------------------'''
# def cal_undistort(img, objpoints, imgpoints):
#     # Use cv2.calibrate Camera and cv2.undistort()
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#     undist = cv2.undistort(img, mtx, dist, None, mtx)
#     return undist


'''================================================
        Function Definitions
==================================================='''
'''----------------------------
        Color Threshold
-------------------------------'''
def color_threshold(image, sthresh=(smin, smax), bthresh=(bmin, bmax)):
    '''----------------
        Expects image to be RGB
    --------------------'''

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    b_channel = lab[:, :, 2]
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= bthresh[0]) & (b_channel <= bthresh[1])] = 1

    color_binary = np.zeros_like(b_channel)
    color_binary[(s_binary == 1) & (b_binary == 1)] = 1

    return color_binary

'''----------------------------
        ABS Sobel
-------------------------------'''
# def abs_sobel_thresh(img, orient="x", thresh=(0, 255)):
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     if orient is 'x':
#         abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
#     if orient is 'y':
#         abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
#
#     scaled_sobel = np.uint8(255*abs_sobel / np.max(abs_sobel))
#     binary_output = np.zeros_like(scaled_sobel)
#
#     binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
#     return binary_output

'''----------------------------
        Mag Threshold
-------------------------------'''
def mag_threshold(image, sobel_kernel=m_kernal, mag_thresh = (mmin, mmax)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag) / 255
    binary_output = np.zeros_like(gradmag)

    binary_output[(gradmag >= mag_thresh[0])& (gradmag <= mag_thresh[1])] = 1
    return binary_output


'''--------------------------------------
        Directional Gradient Threshold
----------------------------------------'''
def dir_threshold(img, sobel_kernel=d_kernal, thresh=(dmin, dmax)):
    # GrayScale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Gradient in X and Y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # ABS of X and Y gradient
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # Direction of Gradient
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)

    # Binary Mask
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    return binary_output

'''----------------------------
        Window Mask
-------------------------------'''
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level+1)*height):int(img_ref.shape[0] - level*height),
            max(0, int(center-width)): min(int(center + width),  img_ref.shape[1])] = 1
    return output

'''----------------------------
        Show Images
-------------------------------'''
def show(img, img2=None, img3=None, title=None):

    # 1 Image
    if (img2 is None):
        plt.imshow(img, cmap='gray')

    # 2 Images
    elif img3 is None:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(img, cmap='gray')
        ax2.imshow(img2, cmap='gray')

    else:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
        ax1.imshow(img, cmap='gray')
        ax2.imshow(img2, cmap='gray')
        ax3.imshow(img3, cmap='gray')

    if(title is not None):
        plt.savefig(title, bbox_inches='tight')

    plt.show()


'''----------------------------
        Get Transform
-------------------------------'''
def get_transform(img):
    x_bottom = 1136
    x_top = 120
    depth = 273
    hood_depth = 30
    dst_offset = 400
    cal1_offset = 27
    cal2_offset = 30

    img_size = (img.shape[1], img.shape[0])

    # src = (x1, y1) , (x2, y2), (x3, y3), (x4, y4)
    x1 = int((img_size[0] - x_top) / 2)
    x2 = int((img_size[0] + x_top) / 2)

    y1 = y2 = int((img_size[1] - depth))

    x3 = int((img_size[0] - x_bottom) / 2)
    x4 = int((img_size[0] + x_bottom) / 2)

    y3 = y4 = (img_size[1] - hood_depth)

    # dst = (j1, k1), (j2, k2), (j3, k3), (j4, k4)
    j1 = j3 = (img_size[0] / 2) - dst_offset
    j2 = j4 = (img_size[0] / 2) + dst_offset

    k1 = k2 = 0
    k3 = k4 = img_size[1]

    src = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    dst = np.float32([[j1, k1], [j2, k2], [j3, k3], [j4, k4]])

    # Perspective Transform -- Matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return (M, Minv)

'''----------------------------
        Warp Image
-------------------------------'''

def warp_image(img, mtx):
    img_size = (img.shape[1], img.shape[0])

    # Perspective Transform
    warped = cv2.warpPerspective(img, mtx, img_size, flags=cv2.INTER_LINEAR)

    return warped

'''----------------------------
        Create Binary
-------------------------------'''
def create_binary(warped):

    # Test for roads that are too bright
    hsv = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    brightness = np.average(v)

    if brightness > 150:
        black = np.zeros_like(warped)
        warped = cv2.addWeighted(warped, 0.5, black, 0.5, 0.0)

    color_binary = color_threshold(warped, (smin, smax), (bmin, bmax))

    dir_binary = dir_threshold(warped, d_kernal, (dmin, dmax))

    mag_binary = mag_threshold(warped, m_kernal, (mmin, mmax))

    output = np.zeros_like(dir_binary)
    output[(color_binary == 1) & (dir_binary == 1) & (mag_binary == 0)] = 1

    return output

'''----------------------------
        
-------------------------------'''

'''------------------
        Tracker
---------------------'''
# Define Variables for Tracker
window_width = 20
window_height = 80
red = [255, 0, 0]
blue = [0, 0, 255]
green = [0, 100, 0]

myXM = 4 / 384
myYM = 10 / 720

# Video or Pictures
# For video -- set mySmooth_factor = 10
# For pictures -- set mySmoothFactor = 0

# Set up the overall class to do all the tracking
curve_centers = tracker(myWindow_width=window_width,
                        myWindow_height=window_height,
                        myMargin=20,
                        myYM= myYM,
                        myXM = myXM,
                        mySmooth_factor=10)


'''----------------------------
        Process Image
-------------------------------'''
def process_image(img):

    # Res Y Values
    yvals = range(0, img.shape[0])
    res_yvals = np.arange(img.shape[0] - (window_height / 2), 0, -window_height)
    left_base = 0
    right_base = 0

    # Undistort
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    # Perspective Transform
    warped_1 = warp_image(undistorted, M)

    # Create Binary
    warped = create_binary(warped_1)
    road = np.zeros_like(img)

    # Pattern Test
    ret, left, right = pattern_fit(warped)


    if ret is True:

        y1 = 0
        y2 = warped.shape[0]

        # Draw Binary
        road[(warped == 1)] = [255, 255, 255]

        # Left Lane
        x1 = left
        x2 = left
        cv2.line(road, (x1, y1), (x2, y2), color=red, thickness=30)

        # Right Lane
        r1 = right
        r2 = right
        cv2.line(road, (r1, y1), (r2, y2), color=blue, thickness=30)

        # Create p in p of road
        small = cv2.resize(road, (0, 0), fx=0.4, fy=0.4)

        # Lane Fill
        inset = 15
        x1 = x1 + inset
        x2 = x2 + inset
        r1 = r1 - inset
        r2 = r2 - inset
        poly_pts = np.array([[x1, y1], [x2, y2], [r2, y2], [r1, y1]], np.int32)
        pts = poly_pts.reshape((-1, 1, 2))
        cv2.fillPoly(road, [pts], green)

        left_base = left
        right_base = right
        curverad = 0.0


    else:
        window_centroids = curve_centers.find_window_centroid(warped)

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # points used to find the left and right lanes
        rightx = []
        leftx = []
        middle_x = []

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)

            # Add center value found in frame to the list of lane points per left, right
            leftx.append(window_centroids[level][0])
            rightx.append(window_centroids[level][1])
            middle_x.append(window_centroids[level][1] - window_centroids[level][0])

            l_points[(l_points == 255) | (l_mask == 1)] = 255
            r_points[(r_points == 255) | (r_mask == 1)] = 255


        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)
        warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)

        result = cv2.addWeighted(template, 1.0, warpage, 1.0, 0.0)
        result[(warped == 1) ] = [255, 255, 255]

        small = cv2.resize(result, (0,0), fx=0.4, fy=0.4)

        # fit the lane boundaries to the left, right center positions found
        left_base = leftx[0]
        right_base = rightx[0]


        left_fit = np.polyfit(res_yvals, leftx, 2)
        left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
        left_fitx = np.array(left_fitx, np.int32)

        right_fit = np.polyfit(res_yvals, rightx, 2)
        right_fitx = right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2]
        right_fitx = np.array(right_fitx, np.int32)

        middle_fit = np.polyfit(res_yvals, middle_x, 2)
        middle_fit_x = middle_fit[0] * yvals * yvals + middle_fit[1] * yvals + middle_fit[2]
        middle_fit_x = np.array(middle_fit_x, np.int32)

        left_lane = np.array(
            list(zip(np.concatenate((left_fitx - window_width / 2, left_fitx[::-1] + window_width / 2), axis=0),
                     np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
        right_lane = np.array(
            list(zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis=0),
                     np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
        middle_marker = np.array(
            list(zip(np.concatenate((left_fitx + window_width / 2, right_fitx[::-1] - window_width / 2), axis=0),
                     np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

        # road_bkg = np.zeros_like(img)
        cv2.fillPoly(road, [left_lane], color=red)
        cv2.fillPoly(road, [right_lane], color=blue)
        cv2.fillPoly(road, [middle_marker], color=green)

        ym_per_pix = curve_centers.ym_per_pixel
        xm_per_pix = curve_centers.xm_per_pixel

        curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix,
                                  np.array(rightx, np.float32) * xm_per_pix,
                                  2)
        curverad = ((1 + (2 * curve_fit_cr[0] * yvals[-1] * ym_per_pix + curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * curve_fit_cr[0])

    road_warped = warp_image(road, Minv)

    result = cv2.addWeighted(img, 1.0, road_warped, 0.5, 0.0)

    result[:small.shape[0], :small.shape[1]] = small

    # Lane Center
    lane_center = left_base + (right_base - left_base) // 2
    midpoint = road_warped.shape[1] // 2
    difference = (midpoint - lane_center) * myXM
    line_length = 50
    line_height = 680

    # Center Grid Line
    x1 = int(left_base + line_length)
    x2 = int(right_base - line_length)
    y1 = y2 = int(line_height)

    # Lane Center Marker -- Line
    j1 = j2 = int(lane_center)
    k1 = int(line_height - 30)
    k2 = int(line_height + 30)

    cv2.line(result, (x1, y1), (x2, y2), red, thickness=4)
    cv2.line(result, (j1, k1), (j2, k2), red, thickness=4)

    cv2.circle(result, (midpoint, line_height), radius=25, color=blue, thickness=3)

    cv2.putText(result,
                "Car is " + str(round(difference, 2)) + ' m off center',
                (700, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)




    #
    # # Draw the text showing curvature, offset and speed
    cv2.putText(result,
                'Radius of Curvature = ' + str(round(curverad, 1)) + '(m)',
                (700, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    # cv2.putText(result,
    #             'Lane Center = ' + str(camera_center),
    #             (50, 100),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             1, (255, 255, 255), 2)

    return result

# Perspective Matricies
img = cv2.imread('./test_images/test1.jpg')
M, Minv = get_transform(img)

'''-----------------------------
    Straight Line Template
--------------------------------'''

# Import Image
img = cv2.imread('test_images/straight_lines1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Process Image
warped = warp_image(img, M)
binary = create_binary(warped)

# Dimensions
img_height = binary.shape[0]
img_width = binary.shape[1]

# Histogram
midpoint = (img_width // 2)
histogram = np.sum(binary, axis=0)
left_lane = np.argmax(histogram[:midpoint])
right_lane = np.argmax(histogram[midpoint:]) + midpoint
spread = right_lane - left_lane

print(spread)

# Template
window_width = 50
s_template = np.zeros((img_height, img_width))
s_template[:, left_lane - window_width:left_lane + window_width] = 1
s_template[:, right_lane - window_width:right_lane + window_width] = 1

# min max
offset = window_width * 2
l_min = left_lane - offset
l_max = left_lane + offset

r_min = right_lane - offset
r_max = right_lane + offset



'''-----------------------------
    Pattern Fit
--------------------------------'''
def pattern_fit(img):

    ''' Expects Binary Image'''
    test = np.zeros_like(img)
    test[(s_template == 1) & (img == 1)] = 1
    test[(s_template == 0) & (img == 1)] = -1

    straight_signal = np.sum(test)

    if straight_signal > 500:

        # histogram = np.sum(img, axis=0)
        # left_lane = np.argmax(histogram[l_min : l_max]) + l_min
        # right_lane = np.argmax(histogram[r_min : r_max]) + r_min

        return (True, left_lane, right_lane)

    else:
        return (False, 0, 0)





































































