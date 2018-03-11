import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
from tracker import tracker

'''============================
        Set Up
==============================='''

# Read in the Saved ObjPoints and ImgPoints
dist_pickle = pickle.load( open( "calibration_pickle.p", "rb"))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

# Import Images
images = glob.glob('./test_images/test*.jpg')

'''================================================
        Function Definitions
==================================================='''
'''----------------------------
        Color Threshold
-------------------------------'''
def color_threshold(image, sthresh=(0, 255), vthresh=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1]) ] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1]) ] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output

'''----------------------------
        ABS Sobel
-------------------------------'''
def abs_sobel_thresh(img, orient="x", thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient is 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient is 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    scaled_sobel = np.uint8(255*abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)

    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

'''----------------------------
        Mag Threshold
-------------------------------'''
def mag_thresh(image, sobel_kernel = 3, mag_thresh = (0, 255)):
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
def dir_threshold(img, sobel_kernel=7, thresh=(0, 0.09)):
    # GrayScale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

'''============================
        Preprocess Images
==============================='''
for idx, fname in enumerate(images):

    # Import Images
    img = cv2.imread(fname)

    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Undistort
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    # Evaluate Thresholds
    preprocessImage = np.zeros_like(img[:, :, 0])
    gradx = abs_sobel_thresh(undistorted, orient='x', thresh=(12, 255))
    grady = abs_sobel_thresh(undistorted, orient='y', thresh=(25, 255))

    # Create Binary Image
    c_binary = color_threshold(undistorted, sthresh=(100, 255), vthresh=(50, 255))
    preprocessImage[((gradx ==1) & (grady == 1) | (c_binary == 1))] = 255

    # Perspective Transform -- Points
    img_size = (img.shape[1], img.shape[0])
    bot_width = 0.76    # percent of bottom trapizoid height
    mid_width = 0.12    # percent of middle trapizoid height
    height_pct = 0.68   # percent for trapizoid height
    bottom_trim = 0.935 # percent from top to bottom to avoid car hood

    # Perspective Transform -- Offest
    offset = img_size[0] * 0.25

    # Perspective Transform -- Source
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

    # Perspective Transform -- Destination
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])

    # Perspective Transform -- Prep
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Perspective Transform
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)

    # Tracker

    # Define Variables for Tracker
    window_width = 25
    window_height = 80

    # Set up the overall class to do all the tracking
    curve_centers = tracker(myWindow_width=     window_width,
                            myWindow_height=    window_height,
                            myMargin=           25,
                            myYM =              10 / 720,
                            myXM =              4 / 384,
                            mySmooth_factor=    15  )

    window_centroids = curve_centers.find_window_centroid(warped)

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # points used to find the left and right lanes
    rightx = []
    leftx = []

    # Go through each level and draw the windows
    for level in range(0, len(window_centroids)):

        # window_mask is a function to draw window areas
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)

        # Add center value found in frame to the list of lane points per left, right
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])

        l_points[(l_points == 255) | (l_mask == 1) ] = 255
        r_points[(r_points == 255) | (r_mask == 1) ] = 255

    # Draw the results
    template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)
    warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    # fit the lane boundaries to the left, right center positions found
    yvals = range(0, warped.shape[0])

    res_yvals = np.arange(warped.shape[0] - (window_height / 2), 0, -window_height)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0] * yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width / 2, left_fitx[::-1] + window_width / 2), axis=0),
                                  np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis=0),
                                   np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    middle_marker = np.array(list(zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis=0),
                                   np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)


    road = np.zeros_like(img)
    # road_bkg = np.zeros_like(img)
    cv2.fillPoly(road,[left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])

    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)

    result = cv2.addWeighted(img, 1.0, road_warped, 5.0, 0.0)

    ym_per_pix = curve_centers.ym_per_pixel
    xm_per_pix = curve_centers.xm_per_pixel

    curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(rightx, np.float32)*xm_per_pix, 2)
    curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])

    # Calculate the offset of the center line
    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - warped.shape[1] / 2)*xm_per_pix
    side_pos = 'left'
    if center_diff <=0:
        side_pos = 'right'

    # Draw the text showing curvature, offset and speed
    cv2.putText(result,
                'Radius of Curvature = ' + str(round(curverad, 3)) + '(m)',
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    cv2.putText(result,
                'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center',
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)


    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('Image')
    ax1.imshow(img)

    ax2.set_title('Road')
    ax2.imshow(result)

    plt.show()

    # Save image
    write_name = './test_images/tracked'+str(idx +1)+'.jpg'
    cv2.imwrite(write_name, result)

