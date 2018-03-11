import numpy as np
import matplotlib.pyplot as plt
import cv2

class tracker():

    def __init__(self, myWindow_width, myWindow_height, myMargin, myYM = 1, myXM = 1, mySmooth_factor = 10):
        # List that stores all the past (left, right) center set values used for smoothing the output
        self.recent_centers = []

        # the window pixel width of the center values, used to count pixels inside center windows to determine curve values
        self.window_width = myWindow_width
        self.window_height = myWindow_height
        self.margin = myMargin
        self.ym_per_pixel = myYM
        self.xm_per_pixel = myXM
        self.smooth_factor = mySmooth_factor


    def find_window_centroid(self, warped):
        '''==============================================
                Convolutions
        =================================================='''

        '''---------------------------------
                Initialize Variables
        ------------------------------------'''
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin
        minpix = 5000


        window_centroids = [] # Store the (left, right) window centroid positions per level
        window = np.ones(window_width) # create our window template that we will use for convolutions

        # Find Histogram of Image
        histogram = np.sum(warped[: -(warped.shape[0] // 4), :], axis=0)

        # Left / Right base
        abs_min_left = 300
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[abs_min_left:midpoint]) + abs_min_left
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        spread = 570

        # # Sum bottom quarter of image to get slice, could use a different ratio
        l_center = 392
        r_center = 955
        #
        # # Add what we found for the first layer
        # window_centroids.append((l_center, r_center))


        # Go through each layer looking for max pixel locations
        n_levels = int(warped.shape[0] / window_height)
        image_height = warped.shape[0]
        image_width = warped.shape[1]
        offset = window_width / 2
        shift = []
        shift.append(0)

        # Convolution Template
        template_width = (window_width * 2) + spread
        template = np.zeros(template_width, dtype='int')
        template[ :window_width] = 1
        template[spread + window_width : ] = 1


        for level in range(0, n_levels):

            # Convolve the window into the vertical slice of the image
            # (x1, y1) (x1, y2)
            y1 = image_height - (window_height * (level +1))
            y2 = image_height - (window_height * level)
            image_layer = np.sum(warped[y1:y2, :], axis=0)
            lane_conv = np.convolve(template, image_layer, mode='valid')

            l_min_index = int(max(l_center - margin - offset, 0))
            l_max_index = int(l_center + margin + offset)

            try:
                new_l_center = np.argmax(lane_conv[l_min_index : l_max_index]) + l_min_index

            except:
                new_l_center = l_center + np.average(shift)

            shift.append(new_l_center - l_center)

            l_center = new_l_center
            r_center = l_center + spread

            window_centroids.append((l_center, r_center))

        # Use for Video
        if self.smooth_factor > 0:
            self.recent_centers.append(window_centroids)

            # Return Averaged values of the line centers
            #   -- helps to keep the markers from jumping around too much
            return np.average(self.recent_centers[-self.smooth_factor:], axis = 0)

        # Use for Pictures
        else:
            return window_centroids








