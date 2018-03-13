import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
from tracker import tracker
import helper_functions

# Read in the Saved ObjPoints and ImgPoints
dist_pickle = pickle.load( open( "calibration_pickle.p", "rb"))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

# Import Images
images = glob.glob('./test_images/test*.jpg')


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

'''------------------------------
        Images
---------------------------------'''

# # Perspective Matricies
# img = cv2.imread('./test_images/test1.jpg')
# M, Minv = helper_functions.get_transform(img)
#
# images = glob.glob('test_images/*.jpg')#
#
# for image in images:
#     img = cv2.imread(image)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     warped = helper_functions.warp_image(img, M)
#
#     result = helper_functions.process_image(img)
#     binary = helper_functions.create_binary(warped)
#     helper_functions.pattern_fit(binary)
#     helper_functions.show(img, result, title="FinalResult")

'''------------------------------
        Video
---------------------------------'''
from moviepy.editor import VideoFileClip

Output_video = 'output_project_video.mp4'
Input_video = 'input_videos/project_video.mp4'

clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(helper_functions.process_image)
video_clip.write_videofile(Output_video, audio=False)