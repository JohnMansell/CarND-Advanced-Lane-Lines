import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt

objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

obj_points = []
img_points = []

images = glob.glob('./camera_cal/calibration*.jpg')

for idx, frame in enumerate(images):
    img = cv2.imread(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret is True:
        print('working on ', frame)
        obj_points.append(objp)
        img_points.append(corners)

        img_corners = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        write_name = './post_calibration/corners_found'+str(idx +1)+'.jpg'
        cv2.imwrite(write_name, img)

        # # Show Images
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        # ax1.set_title(' Original')
        # ax1.imshow(img)
        #
        # ax2.set_title('Corners')
        # ax2.imshow(img_corners)
        #
        # plt.show()


img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

dist_pickle = {}
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist
pickle.dump( dist_pickle, open('./calibration_pickle.p', "wb" ))






