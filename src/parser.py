######################################################################
## This algorithm is created for: parsing the yaml file to undistort frames of the input video
## Iteration: 2
## Date: December 22nd 2022
## Author: @IvyZhang, for the Dartmouth Reality and Robotics Lab
######################################################################


# import packages
import numpy as np
import yaml 
from yaml.loader import SafeLoader
import os as os

import cv2 as cv

# PART ONE: yaml parser
# Read the yaml file
# a) Parse distortion coefficients and output a table/dictionary for the values
# b) Parse the intrinsic coefficients and output a 3 x 3 matrix using numpy

parent = os.path.dirname(os.getcwd())
print(parent)
filePath = parent+'/RoboticsLab/gitHub_VO/Parameters/camera_calibration.yaml'
print(filePath)
with open(filePath) as camCalib:
    data = yaml.load(camCalib, Loader=SafeLoader)

#get the distortion coefficients into an array
distortion_coefficients = data["distortion_coeffs"]
dist_coef_arr = distortion_coefficients[0]
dist_coef_arr = np.array(dist_coef_arr).reshape(1, 5)
print(dist_coef_arr)

#compute the 3x3 matrix
intrinsic_coefficients = data["intrinsic_coeffs"]
int_coef_arr = intrinsic_coefficients[0]
print(int_coef_arr)
int_coeff_mtx = np.array(int_coef_arr)
int_coeff_mtx = int_coeff_mtx.reshape(3,3)
print(int_coeff_mtx)


# PART TWO: UNDISTORT INPUT VIDEO FRAMES
# for each frame do:
#      document the height and width and store in an array
#      cv.getOptimalNewCameraMatrix: pass in the distortion and intrinsic coefficients
#      use cv.undistort()
#      (convert to grayscale if haven't)
#      crop the image
#      save the image


all_frames = []


video = cv.VideoCapture('/Users/ivyzhang/PycharmProjects/RoboticsLab/build_from_robot_view_deepend.mp4')

width = video.get(cv.CAP_PROP_FRAME_WIDTH )
height = video.get(cv.CAP_PROP_FRAME_HEIGHT )
fps = video.get(cv.CAP_PROP_FPS)

out = cv.VideoWriter('parser_two_video.avi', cv.VideoWriter_fourcc(*'MP42'), fps, (width, height))

length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
print("Total number of frames: ", length)
count = 1

while (count<length):
    ret, frame = video.read()
    # Check the 'ret' (return value) to see if we have read all the frames in the video to exit the loop
    
    if not ret:
        print('Processed all frames')
        video.release()
        break

    h, w = frame.shape[:2]
    newCameraMtx, roi = cv.getOptimalNewCameraMatrix(int_coeff_mtx, dist_coef_arr, (w, h), 1, (w, h))

    # undistort
    dst = cv.undistort(frame, int_coeff_mtx, dist_coef_arr, None, newCameraMtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    # /Users/ivyzhang/PycharmProjects/RoboticsLab/gitHub_VO/results_one
    image_frame = cv.imwrite('/Users/ivyzhang/PycharmProjects/RoboticsLab/gitHub_VO/results_one/frame%d.jpg'%count, dst)
    out.write(image_frame)
    
    # #### This part will feed the new frame into a resulting video
    # image = cv.imread(image_frame)
    # height, width, layers = image.shape
    # size = (width, height)
    # all_frames.append(image)
    # increment
    count+=1

out.release()
