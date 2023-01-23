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
import transformations as transf
import matplotlib.pyplot as plt

# this is a random comment

######################################################################
# PART ONE: yaml parser
# Read the yaml file
# a) Parse distortion coefficients and output a table/dictionary for the values
# b) Parse the intrinsic coefficients and output a 3 x 3 matrix using numpy
# Last updated: January 17th 2023
######################################################################

filePath = '/Users/ivyzhang/Documents/droplet_visual_odometry/Parameters/camera_calibration.yaml'
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

######################################################################
# PART TWO: UNDISTORT INPUT VIDEO FRAMES + Find the changes in points detected
# a) Starting from frame2, draw matches between current and previous frames
# b) Compute the essential matrix between frames
# c) Using the essential matrix, compute the relative position and translation
# Last updated: January 17th 2023
######################################################################

video = cv.VideoCapture('/Users/ivyzhang/PycharmProjects/RoboticsLab/build_from_robot_view_deepend.mp4')

width = video.get(cv.CAP_PROP_FRAME_WIDTH )
height = video.get(cv.CAP_PROP_FRAME_HEIGHT )
fps = video.get(cv.CAP_PROP_FPS)

out = cv.VideoWriter('parser_two_video.avi', cv.VideoWriter_fourcc(*'MP42'), fps, (int(width), int(height)))

length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
print("Total number of frames: ", length)
count = 1

all_frames = [] # array to save all video frames; will be sorted in order

orb_feature_detector = cv.ORB_create()

previous_image = None
previous_key_points = None
previous_descriptors = None
current_frame = None


### UTILITY FUNCTION FOR TRANSFORMTION MATRIX ###
def make_transform_mat(translation, euler):
    rotation = transf.euler_matrix(*euler, 'sxyz')    
    translation = transf.translation_matrix(translation)    
    return translation.dot(rotation) 

previous_transformation_mat = make_transform_mat(translation = [0,0,0], euler=[0,0,0])
#current_transformation_mat = None
robot_current_position = make_transform_mat(translation=[0,0,0], euler=[0,0,0]) 

robot_position_list = []

### FRAME ANALYSIS
for count in range(length):
    ret, frame = video.read()
    print("Frame {} of {} ".format(count, length))
    # Check the 'ret' (return value) to see if we have read all the frames in the video to exit the loop
    

    # if not ret:
    #     print('Processed all frames')
    #     video.release()
    #     break

    image_height, image_width = frame.shape[:2]
    new_camera_matrix, _ = cv.getOptimalNewCameraMatrix(int_coeff_mtx, dist_coef_arr, (image_width, image_height), 1, (image_width, image_height))

    # undistort
    current_image = cv.undistort(frame, int_coeff_mtx, dist_coef_arr, None, new_camera_matrix)
    current_key_points = orb_feature_detector.detect(current_image,None)
    # find the keypoints and descriptors with ORB
    current_key_points, current_descriptors = orb_feature_detector.compute(current_image,current_key_points)
    current_image_with_keypoints_drawn = cv.drawKeypoints(current_image, current_key_points, None, color=(0,255,0), flags=0)

    ### STARTING FROM FRAME NUMBER 2
    if previous_image is not None:

        ######## DRAW MATCHES ########
        # do matching here
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(previous_descriptors,current_descriptors)
        img3 = cv.drawMatches(previous_image,current_key_points,current_image_with_keypoints_drawn,current_key_points,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow('pattern',img3),cv.waitKey(5)

        ######## VELOCITY ESTIMATION ########
        # convert previous_key_points and current_key_points into floating point arrays
        array_previous_key_points = cv.KeyPoint_convert(previous_key_points)
        array_current_key_points = cv.KeyPoint_convert(previous_key_points)

        # get the essential matrix
        essMatrix, mask = cv.findEssentialMat(points1=array_current_key_points, points2=array_previous_key_points, cameraMatrix=int_coeff_mtx,method=cv.RANSAC,prob=0.999, threshold=1.0)
        # print(essMatrix)

        # compute the relative position using the essential matrix, key points etc useing cv.relativepose    
        points, relative_rotation, translation, mask = cv.recoverPose(E=essMatrix, points1=array_current_key_points, points2=array_previous_key_points)   
        translation = translation.transpose()[0]
        relative_rotation = np.array(relative_rotation)
       # decompose rotation matrix + find euler
        t=np.array([ 0,  0,  0])
        new_rotation_mat = np.empty((4,4))
        new_rotation_mat = np.vstack((np.hstack((relative_rotation, t[:, None])), [0,0,0,1]))

        # get euler angles
        euler = transf.euler_from_matrix(new_rotation_mat, 'rxyz')

        # compute the current transformation matrix
        euler = np.array(euler)
        
        position_update = make_transform_mat(translation=translation, euler=euler)
        robot_current_position = robot_current_position.dot(position_update)
        _, _, _, robot_current_translation, _ = transf.decompose_matrix(robot_current_position)
        robot_position_list.append(robot_current_translation)        


    previous_image = current_image_with_keypoints_drawn
    previous_key_points = current_key_points # same key points of PREVIOUS frame
    previous_descriptors = current_descriptors
   

out.release()

####### PLOT THE 3D SCATTER PLOT #######
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(0, len(robot_position_list)):
    array_list = np.array(robot_position_list[i].tolist())
    ax.scatter(array_list[0], array_list[1], array_list[2])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
