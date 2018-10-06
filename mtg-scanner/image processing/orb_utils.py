import cv2
import numpy as np
import cPickle as pickle

# https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python

def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])     
        ++i
        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

# Load the images
img1 =cv2.imread('test_img1.jpg')
img2 =cv2.imread('test_img2.jpg')

# Convert them to greyscale
grey_img1 =cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
grey_img2 =cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# SURF extraction
surf = cv2.SURF()
kp1, desc1 = surf.detect(grey_img1,None,useProvidedKeypoints = False)
kp2, desc2 = surf.detect(grey_img2,None,useProvidedKeypoints = False)

#Store and Retrieve keypoint features
temp_array = []
temp = pickle_keypoints(kp1, desc1)
temp_array.append(temp)
temp = pickle_keypoints(kp2, desc2)
temp_array.append(temp)
pickle.dump(temp_array, open("keypoints_database.p", "wb"))

#Retrieve Keypoint Features
keypoints_database = pickle.load( open( "keypoints_database.p", "rb" ) )
kp1, desc1 = unpickle_keypoints(keypoints_database[0])
kp1, desc1 = unpickle_keypoints(keypoints_database[1])