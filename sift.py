import const

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Adapted from Prof Achuta Kadami, Prof Stefano Soatto (CS 188: Introduction to Computer Vision)
def save_color_img(x, file_name):
	plt.figure(figsize=(20,20))
	plt.imshow(x/x.max(),vmin=0,vmax=1)
	plt.savefig(f'{const.DATA_OUTPUT_DIRECTORY}/{file_name}.pdf', format='pdf', bbox_inches='tight')
	return

# Adapted from Prof Achuta Kadami, Prof Stefano Soatto (CS 188: Introduction to Computer Vision)
def plot_correspondences(image1, image2, correspondences, color, file_name=None):
	image = np.concatenate((image1, image2), axis=1)
	for correspondence in correspondences:
		point1, point2 = correspondence
		point1 = (int(round(point1[0])), int(round(point1[1])))
		point2 = (int(round(point2[0])), int(round(point2[1])))
		cv2.circle(image, point1, 10, color, 2, cv2.LINE_AA)
		cv2.circle(image, tuple([point2[0] + image1.shape[1], point2[1]]), 10, 
					color, 2, cv2.LINE_AA)
		cv2.line(image, point1, tuple([point2[0] + image1.shape[1], point2[1]]), 
					color, 2)
	if file_name != None:
		save_color_img(image, file_name)
	return

# Adapted from https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
def grey_sift(image, num_features, file_name=None):
	grey = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	sift = cv2.SIFT_create(num_features)
	key_points = sift.detect(grey,None)
	img = cv2.drawKeypoints(grey, key_points, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	if file_name != None:
		cv2.imwrite(f'{file_name}.jpg',img)

def run_sift(image, num_features):
	grey = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	sift = cv2.SIFT_create(num_features)
	key_points = sift.detect(grey,None)
	key_points,descriptors = sift.compute(grey,key_points)
	return key_points, descriptors

def find_sift_correspondences(kp1, des1, kp2, des2, ratio):
	possible_correspondences = []
	num_kp1 = len(kp1)
	num_kp2 = len(kp2)
	for i in range(0, num_kp1):
		kp = kp1[i]
		kp_des = des1[i]
		dist_to_candidates = {}
		for j in range(0, num_kp2):
			candidate_kp = kp2[j]
			candidate_kp_des = des2[j]
			distance = np.sqrt(np.sum(np.square(kp_des - candidate_kp_des)))
			dist_to_candidates[distance] = candidate_kp
      
		sorted_distances = sorted(dist_to_candidates)
		lowest_dist = sorted_distances[0]
		second_lowest_dist = sorted_distances[1]
		if(lowest_dist < (ratio * second_lowest_dist)):
			possible_correspondences.append((kp.pt, dist_to_candidates[lowest_dist].pt))

	return possible_correspondences

print('sift.py module loaded')
