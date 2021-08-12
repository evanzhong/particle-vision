import util

import random
import numpy as np
import cv2

# Adapted from Prof Achuta Kadami, Prof Stefano Soatto (CS 188: Introduction to Computer Vision)
def plot_correspondences(image1, image2, correspondences, color, file_name=None):
	if(file_name == None): return False
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
	util.write_image(image, file_name)
	return True

# Adapted from https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
def plot_sift_descriptors(image, num_features, file_name=None):
	if(file_name == None): return False
	key_points, descriptors = run_sift(image, num_features)
	image_with_keypoints = cv2.drawKeypoints(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),
																					 key_points,
																					 np.copy(image),
																					 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	util.write_image(image_with_keypoints, file_name)
	return True

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

def compute_homography(correspondences):
    A = np.empty((0,9))
    for correspondence in correspondences:
        pt_0 = correspondence[0]
        pt_1 = correspondence[1]
        x_0 = pt_0[0]
        y_0 = pt_0[1]
        x_1 = pt_1[0]
        y_1 = pt_1[1]
        block = np.asarray([
            [-x_0, -y_0, -1, 0, 0, 0, x_0*x_1, y_0*x_1, x_1],
            [0, 0, 0, -x_0, -y_0, -1, x_0*y_1, y_0*y_1, y_1]
        ])
        A = np.vstack((A, block))

    u, s, vt = np.linalg.svd(A)
    v = np.transpose(vt)
    H = np.array(v[:, -1]) #last column
    H = np.resize(H, (3,3))
    return H

def apply_homography(points, homography):
    transformed_points = []
    for point in points:
        x = point[0]
        y = point[1]
        homo_coord = np.resize(np.asarray([x, y, 1]), (3,1))

        new_coord = np.matmul(homography, homo_coord)
        x_prime = new_coord[0]/new_coord[2]
        y_prime = new_coord[1]/new_coord[2]
        transformed_points.append((x_prime[0], y_prime[0]))

    return transformed_points


def compute_inliers(homography, correspondences, threshold):
    outliers = []
    inliers = []
    for correspondence in correspondences:
        p0 = correspondence[0]
        p1 = correspondence[1]

        p0prime = apply_homography([p0], homography)[0]
        distance = np.sqrt((p1[0] - p0prime[0])**2 + (p1[1] - p0prime[1])**2)
        if (distance < threshold):
            inliers.append(correspondence)
        else:
            outliers.append(correspondence)

    return inliers,outliers

def ransac(correspondences, num_iterations, num_sampled_points, threshold):
    H = None
    best_inliers = []
    best_outliers = []
    max_inliers = 0

    for i in range(0, num_iterations):
        correspondence_subset = random.sample(correspondences, num_sampled_points)
        H_candidate = compute_homography(correspondence_subset)

        inliers, outliers = compute_inliers(H_candidate, correspondences, threshold)
        if(len(inliers) > max_inliers):
            max_inliers = len(inliers)
            H = H_candidate
            best_inliers = inliers
            best_outliers = outliers

    return H, best_inliers, best_outliers

print('sift.py module loaded')
