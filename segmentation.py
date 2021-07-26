import util

import cv2
import numpy as np

# Simple thresholding to convert RGB image into binary image
def threshold_image(image, file_name = None):
	grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	ret, thresholded_image = cv2.threshold(grey_image, 127, 255, 0)
	if file_name != None:
		util.write_image(thresholded_image, file_name)
	return thresholded_image

# Use the Canny edge detector in open-cv to convert RGB image into binary image
def canny_edge_image(image, file_name = None):
	edges = cv2.Canny(image,100,200)
	if file_name != None:
		util.write_image(edges, file_name)
	return edges

# Adapted from https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
def find_contours(image):
	binary_image = canny_edge_image(image) #pick between canny_edge_image and threshold_image
	contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours

def draw_contours(image, contours, color, file_name = None):
	if file_name == None: return False
	image_with_contours = np.copy(image)
	image_with_contours = cv2.drawContours(image_with_contours, contours, -1, color, 2)
	util.write_image(image_with_contours, file_name)
	return True

def find_bounding_boxes(contours, padding=1):
	bounding_boxes = []
	for contour in contours:
		x_coords = []
		y_coords = []
		for point in contour:
			x_coord = point[0][0]
			y_coord = point[0][1]
			x_coords.append(x_coord)
			y_coords.append(y_coord)
		min_x = min(x_coords) - padding
		max_x = max(x_coords) + padding
		min_y = min(y_coords) - padding
		max_y = max(y_coords) + padding
		bounding_boxes.append([min_x, max_x, min_y, max_y]) #consider making this an object?

	return np.asarray(bounding_boxes)

def draw_bounding_boxes(image, bounding_boxes, border_color, border_width=1, file_name = None):
	if file_name == None: return False
	image_with_boxes = np.copy(image)
	for bounding_box in bounding_boxes:
		min_x, max_x, min_y, max_y = bounding_box
		image_with_boxes = cv2.rectangle(image_with_boxes, (min_x, min_y), (max_x, max_y), border_color, border_width)
	util.write_image(image_with_boxes, file_name)
	return True

print('segmentation.py module loaded')
