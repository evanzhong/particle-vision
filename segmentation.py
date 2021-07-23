import const

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Simple thresholding to convert RGB image into binary image
def threshold_image(image, file_name = None):
	grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	ret, thresholded_image = cv2.threshold(grey_image, 127, 255, 0)
	if file_name != None:
		cv2.imwrite(f'{const.DATA_OUTPUT_DIRECTORY}/{file_name}.jpg', thresholded_image)
	return thresholded_image

# Use the Canny edge detector in open-cv to convert RGB image into binary image
def canny_edge_image(image, file_name = None):
	edges = cv2.Canny(image,100,200)
	if file_name != None:
		plt.figure(figsize=(20,20))
		plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		plt.savefig(f'{const.DATA_OUTPUT_DIRECTORY}/{file_name}.pdf', format='pdf', bbox_inches='tight')
	return edges

# Adapted from https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
def find_contours(image, file_name=None):
	binary_image = canny_edge_image(image) #pick between canny_edge_image and threshold_image
	contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	if file_name != None:
		image_with_contours = cv2.drawContours(image, contours, -1, (0,255,0), 3)
		cv2.imwrite(f'{const.DATA_OUTPUT_DIRECTORY}/{file_name}.jpg', image_with_contours)
	return contours

def get_bounding_boxes(contours, padding=1):
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

	return bounding_boxes

def draw_boudning_boxes(image, boudning_boxes, border_width=1):
	image_copy = np.copy(image)
	for boudning_box in boudning_boxes:
		min_x, max_x, min_y, max_y = boudning_box
		image_copy = cv2.rectangle(image_copy, (min_x, min_y), (max_x, max_y), (255, 0, 0), border_width)
	return image_copy

print('segmentation.py module loaded')
