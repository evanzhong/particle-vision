import util

import cv2
import numpy as np

# Simple thresholding to convert RGB image into binary image
def threshold_image(image):
	grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	ret, thresholded_image = cv2.threshold(grey_image, 127, 255, cv2.THRESH_BINARY)
	return thresholded_image

def otsu_threshold_image(image, should_apply_gaussian = False):
	image_to_use = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	if should_apply_gaussian:
		image_to_use = cv2.GaussianBlur(image_to_use, (3,3), 0)

	ret, thresholded_image = cv2.threshold(image_to_use, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return thresholded_image

# Use the Canny edge detector in open-cv to convert RGB image into binary image
def canny_edge_image(image):
	edges = cv2.Canny(image,100,200)
	return edges

# Adapted from https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
def find_contours(image):
	binary_image = canny_edge_image(image) #pick between canny_edge_image and threshold_image
	contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours

def draw_contours(image, contours, color):
	image_with_contours = np.copy(image)
	image_with_contours = cv2.drawContours(image_with_contours, contours, -1, color, 2)
	return image_with_contours

def find_bounding_boxes(contours, padding=1):
	bounding_boxes = []
	for contour in contours:
		x, y, width, height = cv2.boundingRect(contour)
		if (x - padding) >= 0: x -= padding
		if (y - padding) >= 0: y -= padding
		width += padding #should this also be gated?
		height += padding
		bounding_boxes.append((x, y, width, height))
	return bounding_boxes

def draw_bounding_boxes(image, bounding_boxes, border_color, border_width=1):
	image_with_boxes = np.copy(image)
	for bounding_box in bounding_boxes:
		x, y, width, height = bounding_box
		image_with_boxes = cv2.rectangle(image_with_boxes, (x, y), (x + width, y + height), border_color, border_width)
	return image_with_boxes

def group_bounding_boxes(bounding_boxes, ratio, should_duplicate=False):
	if should_duplicate:
		boxes_to_use = bounding_boxes + bounding_boxes #see https://answers.opencv.org/question/204530/merge-overlapping-rectangles/
	else:
		boxes_to_use = bounding_boxes
	grouped_boxes, weights = cv2.groupRectangles(boxes_to_use, 1, ratio)
	return grouped_boxes

def merge_overlapping_bounding_boxes(bounding_boxes, debug_image = None):
	bounding_boxes_copy = list.copy(bounding_boxes)

	has_intersections = True
	repeat_counter = 0
	while has_intersections:
		has_intersections = False
		for i in range(len(bounding_boxes_copy)):
			box_i = bounding_boxes_copy[i]
			if(util.get_box_area(box_i) == 0): continue
			for j in range(len(bounding_boxes_copy)):
				box_j = bounding_boxes_copy[j]
				if (i == j or util.get_box_area(box_j) == 0): continue
				if util.has_intersection(box_i, box_j):
					has_intersections = True
					bounding_boxes_copy.append(util.union(box_i, box_j))
					bounding_boxes_copy[i] = (0, 0, 0, 0)
					bounding_boxes_copy[j] = (0, 0, 0, 0)
					if debug_image is not None:
						print(f'[DEBUG] Merging box_i: {box_i}, with  box_j: {box_j} to get {util.union(box_i, box_j)} (check img \'repeat_counter({repeat_counter}) i({i}) j({j})\')')
						util.write_image(
							image=draw_bounding_boxes(debug_image, util.remove_null_boxes(bounding_boxes_copy), (0,0,255), 1),
							file_output_name=f'repeat_counter({repeat_counter}) i({i}) j({j})'
						)
					break
		repeat_counter += 1

	return util.remove_null_boxes(bounding_boxes_copy)

def crop_image(image, box):
  x, y, width, height = box
  cropped_image = image[y:y+height, x:x+width]
  return cropped_image

def get_non_zero_pixel_area(image):
	binary_image = otsu_threshold_image(image)
	return cv2.countNonZero(binary_image)

print('segmentation.py module loaded')
