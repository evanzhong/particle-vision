import util

import cv2
import numpy as np

# Simple thresholding to convert RGB image into binary image
def threshold_image(image):
	grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	ret, thresholded_image = cv2.threshold(grey_image, 127, 255, cv2.THRESH_BINARY)
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
		bounding_boxes.append([x, y, width, height])
	return np.asarray(bounding_boxes)

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

def box_area(box):
	return box[2] * box[3]

# Adapted from https://stackoverflow.com/questions/46260892/finding-the-union-of-multiple-overlapping-rectangles-opencv-python/57546435
def has_intersection(box_1,box_2):
	x = max(box_1[0], box_2[0])
	y = max(box_1[1], box_2[1])
	width = min(box_1[0]+box_1[2], box_2[0]+box_2[2]) - x
	height = min(box_1[1]+box_1[3], box_2[1]+box_2[3]) - y
	if width < 0 or height < 0: return False
	else: return box_area([x, y, width, height]) > 0

# Adapted from https://stackoverflow.com/questions/46260892/finding-the-union-of-multiple-overlapping-rectangles-opencv-python/57546435
def union(box_1,box_2):
  x = min(box_1[0], box_2[0])
  y = min(box_1[1], box_2[1])
  width = max(box_1[0]+box_1[2], box_2[0]+box_2[2]) - x
  height = max(box_1[1]+box_1[3], box_2[1]+box_2[3]) - y
  return [x, y, width, height]

def remove_null_boxes(boxes):
	mask = []
	for i in range(len(boxes)):
		if box_area(boxes[i]) == 0: mask.append(True)
		else: mask.append(False)
	return np.delete(boxes, mask, 0)

def merge_overlapping_bounding_boxes(bounding_boxes, debug_image = None):
	bounding_boxes_copy = np.copy(bounding_boxes)

	has_intersections = True
	repeat_counter = 0
	while has_intersections:
		has_intersections = False
		for i in range(len(bounding_boxes_copy)):
			box_i = bounding_boxes_copy[i]
			if(box_area(box_i) == 0): continue
			for j in range(len(bounding_boxes_copy)):
				box_j = bounding_boxes_copy[j]
				if (i == j or box_area(box_j) == 0): continue
				if has_intersection(box_i, box_j):
					has_intersections = True
					bounding_boxes_copy = np.append(bounding_boxes_copy, [union(box_i, box_j)], 0)
					bounding_boxes_copy[i] = [0, 0, 0, 0]
					bounding_boxes_copy[j] = [0, 0, 0, 0]
					if debug_image is not None:
						print(f'[DEBUG] Merging box_i: {box_i}, with  box_j: {box_j} to get {union(box_i, box_j)} (check img \'repeat_counter({repeat_counter}) i({i}) j({j})\')')
						util.write_image(
							image=draw_bounding_boxes(debug_image, remove_null_boxes(bounding_boxes_copy), (0,0,255), 1),
							file_output_name=f'repeat_counter({repeat_counter}) i({i}) j({j})'
						)
					break
		repeat_counter += 1

	return remove_null_boxes(bounding_boxes_copy)


print('segmentation.py module loaded')
