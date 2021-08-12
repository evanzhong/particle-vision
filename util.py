import const
from segmentation import crop_image

import cv2

def get_no_extension_filename(full_file_name):
	no_extension_filename = full_file_name.split('.')[0]
	no_extension_filename = no_extension_filename.replace('/', '_')
	return no_extension_filename

def read_image(file_name):
  return cv2.imread(file_name)

def read_all_images(file_names):
	images = [read_image(file_name) for file_name in file_names]
	return list(zip(images, file_names))

def write_image(image, file_output_name):
  return cv2.imwrite(f'{const.DATA_OUTPUT_DIRECTORY}/{file_output_name}.{const.IMAGE_FILE_EXTENSION}', image)

def get_readable_correspondence(correspondence):
  point_0, point_1 = correspondence
  return ((round(point_0[0]), round(point_0[1])), (round(point_1[0]), round(point_1[1])))

def get_box_area(box):
	x, y, width, height = box
	return width * height

# Adapted from https://stackoverflow.com/questions/46260892/finding-the-union-of-multiple-overlapping-rectangles-opencv-python/57546435
def has_intersection(box_1,box_2):
	x = max(box_1[0], box_2[0])
	y = max(box_1[1], box_2[1])
	width = min(box_1[0]+box_1[2], box_2[0]+box_2[2]) - x
	height = min(box_1[1]+box_1[3], box_2[1]+box_2[3]) - y
	if width < 0 or height < 0: return False
	else: return get_box_area((x, y, width, height)) > 0

# Adapted from https://stackoverflow.com/questions/46260892/finding-the-union-of-multiple-overlapping-rectangles-opencv-python/57546435
def union(box_1,box_2):
  x = min(box_1[0], box_2[0])
  y = min(box_1[1], box_2[1])
  width = max(box_1[0]+box_1[2], box_2[0]+box_2[2]) - x
  height = max(box_1[1]+box_1[3], box_2[1]+box_2[3]) - y
  return (x, y, width, height)

def remove_null_boxes(boxes):
	return [box for box in boxes if get_box_area(box) != 0]

def get_box_containing_point(point, boxes):
	point_x, point_y = point
	for box in boxes:
		box_x, box_y, box_width, box_height = box
		point_is_in_box = (
			point_x >= box_x and
			point_x <= box_x + box_width and
			point_y >= box_y and
			point_y <= box_y + box_height
		)
		if point_is_in_box:
			return box
	return None

def get_image_center(image, margins):
  image_shape = image.shape
  margin_top, margin_right, margin_bot, margin_left = margins
  max_y, max_x, channels = image_shape
  center_box = (margin_left, margin_top, max_x-(margin_right*2), max_y-(margin_bot*2))#TODO fix this bug, shouldn't be *2 should be -right-left
  cropped_image = crop_image(image=image, box=center_box)
  return cropped_image

print('util.py module loaded')
