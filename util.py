import const

import cv2

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

print('util.py module loaded')
