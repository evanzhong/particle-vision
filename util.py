import const

import cv2

def read_image(file_name):
  return cv2.imread(file_name)

def write_image(image, file_output_name):
  return cv2.imwrite(f'{const.DATA_OUTPUT_DIRECTORY}/{file_output_name}.{const.IMAGE_FILE_EXTENSION}', image)

print('util.py module loaded')
