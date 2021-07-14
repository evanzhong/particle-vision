import const

import cv2
# Adapted from https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
def find_contours(image):
	grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	ret, thresh = cv2.threshold(grey, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	image_with_contours = cv2.drawContours(image, contours, -1, (0,255,0), 3)
	cv2.imwrite(f'test.jpg', image_with_contours)
