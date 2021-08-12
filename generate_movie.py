import const
import util
import segmentation as seg
import cv2

# Update the global variables below to whatever movie you want
ROIS = [(1981, (9, (828, 725, 71, 56))), (1935, (10, (828, 724, 71, 56))), (1976, (11, (829, 724, 71, 56))), (1998, (12, (828, 723, 72, 56))), (1975, (13, (828, 724, 72, 55))), (1931, (14, (828, 723, 72, 56))), (1976, (15, (828, 723, 72, 56))), (1940, (16, (828, 723, 72, 56))), (1944, (17, (828, 723, 72, 56))), (1943, (18, (828, 722, 72, 57))), (1942, (19, (827, 722, 73, 57))), (1928, (20, (828, 723, 61, 69))), (1923, (21, (828, 724, 61, 68))), (1942, (22, (829, 724, 61, 69))), (1973, (23, (829, 718, 64, 72))), (1924, (24, (829, 723, 62, 68))), (1913, (25, (829, 718, 62, 72))), (1939, (26, (829, 723, 61, 68))), (1966, (27, (829, 716, 61, 75))), (1927, (28, (828, 717, 62, 74))), (1912, (29, (829, 723, 61, 68))), (1957, (30, (829, 715, 60, 77))), (1932, (31, (829, 718, 61, 74))), (767, (32, (829, 723, 49, 29))), (1937, (33, (829, 717, 61, 74))), (1936, (34, (829, 723, 61, 69))), (1952, (35, (829, 718, 61, 75))), (1955, (36, (829, 719, 60, 72))), (1971, (37, (830, 717, 60, 75))), (1943, (38, (832, 718, 58, 73))), (1962, (39, (828, 717, 62, 75))), (1931, (40, (827, 717, 63, 74))), (1955, (41, (827, 717, 63, 75))), (1979, (42, (828, 717, 62, 75))), (1901, (43, (832, 721, 58, 72))), (1953, (44, (829, 719, 62, 73))), (1930, (45, (828, 718, 62, 74))), (1947, (46, (828, 720, 62, 72))), (1971, (47, (828, 717, 62, 74))), (1962, (48, (829, 718, 61, 73))), (1936, (49, (828, 721, 62, 71))), (1953, (50, (828, 719, 62, 72)))]
USE_FRAME_0_BOX = False
PADDING = 25
GENERATE_BINARY_IMAGE = False
MARGINS_USED = (300, 275, 400, 175)

def get_padded_box(box, padding):
  x, y, width, height = box
  return (x-padding, y-padding, width+(padding*2), height+(padding*2)) #does not check for out of bounds

for ROI in ROIS:
  pixel_area = ROI[0]
  frame_num = ROI[1][0]
  box = ROI[1][1]
  full_frame = util.read_image(
    file_name=const.MINION_3_FRAMES[frame_num]
  )
  if MARGINS_USED != None:
    full_frame = util.get_image_center(image=full_frame, margins=MARGINS_USED)

  box_for_padding = box
  if USE_FRAME_0_BOX:
    box_for_padding = ROIS[0][1][1]

  frame_with_boxes = seg.draw_bounding_boxes(full_frame, [box], const.COLOR_GREEN, 1)
  box_cropped = seg.crop_image(image=frame_with_boxes, box=get_padded_box(box_for_padding, PADDING))

  text_color = const.COLOR_RED
  if GENERATE_BINARY_IMAGE:
    box_cropped = seg.otsu_threshold_image(box_cropped)
    text_color = (255, 255, 255)
  box_cropped = cv2.putText(box_cropped, f'Frame: {frame_num}', (0,len(box_cropped)-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

  util.write_image(box_cropped, f'frame_{frame_num}')
