import const
import util
import segmentation as seg
import cv2

# Update the global variables below to whatever movie you want
ROIS = [(102, (0, (850, 452, 16, 12))), (873, (1, (1518, 277, 32, 53))), (857, (2, (1519, 277, 31, 53))), (841, (3, (1518, 276, 32, 54))), (882, (4, (1519, 277, 34, 53))), (869, (5, (1518, 277, 32, 53))), (870, (6, (1518, 277, 37, 53))), (889, (7, (1518, 275, 36, 55))), (924, (8, (1519, 276, 34, 55))), (879, (9, (1518, 276, 34, 55))), (883, (10, (1519, 276, 34, 56))), (1038, (11, (1512, 275, 40, 57))), (936, (12, (1518, 276, 34, 56))), (940, (13, (1518, 276, 36, 55))), (864, (14, (1519, 275, 32, 56))), (884, (15, (1518, 275, 33, 57))), (862, (16, (1519, 276, 32, 55))), (880, (17, (1519, 275, 33, 56))), (915, (18, (1519, 275, 33, 56))), (912, (19, (1518, 276, 33, 55))), (911, (20, (1518, 275, 34, 56))), (908, (21, (1518, 274, 34, 57))), (882, (22, (1518, 276, 34, 55))), (907, (23, (1518, 275, 33, 56))), (918, (24, (1518, 276, 33, 55))), (905, (25, (1519, 275, 32, 56))), (889, (26, (1518, 276, 34, 55))), (903, (27, (1519, 276, 34, 55))), (921, (28, (1518, 273, 33, 58))), (884, (29, (1518, 275, 33, 55))), (884, (30, (1518, 275, 33, 56))), (903, (31, (1519, 275, 34, 56))), (894, (32, (1519, 276, 34, 54))), (893, (33, (1518, 276, 34, 54))), (881, (34, (1518, 276, 36, 54))), (870, (35, (1518, 275, 34, 55))), (859, (36, (1519, 274, 33, 56))), (890, (37, (1518, 276, 34, 55))), (914, (38, (1518, 275, 36, 56))), (796, (39, (1519, 274, 27, 56))), (887, (40, (1518, 276, 34, 55))), (923, (41, (1518, 276, 36, 54))), (907, (42, (1518, 275, 33, 55))), (900, (43, (1518, 276, 35, 54))), (861, (44, (1518, 275, 35, 56))), (911, (45, (1518, 274, 37, 57))), (531, (46, (1519, 298, 34, 32))), (885, (47, (1519, 276, 35, 54)))]
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
  if USE_FRAME_0_BOX:
    box = ROIS[0][1][1]

  frame_with_boxes = seg.draw_bounding_boxes(full_frame, [box], const.COLOR_GREEN, 1)
  box_cropped = seg.crop_image(image=frame_with_boxes, box=get_padded_box(box, PADDING))

  text_color = const.COLOR_RED
  if GENERATE_BINARY_IMAGE:
    box_cropped = seg.otsu_threshold_image(box_cropped)
    text_color = (255, 255, 255)
  box_cropped = cv2.putText(box_cropped, f'Frame: {frame_num}', (0,len(box_cropped)-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

  util.write_image(box_cropped, f'frame_{frame_num}')
