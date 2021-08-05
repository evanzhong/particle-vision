import const
import util
import segmentation as seg
import cv2

# Update the global variables below to whatever movie you want
ROIS = [(851, (0, (1693, 576, 32, 54))), (873, (1, (1693, 577, 32, 53))), (857, (2, (1694, 577, 31, 53))), (841, (3, (1693, 576, 32, 54))), (882, (4, (1694, 577, 34, 53))), (869, (5, (1693, 577, 32, 53))), (870, (6, (1693, 577, 37, 53))), (889, (7, (1693, 575, 36, 55))), (924, (8, (1694, 576, 34, 55))), (879, (9, (1693, 576, 34, 55))), (883, (10, (1694, 576, 34, 56))), (1038, (11, (1687, 575, 40, 57))), (936, (12, (1693, 576, 34, 56))), (940, (13, (1693, 576, 36, 55))), (864, (14, (1694, 575, 32, 56))), (884, (15, (1693, 575, 33, 57))), (862, (16, (1694, 576, 32, 55))), (880, (17, (1694, 575, 33, 56))), (915, (18, (1694, 575, 33, 56))), (912, (19, (1693, 576, 33, 55))), (911, (20, (1693, 575, 34, 56))), (908, (21, (1693, 574, 34, 57))), (882, (22, (1693, 576, 34, 55))), (907, (23, (1693, 575, 33, 56))), (918, (24, (1693, 576, 33, 55))), (905, (25, (1694, 575, 32, 56))), (889, (26, (1693, 576, 34, 55))), (903, (27, (1694, 576, 34, 55))), (921, (28, (1693, 573, 33, 58))), (884, (29, (1693, 575, 33, 55))), (884, (30, (1693, 575, 33, 56))), (903, (31, (1694, 575, 34, 56))), (894, (32, (1694, 576, 34, 54))), (893, (33, (1693, 576, 34, 54))), (881, (34, (1693, 576, 36, 54))), (870, (35, (1693, 575, 34, 55))), (859, (36, (1694, 574, 33, 56))), (890, (37, (1693, 576, 34, 55))), (914, (38, (1693, 575, 36, 56))), (796, (39, (1694, 574, 27, 56))), (887, (40, (1693, 576, 34, 55))), (923, (41, (1693, 576, 36, 54))), (907, (42, (1693, 575, 33, 55))), (900, (43, (1693, 576, 35, 54))), (861, (44, (1693, 575, 35, 56))), (911, (45, (1693, 574, 37, 57))), (531, (46, (1694, 598, 34, 32))), (885, (47, (1694, 576, 35, 54)))]
FRAME_0_BOX = ROIS[0][1][1]
PADDING = 25
BINARY_IMAGE = False

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
  frame_with_boxes = seg.draw_bounding_boxes(full_frame, [box], const.COLOR_GREEN, 1)
  box_cropped = seg.crop_image(image=frame_with_boxes, box=get_padded_box(FRAME_0_BOX, PADDING))

  text_color = const.COLOR_RED
  if BINARY_IMAGE:
    box_cropped = seg.otsu_threshold_image(box_cropped)
    text_color = (255, 255, 255)
  box_cropped = cv2.putText(box_cropped, f'Frame: {frame_num}', (0,len(box_cropped)-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

  util.write_image(box_cropped, f'{frame_num}')
