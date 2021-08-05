import util
import const
import sift
import segmentation as seg

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

def filter_correspondences_and_boxes(correspondences, boxes_0, boxes_1):
  filtered_correspondences = []
  filtered_boxes_0 = []
  filtered_boxes_1 = []
  for correspondence in correspondences:
    point_0, point_1 = correspondence
    
    box_0_containing_point = get_box_containing_point(point_0, boxes_0)
    box_1_containing_point = get_box_containing_point(point_1, boxes_1)
    if (box_0_containing_point is not None and
        box_1_containing_point is not None and
        box_0_containing_point not in filtered_boxes_0 and
        box_1_containing_point not in filtered_boxes_1
        ):
        filtered_correspondences.append(correspondence)
        filtered_boxes_0.append(box_0_containing_point)
        filtered_boxes_1.append(box_1_containing_point)

  return (filtered_correspondences, filtered_boxes_0, filtered_boxes_1)

def compare_two_images(img_0, img_1, num_sift_features, sift_correspondence_ratio, should_save_images = False):
  img0_keypoints, img0_descriptors = sift.run_sift(img_0, num_features=num_sift_features)
  img1_keypoints, img1_descriptors = sift.run_sift(img_1, num_features=num_sift_features)

  img0_img1_correspondences = sift.find_sift_correspondences(img0_keypoints,
                                                             img0_descriptors,
                                                             img1_keypoints,
                                                             img1_descriptors,
                                                             ratio=sift_correspondence_ratio)

  img0_contours = seg.find_contours(img_0)
  img1_contours = seg.find_contours(img_1)
  img0_boxes = seg.merge_overlapping_bounding_boxes(
    bounding_boxes=seg.find_bounding_boxes(img0_contours, padding=0)
  )
  img1_boxes = seg.merge_overlapping_bounding_boxes(
    bounding_boxes=seg.find_bounding_boxes(img1_contours, padding=0)
  )

  filtered_correspondences, filtered_boxes_0, filtered_boxes_1 = filter_correspondences_and_boxes(
    correspondences=img0_img1_correspondences,
    boxes_0=img0_boxes,
    boxes_1=img1_boxes
  )

  if should_save_images:
    img0_drawn_boxes = seg.draw_bounding_boxes(img_0, bounding_boxes=img0_boxes, border_color=const.COLOR_GREEN)
    img1_drawn_boxes = seg.draw_bounding_boxes(img_1, bounding_boxes=img1_boxes, border_color=const.COLOR_GREEN)
    sift.plot_correspondences(image1=img0_drawn_boxes,
                              image2=img1_drawn_boxes,
                              correspondences=img0_img1_correspondences,
                              color=const.COLOR_RED,
                              file_name='img0_img1_corr_and_boxes')
    sift.plot_correspondences(image1=img0_drawn_boxes,
                              image2=img1_drawn_boxes,
                              correspondences=filtered_correspondences,
                              color=const.COLOR_RED,
                              file_name='img0_img1_corr_filtered_and_boxes')

    img0_drawn_filtered_boxes = seg.draw_bounding_boxes(img_0, bounding_boxes=filtered_boxes_0, border_color=const.COLOR_GREEN)
    img1_drawn_filtered_boxes = seg.draw_bounding_boxes(img_1, bounding_boxes=filtered_boxes_1, border_color=const.COLOR_GREEN)
    sift.plot_correspondences(image1=img0_drawn_filtered_boxes,
                              image2=img1_drawn_filtered_boxes,
                              correspondences=img0_img1_correspondences,
                              color=const.COLOR_RED,
                              file_name='img0_img1_corr_and_boxes_filtered')
    sift.plot_correspondences(image1=img0_drawn_filtered_boxes,
                              image2=img1_drawn_filtered_boxes,
                              correspondences=filtered_correspondences,
                              color=const.COLOR_RED,
                              file_name='img0_img1_corr_filtered_and_boxes_filtered')

  return filtered_correspondences, filtered_boxes_0, filtered_boxes_1

def readable_correspondence(correspondence):
  point_0, point_1 = correspondence
  return ((round(point_0[0]), round(point_0[1])), (round(point_1[0]), round(point_1[1])))

def analyze_frames(frames, num_sift_features, sift_correspondence_ratio, should_save_images=False):
  GLOBAL_MAP = {}
  GLOBAL_LIST = []
  for frame_n in range(len(frames)-1):
    imgN  = util.read_image(frames[frame_n])
    imgN_1  = util.read_image(frames[frame_n+1])

    corrs, boxes_N, boxes_N_1 = compare_two_images(
      imgN,
      imgN_1,
      num_sift_features=num_sift_features,
      sift_correspondence_ratio=sift_correspondence_ratio,
      should_save_images=should_save_images
    )

    # Initialize the GLOBAL objects with frame_0 information
    if frame_n == 0:
      for box_0 in boxes_N:
        list_index = len(GLOBAL_LIST)
        GLOBAL_MAP[box_0] = list_index

        ROI_area = seg.get_non_zero_pixel_area(
          image=seg.crop_image(image=imgN, box=box_0)
        )
        GLOBAL_LIST.append([])
        GLOBAL_LIST[list_index].append((ROI_area, (0, box_0)))

    print(f'\nGLOBAL_MAP: {GLOBAL_MAP}')
    curr_map = {} #object to collect current maaping
    for corr, box_N, box_N_1 in zip(corrs, boxes_N, boxes_N_1):
      imgN_ROI = seg.crop_image(imgN, box_N)
      imgN_1_ROI = seg.crop_image(imgN_1, box_N_1)
      imgN_ROI_area = seg.get_non_zero_pixel_area(imgN_ROI)
      imgN_1_ROI_area = seg.get_non_zero_pixel_area(imgN_1_ROI)
      print(f'({imgN_ROI_area},{imgN_1_ROI_area}) img{frame_n}_{box_N} to img{frame_n+1}_{box_N_1} via {readable_correspondence(corr)}')
      if should_save_images: util.write_image(imgN_ROI, f'img{frame_n}_{box_N}')

      list_index = -1
      if box_N in GLOBAL_MAP:
        list_index = GLOBAL_MAP[box_N]
      else:
        list_index = len(GLOBAL_LIST)
        GLOBAL_LIST.append([])

      GLOBAL_LIST[list_index].append((imgN_1_ROI_area, (frame_n+1, box_N_1)))
      curr_map[box_N_1] = list_index
    GLOBAL_MAP = curr_map

  print(GLOBAL_LIST)
  return GLOBAL_LIST

if __name__ == "__main__":
  analyze_frames(
    frames=const.MINION_3_FRAMES,
    num_sift_features=3000,
    sift_correspondence_ratio=0.6,
    should_save_images=False
  )
