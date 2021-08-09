import util
import const
import sift
import segmentation as seg

def filter_correspondences_and_boxes(correspondences, boxes_0, boxes_1):
  filtered_correspondences = []
  filtered_boxes_0 = []
  filtered_boxes_1 = []
  for correspondence in correspondences:
    point_0 = correspondence[0]
    point_1 = correspondence[1]
    
    box_0_containing_point = util.get_box_containing_point(point_0, boxes_0)
    box_1_containing_point = util.get_box_containing_point(point_1, boxes_1)
    if (box_0_containing_point is not None and
        box_1_containing_point is not None and
        box_0_containing_point not in filtered_boxes_0 and
        box_1_containing_point not in filtered_boxes_1
        ):
        filtered_correspondences.append(correspondence)
        filtered_boxes_0.append(box_0_containing_point)
        filtered_boxes_1.append(box_1_containing_point)

  return (filtered_correspondences, filtered_boxes_0, filtered_boxes_1)

def compare_two_images(img_0, img_1, num_sift_features, sift_correspondence_ratio):
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

  return filtered_correspondences, filtered_boxes_0, filtered_boxes_1

def track_particle_motion(frames, num_sift_features, sift_correspondence_ratio, should_save_comparisons=False, should_save_ROIs=False, margins=None):
  GLOBAL_MAP = {}
  GLOBAL_LIST = []
  for frame_n in range(len(frames)-1):
    frame_n_filename = frames[frame_n]
    frame_n_1_filename = frames[frame_n+1]
    imgN  = util.read_image(frame_n_filename)
    imgN_1  = util.read_image(frame_n_1_filename)
    if margins != None:
      imgN = get_image_center(image=imgN, margins=margins)
      imgN_1 = get_image_center(image=imgN_1, margins=margins)

    corrs, boxes_N, boxes_N_1 = compare_two_images(
      imgN,
      imgN_1,
      num_sift_features=num_sift_features,
      sift_correspondence_ratio=sift_correspondence_ratio,
    )

    if should_save_comparisons:
      imgN_with_boxes = seg.draw_bounding_boxes(imgN, bounding_boxes=boxes_N, border_color=const.COLOR_GREEN)
      img_N_1_with_boxes = seg.draw_bounding_boxes(imgN_1, bounding_boxes=boxes_N_1, border_color=const.COLOR_GREEN)
      sift.plot_correspondences(
        image1=imgN_with_boxes,
        image2=img_N_1_with_boxes,
        correspondences=corrs,
        color=const.COLOR_RED,
        file_name=f'{util.get_no_extension_filename(frame_n_filename)}_to_{util.get_no_extension_filename(frame_n_1_filename)}'
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
      print(f'({imgN_ROI_area},{imgN_1_ROI_area}) img{frame_n}_{box_N} to img{frame_n+1}_{box_N_1} via {util.get_readable_correspondence(corr)}')
      if should_save_ROIs: util.write_image(imgN_ROI, f'img{frame_n}_{box_N}')

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

def bulk_carbon(frames, margins=None, should_save_images=False):
  frame_to_carbon_map = {}
  for frame in frames:
    frame_image = util.read_image(frame)
    if margins != None:
      frame_image = get_image_center(image=frame_image, margins=margins)

    contours = seg.find_contours(image=frame_image)
    boxes = seg.merge_overlapping_bounding_boxes(
      bounding_boxes=seg.find_bounding_boxes(contours=contours, padding=0)
    )

    if should_save_images:
      util.write_image(
        image=seg.draw_bounding_boxes(image=frame_image, bounding_boxes=boxes, border_color=const.COLOR_GREEN),
        file_output_name=f'{util.get_no_extension_filename(frame)}_boxes'
      )

    total_pixel_area_for_frame = 0
    num_particles = len(boxes)
    for box in boxes:
      pixel_area = seg.get_non_zero_pixel_area(
        image=seg.crop_image(image=frame_image, box=box)
      )
      total_pixel_area_for_frame += pixel_area
    print(f'Frame: {frame} has total particle pixel area: {total_pixel_area_for_frame} and total # of particles: {num_particles}')
    frame_to_carbon_map[frame] = (total_pixel_area_for_frame, num_particles)

  print(frame_to_carbon_map)
  return frame_to_carbon_map

def get_image_center(image, margins):
  image_shape = image.shape
  margin_top, margin_right, margin_bot, margin_left = margins
  max_y, max_x, channels = image_shape
  center_box = (margin_left, margin_top, max_x-(margin_right*2), max_y-(margin_bot*2))
  cropped_image = seg.crop_image(image=image, box=center_box)
  return cropped_image

if __name__ == "__main__":
  MARGINS_TO_USE = (300, 275, 400, 175)
  FRAMES_TO_ANALYZE = const.MINION_3_FRAMES

  track_particle_motion(
    frames=FRAMES_TO_ANALYZE,
    num_sift_features=3000,
    sift_correspondence_ratio=0.6,
    should_save_comparisons=False,
    should_save_ROIs=False,
    margins=MARGINS_TO_USE
  )

  bulk_carbon(
    frames=FRAMES_TO_ANALYZE,
    margins=MARGINS_TO_USE,
    should_save_images=False,
  )
