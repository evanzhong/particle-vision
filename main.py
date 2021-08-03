import util
import const
import sift
import segmentation as seg

def get_box_containing_point(point, boxes):
	point_x = point[0]
	point_y = point[1]
	for box in boxes:
		box_x = box[0]
		box_y = box[1]
		box_width = box[2]
		box_height = box[3]
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
    point_0 = correspondence[0]
    point_1 = correspondence[1]
    
    box_0_containing_point = get_box_containing_point(point_0, boxes_0)
    box_1_containing_point = get_box_containing_point(point_1, boxes_1)
    if (box_0_containing_point is not None and 
        box_1_containing_point is not None):
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
                              file_name='m3_img0_img1_corr_and_boxes')
    sift.plot_correspondences(image1=img0_drawn_boxes,
                              image2=img1_drawn_boxes,
                              correspondences=filtered_correspondences,
                              color=const.COLOR_RED,
                              file_name='m3_img0_img1_corr_filtered_and_boxes')

    img0_drawn_filtered_boxes = seg.draw_bounding_boxes(img_0, bounding_boxes=filtered_boxes_0, border_color=const.COLOR_GREEN)
    img1_drawn_filtered_boxes = seg.draw_bounding_boxes(img_1, bounding_boxes=filtered_boxes_1, border_color=const.COLOR_GREEN)
    sift.plot_correspondences(image1=img0_drawn_filtered_boxes,
                              image2=img1_drawn_filtered_boxes,
                              correspondences=img0_img1_correspondences,
                              color=const.COLOR_RED,
                              file_name='m3_img0_img1_corr_and_boxes_filtered')
    sift.plot_correspondences(image1=img0_drawn_filtered_boxes,
                              image2=img1_drawn_filtered_boxes,
                              correspondences=filtered_correspondences,
                              color=const.COLOR_RED,
                              file_name='m3_img0_img1_corr_filtered_and_boxes_filtered')

  return filtered_correspondences, filtered_boxes_0, filtered_boxes_1

if __name__ == "__main__":
  m3_img0  = util.read_image(const.MINION_3_FRAMES[0])
  m3_img1  = util.read_image(const.MINION_3_FRAMES[1])
  compare_two_images(m3_img0, m3_img1, num_sift_features=3000, sift_correspondence_ratio=0.6, should_save_images=True)
