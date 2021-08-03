import util
import const
import sift
import segmentation as seg

m3_img0  = util.read_image(const.MINION_3_FRAMES[0])
m3_img1  = util.read_image(const.MINION_3_FRAMES[1])

m3_img0_keypoints, m3_img0_descriptors = sift.run_sift(m3_img0, num_features=3000)
m3_img1_keypoints, m3_img1_descriptors = sift.run_sift(m3_img1, num_features=3000)

m3_img0_img1_correspondences = sift.find_sift_correspondences(m3_img0_keypoints,
                                                              m3_img0_descriptors,
                                                              m3_img1_keypoints,
                                                              m3_img1_descriptors,
                                                              ratio=0.6)

m3_img0_contours = seg.find_contours(m3_img0)
m3_img1_contours = seg.find_contours(m3_img1)

m3_img0_boxes = seg.merge_overlapping_bounding_boxes(seg.find_bounding_boxes(m3_img0_contours, padding=0))
m3_img1_boxes = seg.merge_overlapping_bounding_boxes(seg.find_bounding_boxes(m3_img1_contours, padding=0))

m3_img0_with_boxes = seg.draw_bounding_boxes(m3_img0,
                                             bounding_boxes=m3_img0_boxes,
                                             border_color=const.COLOR_GREEN,
                                             border_width=2,
                                             file_name='m3_img0_with_boxes')
m3_img1_with_boxes = seg.draw_bounding_boxes(m3_img1,
                                             bounding_boxes=m3_img1_boxes,
                                             border_color=const.COLOR_GREEN,
                                             border_width=2,
                                             file_name='m3_img1_with_boxes')

sift.plot_correspondences(image1=m3_img0_with_boxes,
                          image2=m3_img1_with_boxes,
                          correspondences=m3_img0_img1_correspondences,
                          color=const.COLOR_RED,
                          file_name='m3_img0_img1_corr_and_boxes')

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

m3_img0_img1_correspondences_filtered, m3_img0_boxes_filtered, m3_img1_boxes_filtered = filter_correspondences_and_boxes(correspondences=m3_img0_img1_correspondences, boxes_0=m3_img0_boxes, boxes_1=m3_img1_boxes)

sift.plot_correspondences(image1=m3_img0_with_boxes,
                          image2=m3_img1_with_boxes,
                          correspondences=m3_img0_img1_correspondences_filtered,
                          color=const.COLOR_RED,
                          file_name='m3_img0_img1_correspondences_filtered')
print(f'm3_img0_img1_correspondences length: {len(m3_img0_img1_correspondences)} m3_img0_img1_correspondences_filtered length: {len(m3_img0_img1_correspondences_filtered)}')

m3_img0_with_boxes_filtered = seg.draw_bounding_boxes(m3_img0, bounding_boxes=m3_img0_boxes_filtered, border_color=const.COLOR_GREEN, file_name='m3_img0_with_boxes_filtered')
m3_img1_with_boxes_filtered = seg.draw_bounding_boxes(m3_img1, bounding_boxes=m3_img1_boxes_filtered, border_color=const.COLOR_GREEN, file_name='m3_img1_with_boxes_filtered')
sift.plot_correspondences(
  image1=m3_img0_with_boxes_filtered,
  image2=m3_img1_with_boxes_filtered,
  correspondences=m3_img0_img1_correspondences_filtered,
  color=const.COLOR_RED,
  file_name='m3_img0_img1_corr_AND_boxes_filtered'
)
print(f'check image m3_img0_img1_corr_AND_boxes_filtered that only boxes with corresopndences appear')