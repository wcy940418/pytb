import tensorflow as tf
import numpy as np
from constants import *
import constants as c
import math


def box_scale(k):
	s_min = box_s_min
	s_max = 0.95
	m = 6.0

	s_k = s_min + (s_max - s_min) * (k - 1.0) / (m - 1.0) # equation 2

	return s_k
# center
def default_boxes(out_shapes):
	boxes = []
	print out_shapes
	for o_i in range(len(out_shapes)):
		layer_boxes = []
		layer_shape = out_shapes[o_i]
		s_k = box_scale(o_i + 1)
		for x in range(layer_shape[2]):
			x_boxes = []
			for y in range(layer_shape[1]):
				y_boxes = []
				rs = box_ratios
				for i in range(len(rs)):
					scale = s_k
					default_w = scale * np.sqrt(rs[i])
					default_h = scale / np.sqrt(rs[i])
					c_x = (x + 0.5) / float(layer_shape[2])
					c_y = (y + 0.5) / float(layer_shape[1])
					y_boxes.append([c_x, c_y, default_w, default_h])
					c_y = (y + 1.0) / float(layer_shape[1])
					y_boxes.append([c_x, c_y, default_w, default_h])
				x_boxes.append(y_boxes)
			layer_boxes.append(x_boxes)
		boxes.append(layer_boxes)
	return boxes
# center -> corner
def center2cornerbox(rect):
	return [rect[0] - rect[2]/2.0, rect[1] - rect[3]/2.0, rect[2], rect[3]]
# corner -> center
def corner2centerbox(rect):
	return [rect[0] + rect[2]/2.0, rect[1] + rect[3]/2.0, rect[2], rect[3]]
# center -> diag
def center2diagbox(rect):
	return [rect[0] - rect[2] / 2.0, rect[1] - rect[3] / 2.0, rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0]
# corner -> diag
def corner2diagbox(rect):
	return [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
# corner
def calc_intersection(r1, r2):
	left = max(r1[0], r2[0])
	right = min(r1[0] + r1[2], r2[0] + r2[2])
	bottom = min(r1[1] + r1[3], r2[1] + r2[3])
	top = max(r1[1], r2[1])

	if left < right and top < bottom:
		return (right - left) * (bottom - top)

	return 0
# corner
def clip_box(r):
	return [r[0], r[1], max(r[2], 0.01), max(r[3], 0.01)]
# corner
def calc_jaccard(r1, r2):
	r1_ = clip_box(r1)
	r2_ = clip_box(r2)
	intersection = calc_intersection(r1_, r2_)
	union = r1_[2] * r1_[3] + r2_[2] * r2_[3] - intersection

	if union <= 0:
		return 0

	j = intersection / union

	return j
# corner
def calc_overlap(r1, host):
	intersection = calc_intersection(r1, host)
	return intersection / (1e-5 + host[2] * host[3])
# center
def calc_offsets(default, truth):
	dX = (truth[0] - default[0]) / float(default[2])
	dY = (truth[1] - default[1]) / float(default[3])
	dW = np.log(truth[2] / default[2])
	dH = np.log(truth[3] / default[3])
	return [dX, dY, dW, dH]
# corner
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	# x2 = boxes[:, 2]
	# y2 = boxes[:, 3]
	# using corner box
	x2 = x1 + boxes[:, 2]
	y2 = y1 + boxes[:, 3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick]
def check_limits(x):
	def supress(x):
		if x >= 0.0 and x <= 1.0:
			return True
		return False
	x = center2diagbox(x)
	if supress(x[0]) and supress(x[1]) and supress(x[2]) and supress(x[3]):
		return True
	return False
# center
def format_output(softmaxed_pred_labels_max_prob, softmaxed_pred_labels_max_index, pred_locs):
	boxes = [[[[None for i in range(layer_boxes[layer])] for x in range(c.out_shapes[layer][2])] for y in range(c.out_shapes[layer][1])] for layer in range(len(layer_boxes))]
	index = 0
	confidences = []
	for layer in range(len(layer_boxes)):
		for x in range(c.out_shapes[layer][2]):
			for y in range(c.out_shapes[layer][1]):
				for i in range(layer_boxes[layer]):
					# convert output predicted location to center based box
					diffs = pred_locs[index]
					default = c.defaults[layer][x][y][i]
					c_x = default[0] + default[2] * diffs[0]
					c_y = default[1] + default[3] * diffs[1]
					w = default[2] * np.exp(diffs[2])
					h = default[3] * np.exp(diffs[3])
					boxes[layer][x][y][i] = [c_x, c_y, w, h]
					# convert output predicted label
					# logits = pred_labels[index]
					# max_logits = np.amax(np.exp(logits) / np.sum(np.exp(logits)))
					# max_logits_label = np.argmax(logits)
					max_logits = softmaxed_pred_labels_max_prob[index]
					max_logits_label = softmaxed_pred_labels_max_index[index]
					indices = [layer, x, y, i]
					info = (indices, max_logits, max_logits_label)
					confidences.append(info)
					index += 1
	return boxes, confidences

def post_process(boxes, confidences, min_conf=0.01, nms=0.45):
	raw_boxes = []
	for indices, max_logits, max_logits_label in  confidences:
		if max_logits_label != classes and max_logits >= min_conf:
			if check_limits(boxes[indices[0]][indices[1]][indices[2]][indices[3]]):
				raw_boxes.append(boxes[indices[0]][indices[1]][indices[2]][indices[3]])
	raw_boxes = np.asarray(raw_boxes)
	return non_max_suppression_fast(raw_boxes, nms)
	return raw_boxes

def prepare_feed(matches):
	positives_list = []
	negatives_list = []
	true_labels_list = []
	true_locs_list = []

	for o in range(len(layer_boxes)):
		for x in range(c.out_shapes[o][2]):
			for y in range(c.out_shapes[o][1]):
				for i in range(layer_boxes[o]):
					match = matches[o][x][y][i]

					if isinstance(match, tuple): # there is a ground truth assigned to this default box
						positives_list.append(1)
						negatives_list.append(0)
						true_labels_list.append(match[1]) #id
						default = c.defaults[o][x][y][i]
						true_locs_list.append(calc_offsets(default, corner2centerbox(match[0])))
					elif match == -1: # this default box was chosen to be a negative
						positives_list.append(0)
						negatives_list.append(1)
						true_labels_list.append(classes) # background class
						true_locs_list.append([0]*4)
					else: # no influence for this training step
						positives_list.append(0)
						negatives_list.append(0)
						true_labels_list.append(classes)  # background class
						true_locs_list.append([0]*4)

	a_positives = np.asarray(positives_list)
	a_negatives = np.asarray(negatives_list)
	a_true_labels = np.asarray(true_labels_list)
	a_true_locs = np.asarray(true_locs_list)

	return a_positives, a_negatives, a_true_labels, a_true_locs
