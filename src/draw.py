import tensorflow as tf
import constants as c
from constants import layer_boxes, classes
import numpy as np
import sys
import cv2
import colorsys
import boxproc


def draw_rect(image, rectangle, color, thickness=1):
	left_top = (int(rectangle[0] * image.shape[1]), int(rectangle[1] * image.shape[0]))
	right_bottom = (int((rectangle[0] + max(rectangle[2], 0)) * image.shape[1]), int((rectangle[1] + max(rectangle[3], 0)) * image.shape[0]))
	cv2.rectangle(image, left_top, right_bottom, color, thickness)

def draw_output(image, boxes, confidences, wait=1000):
	img = (image * 255.0).astype(np.uint8)
	picks = boxproc.post_process(boxes, confidences)
	print picks
	color_r = (0, 0, 255)
	for box in picks:
		draw_rect(img, box, color_r, 2)
	cv2.imshow("outputs", img)
	cv2.waitKey(wait)

def draw_matches(image, default_boxes, matches, anns, wait=1000):
	img = (image * 255.0).astype(np.uint8)
	print img.dtype
	color_r = (0, 0, 255)
	color_b = (255, 0, 0)
	color_g = (0, 255, 0)
	for layer in range(len(layer_boxes)):
		for x in range(c.out_shapes[layer][2]):
			for y in range(c.out_shapes[layer][1]):
				for i in range(layer_boxes[layer]):
					match = matches[layer][x][y][i]
					if match == -1:
						coord = boxproc.center2cornerbox(default_boxes[layer][x][y][i])
						draw_rect(img, coord, color_b)
					elif isinstance(match, tuple):
						coord = boxproc.center2cornerbox(default_boxes[layer][x][y][i])
						draw_rect(img, coord, color_r)

	for (gt_box, box_id) in anns:
		draw_rect(img, gt_box, color_g)
	cv2.imshow("matches", img)
	cv2.waitKey(wait)