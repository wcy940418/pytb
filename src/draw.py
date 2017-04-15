import tensorflow as tf
import constants as c
from constants import layer_boxes, classes
import numpy as np
import sys
import os
import cv2
import colorsys
import boxproc


def draw_rect(image, rectangle, color, thickness=1):
	left_top = (int(rectangle[0] * image.shape[1]), int(rectangle[1] * image.shape[0]))
	right_bottom = (int((rectangle[0] + max(rectangle[2], 0)) * image.shape[1]), int((rectangle[1] + max(rectangle[3], 0)) * image.shape[0]))
	cv2.rectangle(image, left_top, right_bottom, color, thickness)

def draw_output(image, boxes, confidences, wait=1000, mode='display', step=0, path=None):
	img = (image * 255.0).astype(np.uint8)
	picks = boxproc.post_process(boxes, confidences)
	color_r = (0, 0, 255)
	for box in picks:
		draw_rect(img, box, color_r, 1)
	if mode == 'display':
		cv2.imshow("outputs", img)
		cv2.waitKey(wait)
	elif mode == 'save':
		write_path = os.path.join(path, str(step) + '_output.jpg')
		# print write_path
		cv2.imwrite(write_path, img)

def draw_matches(image, default_boxes, matches, anns, wait=1000, mode='display', step=0, path=None):
	img = (image * 255.0).astype(np.uint8)
	color_r = (0, 0, 255)
	color_b = (255, 0, 0)
	color_g = (0, 255, 0)
	for layer in range(len(layer_boxes)):
		for y in range(c.out_shapes[layer][1]):
			for x in range(c.out_shapes[layer][2]):
				for i in range(layer_boxes[layer]):
					match = matches[layer][y][x][i]
					if match == -1:
						coord = boxproc.center2cornerbox(default_boxes[layer][y][x][i])
						draw_rect(img, coord, color_b)
					elif isinstance(match, tuple):
						coord = boxproc.center2cornerbox(default_boxes[layer][y][x][i])
						draw_rect(img, coord, color_r)

	for (gt_box, box_id) in anns:
		draw_rect(img, gt_box, color_g)
	if mode == 'display':
		cv2.imshow("matches", img)
		cv2.waitKey(wait)
	elif mode == 'save':
		write_path = os.path.join(path, str(step) + '_matches.jpg')
		# print write_path
		cv2.imwrite(write_path, img)