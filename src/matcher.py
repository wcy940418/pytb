import constants as c
from constants import layer_boxes, classes, negposratio
# cant import out_shapes and defaults here since its still not initialized
from boxproc import center2cornerbox, calc_jaccard
import numpy as np

def get_top_confidences(pred_labels):
	confidences = []

	for index in range(pred_labels.shape[0]):
		logits = pred_labels[index]
		probs = np.exp(logits) / np.sum(np.exp(logits))
		top_probs = np.amax(probs)
		top_probs_label = np.argmax(probs)
		confidences.append((index, top_probs, top_probs_label))
	top_confidences = sorted(confidences, key=lambda x: x[1], reverse=True)
	return top_confidences

class Matcher:
	def __init__(self):
		self.index2indices = []
		for o_i in range(len(layer_boxes)):
			for x in range(c.out_shapes[o_i][2]):
				for y in range(c.out_shapes[o_i][1]):
					for i in range(layer_boxes[o_i]):
						self.index2indices.append([o_i, x, y, i])

	def match_boxes(self, pred_labels, anns):
		matches = [[[[None for i in range(c.layer_boxes[o])] for y in range(c.out_shapes[o][1])] for y in range(c.out_shapes[o][2])]
				 for o in range(len(layer_boxes))]
		positive_count = 0

		for index, (gt_box, box_id) in zip(range(len(anns)), anns):

			top_match = (None, 0)

			for o in range(len(layer_boxes)):
				x1 = max(int(gt_box[0] * c.out_shapes[o][2]), 0)
				y1 = max(int(gt_box[1] * c.out_shapes[o][1]), 0)
				x2 = min(int((gt_box[0] + gt_box[2]) * c.out_shapes[o][2])+2, c.out_shapes[o][2])
				y2 = min(int((gt_box[1] + gt_box[3]) * c.out_shapes[o][1])+2, c.out_shapes[o][1])

				for y in range(y1, y2):
					for x in range(x1, x2):
						for i in range(layer_boxes[o]):
							box = c.defaults[o][x][y][i]
							jacc = calc_jaccard(gt_box, center2cornerbox(box)) #gt_box is corner, box is center-based so convert
							if jacc >= 0.5:
								matches[o][x][y][i] = (gt_box, box_id)
								positive_count += 1
							if jacc > top_match[1]:
								top_match = ([o, x, y, i], jacc)

			top_box = top_match[0]
			#if box's jaccard is <0.5 but is the best
			if top_box is not None and matches[top_box[0]][top_box[1]][top_box[2]][top_box[3]] is None:
				positive_count += 1
				matches[top_box[0]][top_box[1]][top_box[2]][top_box[3]] = (gt_box, box_id)

		negative_max = positive_count * negposratio
		negative_count = 0

		confidences = get_top_confidences(pred_labels)

		for index, top_probs, top_probs_label in confidences:
			indices = self.index2indices[index]

			if matches[indices[0]][indices[1]][indices[2]][indices[3]] is None and top_probs_label != classes:
				matches[indices[0]][indices[1]][indices[2]][indices[3]] = -1
				negative_count += 1

				if negative_count >= negative_max:
					break

		return matches
