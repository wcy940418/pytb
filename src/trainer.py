import tensorflow as tf
from matcher import Matcher
from model import TB, TB_Loss
import svt_data_loader as sLoader
import constants as c
from constants import layer_boxes, classes, image_size
import numpy as np
import signal
import sys
import cv2
import time
import boxproc
import draw
import utility
import os

class Conf:
	def __init__(self):
		self.trainBatchSize = 8
		self.testBatchSize = 2
		self.maxIteration = 2000000
		self.displayInterval = 100
		self.evalInterval = 50
		self.testInterval = 1000
		self.saveInterval = 100
		self.modelDir = os.path.abspath(os.path.join('..', 'model', 'ckpt'))
		self.trainDataSet = os.path.join('..', 'data', 'svt1', 'train.xml')
		self.testDataSet = os.path.join('..', 'data', 'svt1', 'test.xml')
		self.display = False

if __name__ == '__main__':
	gConfig = Conf()
	# Start a new session
	sess = tf.InteractiveSession()
	# Declare placeholders for graph
	images = tf.placeholder("float", [None, image_size, image_size, 3])
	trainPhase = tf.placeholder(tf.bool)
	# Build graph
	tb = TB(images, trainPhase, sess)
	total_boxes = tb.pred_labels.shape[1]
	c.out_shapes = [out.get_shape().as_list() for out in tb.outputs]
	c.defaults = boxproc.default_boxes(c.out_shapes)
	# Declare placeholders for loss functions
	positives_ph = tf.placeholder(tf.float32, [None, total_boxes])
	negatives_ph = tf.placeholder(tf.float32, [None, total_boxes])
	true_labels_ph = tf.placeholder(tf.int32, [None, total_boxes])
	true_locs_ph = tf.placeholder(tf.float32, [None, total_boxes, 4])
	# Build loss functions
	loss = TB_Loss(tb.pred_labels, tb.pred_locs, true_labels_ph, true_locs_ph, positives_ph, negatives_ph)
	ckpt = utility.checkPointLoader(gConfig.modelDir)
	box_matcher = Matcher()
	train_loader = sLoader.SVT(gConfig.trainDataSet, gConfig.testDataSet)
	def signal_handler(signal, frame):
		print('You pressed Ctrl+C!')
		tb.saveModel(gConfig.modelDir, step)
		print("%d steps trained model has saved" % step)
		sys.exit(0)
	signal.signal(signal.SIGINT, signal_handler)
	global_step = tf.Variable(0)
	optimizer = tf.train.AdamOptimizer(0.001).minimize(loss.total_loss, global_step=global_step)
	if ckpt is None:
		init = tf.global_variables_initializer()
		sess.run(init)
		step = 0
	else:
		tb.loadModel(ckpt)
		step = sess.run([global_step])
	while True:
		t = time.time()
		imgs, anns = train_loader.nextBatch(gConfig.trainBatchSize)
		pred_labels, pred_locs = sess.run([tb.pred_labels, tb.pred_locs], feed_dict={tb.input: imgs, tb.trainPhase: False})
		batch_values = [None for i in range(gConfig.trainBatchSize)]
		def build_match_boxes(batch):
			matches = box_matcher.match_boxes(pred_labels[batch], anns[batch])
			positives, negatives, tru_labels, true_locs = boxproc.prepare_feed(matches)
			batch_values[batch] = (positives, negatives, tru_labels, true_locs)
			if batch == 0 && gConfig.display:
				boxes, confidences = boxproc.format_output(pred_labels[batch], pred_locs[batch])
				draw.draw_output(imgs[batch], boxes, confidences)
				draw.draw_matches(imgs[batch], c.defaults, matches, anns[batch])
		for batch in range(gConfig.trainBatchSize):
			build_match_boxes(batch)
		positives, negatives, true_labels, true_locs = [np.stack(m) for m in zip(*batch_values)]

		cost, _, step = sess.run([loss.total_loss, optimizer, global_step], feed_dict={
											tb.input: imgs, 
											tb.trainPhase: True, 
											loss.positives: positives,
											loss.negatives: negatives,
											loss.true_labels: true_labels,
											loss.true_locs: true_locs
											})
		t = time.time() - t
		print("step:%d, loss: %f, time elapse: %.2f secs" % (step, cost, t))
		if step >= gConfig.maxIteration:
			print("%d training has completed" % gConfig.maxIteration)
			tb.saveModel(gConfig.modelDir, step)
			sys.exit(0)
		if step != 0 and step % gConfig.saveInterval == 0:
			print("%d training has saved" % step)
			tb.saveModel(gConfig.modelDir, step)

		