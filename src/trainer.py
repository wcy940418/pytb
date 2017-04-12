import tensorflow as tf
from matcher import Matcher
from model import TB, TB_Loss
import svt_data_loader as sLoader
import dataset
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
		self.maxIteration = 60000
		self.displayInterval = 1
		self.evalInterval = 50
		self.testInterval = 1
		self.saveInterval = 5000
		self.modelDir = os.path.abspath(os.path.join('..', 'model', 'ckpt'))
		# self.trainDataSet = os.path.join('..', 'data', 'svt1', 'train.xml')
		self.trainDataSet = os.path.join('..', 'data', 'SynthTextLmdb')
		self.auxTrainDataSet = os.path.join('..', 'data', 'SynthText')
		# self.testDataSet = os.path.join('..', 'data', 'svt1', 'test.xml')
		self.display = False
		self.saveSnapShot = True
		self.trainLogPath = os.path.abspath(os.path.join('..', 'model', 'train'))
		self.snapShotPath = os.path.abspath(os.path.join('..', 'model', 'snapShot'))

if __name__ == '__main__':
	gConfig = Conf()
	# Start a new session
	sess = tf.InteractiveSession()
	# Declare placeholders for graph
	images = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
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
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(gConfig.trainLogPath, sess.graph)
	ckpt = utility.checkPointLoader(gConfig.modelDir)
	box_matcher = Matcher()
	# train_loader = sLoader.SVT(gConfig.trainDataSet, gConfig.testDataSet)
	train_loader = dataset.SynthLmdb(gConfig.trainDataSet, gConfig.auxTrainDataSet)
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
		step = sess.run(global_step)
	start_time = time.time()
	t = start_time
	while True:
		imgs, anns = train_loader.nextBatch(gConfig.trainBatchSize)
		softmaxed_pred_labels_max_prob, softmaxed_pred_labels_max_index, pred_locs \
			= sess.run([tb.softmaxed_pred_labels_max_prob, tb.softmaxed_pred_labels_max_index, tb.pred_locs], \
			feed_dict={tb.input: imgs, tb.trainPhase: False})
		batch_values = [None for i in range(gConfig.trainBatchSize)]
		def build_match_boxes(batch, step, snapShotInterval):
			matches = box_matcher.match_boxes(softmaxed_pred_labels_max_prob[batch], softmaxed_pred_labels_max_index[batch], anns[batch])
			positives, negatives, tru_labels, true_locs = boxproc.prepare_feed(matches)
			batch_values[batch] = (positives, negatives, tru_labels, true_locs)
			if batch == 0 and gConfig.display:
				boxes, confidences = boxproc.format_output(softmaxed_pred_labels_max_prob[batch], softmaxed_pred_labels_max_index[batch], pred_locs[batch])
				draw.draw_output(imgs[batch], boxes, confidences)
				draw.draw_matches(imgs[batch], c.defaults, matches, anns[batch])
			if batch == 0 and gConfig.saveSnapShot and step != 0 and step % snapShotInterval == 0:
				boxes, confidences = boxproc.format_output(softmaxed_pred_labels_max_prob[batch], softmaxed_pred_labels_max_index[batch], pred_locs[batch])
				draw.draw_output(imgs[batch], boxes, confidences, mode='save', step=step, path=gConfig.snapShotPath)
				draw.draw_matches(imgs[batch], c.defaults, matches, anns[batch], mode='save', step=step, path=gConfig.snapShotPath)
		for batch in range(gConfig.trainBatchSize):
			build_match_boxes(batch, step, gConfig.testInterval)
		positives, negatives, true_labels, true_locs = [np.stack(m) for m in zip(*batch_values)]

		cost, _, step, summary = sess.run([loss.total_loss, optimizer, global_step, merged], feed_dict={
											tb.input: imgs, 
											tb.trainPhase: True, 
											loss.positives: positives,
											loss.negatives: negatives,
											loss.true_labels: true_labels,
											loss.true_locs: true_locs
											})
		if step != 0 and step % gConfig.displayInterval == 0:
			t_step = time.time() - t
			t = time.time()
			total_time = time.time() - start_time
			train_writer.add_summary(summary, step)
			print("step:%d, loss: %f, step time usage: %.2f, time elapse: %.2f secs" % (step, cost, t_step, total_time))
		if step >= gConfig.maxIteration:
			print("%d training has completed" % gConfig.maxIteration)
			tb.saveModel(gConfig.modelDir, step)
			sys.exit(0)
		if step != 0 and step % gConfig.saveInterval == 0:
			print("%d training has saved" % step)
			tb.saveModel(gConfig.modelDir, step)

		