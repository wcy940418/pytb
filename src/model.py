import tensorflow as tf
import numpy as np
import basic_layer as bl
import time
from constants import *
import os

class TB:
	def __init__(self, images, trainPhase, sess):
		self.input = images
		self.trainPhase = trainPhase
		self.sess = sess
		self.parameters = []
		print("build model started")
		start_time = time.time()
		self.initialVGGLayers()
		self.initialTextBoxLayers()
		self.initialOutputs()
		print("build model finished: %ds" % (time.time() - start_time))

	def initialVGGLayers(self):
		with tf.variable_scope('VGG16') as scope:
			with tf.name_scope('preprocess') as scope:
				images = tf.reshape(self.input, [-1, 300, 300, 3])
				images = images * 255.0
				mean = tf.constant([-103.939, -116.779, -123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
				self.imgs = tf.add(images, mean)
				tf.summary.image('input image', self.imgs, 3)
			self.conv1_1 = bl.conv2d(self.imgs, 3, 64, name='conv1_1', trainable=False, parameters=self.parameters)#300
			self.conv1_2 = bl.conv2d(self.conv1_1, 64, 64, name='conv1_2', trainable=False, parameters=self.parameters)#300
			self.pool1 = bl.maxPool(self.conv1_2, name='pool1')#150
			self.conv2_1 = bl.conv2d(self.pool1, 64, 128, name='conv2_1', trainable=False, parameters=self.parameters)#150
			self.conv2_2 = bl.conv2d(self.conv2_1, 128, 128, name='conv2_2', trainable=False, parameters=self.parameters)#150
			self.pool2 = bl.maxPool(self.conv2_2, name='pool2')#75
			self.conv3_1 = bl.conv2d(self.pool2, 128, 256, name='conv3_1', parameters=self.parameters)#75
			self.conv3_2 = bl.conv2d(self.conv3_1, 256, 256, name='conv3_2', parameters=self.parameters)#75
			self.conv3_3 = bl.conv2d(self.conv3_2, 256, 256, name='conv3_3', parameters=self.parameters)#75
			self.pool3 = bl.maxPool(self.conv3_3, name='pool3')#38
			self.conv4_1 = bl.conv2d(self.pool3, 256, 512, name='conv4_1', parameters=self.parameters)#38
			self.conv4_2 = bl.conv2d(self.conv4_1, 512, 512, name='conv4_2', parameters=self.parameters)#38
			self.conv4_3 = bl.conv2d(self.conv4_2, 512, 512, name='conv4_3', parameters=self.parameters)#38
			self.pool4 = bl.maxPool(self.conv4_3, name='pool4')#19
			self.conv5_1 = bl.conv2d(self.pool4, 512, 512, name='conv5_1', parameters=self.parameters)#19
			self.conv5_2 = bl.conv2d(self.conv5_1, 512, 512, name='conv5_2', parameters=self.parameters)#19
			self.conv5_3 = bl.conv2d(self.conv5_2, 512, 512, name='conv5_3', parameters=self.parameters)#19
			self.pool5 = bl.maxPool(self.conv5_3, stride=1, kernel=3, name='pool5')#19
			self.fc6 = bl.conv2d(self.pool5, 512, 1024, name='fc6', parameters=self.parameters)#19
			self.fc7 = bl.conv2d(self.fc6, 1024, 1024, kernel=[1,1], name='fc7', parameters=self.parameters)#19
			self.conv6_1 = bl.conv2d(self.fc7, 1024, 256, kernel=[1,1], name='conv6_1', parameters=self.parameters)#19
			self.conv6_2 = bl.conv2d(self.conv6_1, 256, 512, strides = [2, 2], name='conv6_2', parameters=self.parameters)#10
			self.conv7_1 = bl.conv2d(self.conv6_2, 512, 128, kernel=[1,1], name='conv7_1', parameters=self.parameters)#10
			self.conv7_2 = bl.conv2d(self.conv7_1, 128, 256, strides=[2,2], name='conv7_2', parameters=self.parameters)#5
			self.conv8_1 = bl.conv2d(self.conv7_2, 256, 128, kernel=[1,1], name='conv8_1', parameters=self.parameters)#5
			self.conv8_2 = bl.conv2d(self.conv8_1, 128, 256, strides=[2,2], name='conv8_2', parameters=self.parameters)#3
			globalPoolingH = self.conv8_2.get_shape().as_list()[1]
			globalPoolingW = self.conv8_2.get_shape().as_list()[2]
			globalPoolingSize = [globalPoolingH, globalPoolingW]
			self.pool6 = tf.layers.average_pooling2d(self.conv8_2, globalPoolingSize, globalPoolingSize, "valid")#1
	def initialTextBoxLayers(self):
		with tf.variable_scope('tb_extension') as scope:
			c_ = classes + 1
			self.conv4_3_norm = tf.layers.batch_normalization(self.conv4_3, center=True, training=self.trainPhase)
			self.conv4_3_norm_mbox_loc = bl.conv2d(self.conv4_3_norm, 512, layer_boxes[0] * c_ , kernel=[1,5], name='conv4_3_norm_mbox_loc', parameters=self.parameters)
			self.fc7_mbox_loc = bl.conv2d(self.fc7, 1024, layer_boxes[1] * c_ , kernel=[1,5], name='fc7_mbox_loc', parameters=self.parameters)
			self.conv6_2_mbox_loc = bl.conv2d(self.conv6_2, 512, layer_boxes[2] * c_ , kernel=[1,5], name='conv6_2_mbox_loc', parameters=self.parameters)
			self.conv7_2_mbox_loc = bl.conv2d(self.conv7_2, 256, layer_boxes[3] * c_ , kernel=[1,5], name='conv7_2_mbox_loc', parameters=self.parameters)
			self.conv8_2_mbox_loc = bl.conv2d(self.conv8_2, 256, layer_boxes[4] * c_ , kernel=[1,5], name='conv8_2_mbox_loc', parameters=self.parameters)
			self.pool6_mbox_loc = bl.conv2d(self.pool6, 256, layer_boxes[5] * c_ , kernel=[1,1], name='pool6_mbox_loc', parameters=self.parameters)
			self.out1 = bl.conv2d(self.conv4_3_norm, 512, layer_boxes[0] * 4, kernel=[1,5], name='out1', parameters=self.parameters)
			self.out2 = bl.conv2d(self.fc7, 1024, layer_boxes[1] * 4, kernel=[1,5], name='out2', parameters=self.parameters)
			self.out3 = bl.conv2d(self.conv6_2, 512, layer_boxes[2] * 4, kernel=[1,5], name='out3', parameters=self.parameters)
			self.out4 = bl.conv2d(self.conv7_2, 256, layer_boxes[3] * 4, kernel=[1,5], name='out4', parameters=self.parameters)
			self.out5 = bl.conv2d(self.conv8_2, 256, layer_boxes[4] * 4, kernel=[1,5], name='out5', parameters=self.parameters)
			self.out6 = bl.conv2d(self.pool6, 256, layer_boxes[5] * 4, kernel=[1,1], name='out6', parameters=self.parameters)
	def initialOutputs(self):
		c_ = classes + 1
		outputs = [self.out1, self.out2, self.out3, self.out4, self.out5, self.out6]
		outlist = []
		for i, out in zip(range(len(outputs)), outputs):
			w = out.get_shape().as_list()[2]
			h = out.get_shape().as_list()[1]
			# out1 = tf.transpose(out, perm=[0, 2, 1, 3])
			out_reshaped = tf.reshape(out, [-1,w*h*layer_boxes[i], c_ + 4])
			outlist.append(out_reshaped)
		formatted_outs = tf.concat(outlist, 1)

		pred_labels = formatted_outs[:, :, 4:]
		pred_locs = formatted_outs[:, :, :4]
		self.outputs = outputs
		self.pred_labels = pred_labels
		self.pred_locs = pred_locs
		self.softmaxed_pred_labels = tf.nn.softmax(pred_labels)
		self.softmaxed_pred_labels_max_prob = tf.reduce_max(self.softmaxed_pred_labels, axis=-1)
		self.softmaxed_pred_labels_max_index = tf.argmax(self.softmaxed_pred_labels, axis=-1)
	def loadModel(self, modelFile):
		saver = tf.train.Saver()
		saver.restore(self.sess, modelFile)
		print("Model restored")
	def saveModel(self, modelFile, step):
		saver = tf.train.Saver()
		save_path = os.path.join(modelFile, "ckpt-%08d" % step)
		if not os.path.isdir(save_path):
			os.mkdir(save_path)
		savePath = saver.save(self.sess, os.path.join(save_path, "ckpt-%08d" % step))
		print("Model saved at: %s" % savePath)
		return savePath
	# def loadWeights(self, weightFile):
	# 	weights = np.load(weightFile)
	# 	keys = sorted(weights.keys())
	# 	for i in range(26):
	# 		k = keys[i]
	# 		print i, k, np.shape(weights[k])
	# 		self.sess.run(self.parameters[i].assign(weights[k]))
	def loadWeights(self, weightFile):
		weights = np.load(weightFile)
		weights = weights.item()
		layer_list = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', \
					  'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', \
					  'conv5_3']
		for i, layer in zip(range(len(layer_list)), layer_list):
			weight = weights[layer]['weights']
			bias = weights[layer]['biases']
			print layer, weight.shape, bias.shape
			self.sess.run(self.parameters[i * 2].assign(weight))
			self.sess.run(self.parameters[i * 2 + 1].assign(bias))


class TB_Loss():
	def __init__(self, pred_labels, pred_locs, true_labels, true_locs, positives, negatives):
		self.pred_labels = pred_labels
		self.pred_locs = pred_locs
		self.true_labels = true_labels
		self.true_locs = true_locs
		self.positives = positives
		self.negatives = negatives
		posandnegs = tf.add(self.positives, self.negatives)
		positive_sum = tf.reduce_sum(self.positives, reduction_indices=1)
		class_loss = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_labels, labels=self.true_labels), posandnegs)
		class_loss = tf.reduce_sum(class_loss, reduction_indices=1) / (1e-5 + tf.reduce_sum(posandnegs, reduction_indices=1))
		loc_loss = tf.multiply(tf.reduce_sum(smooth_l1(self.pred_locs - self.true_locs), reduction_indices=2), positives)
		loc_loss = tf.reduce_sum(loc_loss, reduction_indices=1) / (1e-5 + tf.reduce_sum(positives, reduction_indices=1))
		total_loss = class_loss + 1.0 * loc_loss
		condition = tf.equal(positive_sum, 0)
		total_loss_wo_inf = tf.where(condition, positive_sum, total_loss)
		self.total_loss = tf.reduce_mean(total_loss_wo_inf)
		tf.summary.scalar('loss', self.total_loss)



def smooth_l1(x):
	l2 = 0.5 * (x**2.0)
	l1 = tf.abs(x) - 0.5

	condition = tf.less(tf.abs(x), 1.0)
	re = tf.where(condition, l2, l1)

	return re