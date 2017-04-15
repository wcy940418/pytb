import tensorflow as tf
import numpy as np

def weightVariable(shape):
	
	# initial = tf.truncated_normal(shape, stddev=0.1, name='weights')
	# return tf.Variable(initial)
	return tf.get_variable('weights', shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def biasVariable(shape):
	initial = tf.constant(0.0, shape=shape, name='biases')
	return tf.Variable(initial)

def maxPool(input, stride=2, kernel=2, padding='SAME', name='pool'):
	return tf.nn.max_pool(input, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def conv2d(input, inputNum, outputNum, kernel=[3, 3], strides=[1, 1], padding='SAME', bn=False, trainPhase=True, name='conv2d', freeze=False):
	with tf.variable_scope(name) as scope:		
		_W = weightVariable([kernel[0], kernel[1], inputNum, outputNum])
		_b = biasVariable([outputNum])
		if freeze:
			W = tf.stop_gradient(_W)
			b = tf.stop_gradient(_b)
		else:
			W = _W
			b = _b
		conv_out = tf.nn.conv2d(input, W, strides=[1, strides[0], strides[1], 1], padding=padding)
		biased_out = tf.nn.bias_add(conv_out, b)
		out = tf.nn.relu(biased_out)
		if bn:
			out = tf.layers.batch_normalization(out, center=True, training=trainPhase)
		return out
