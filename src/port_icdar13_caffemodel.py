import caffe
import numpy as np
import os

preTrainedDir = os.path.join('..','model','pretrained')
modelProto = os.path.join(preTrainedDir, 'deploy.prototxt')
modelWeights = os.path.join(preTrainedDir, 'TextBoxes_icdar13.caffemodel')
outFile = os.path.join(preTrainedDir, 'TextBoxes_icdar13_train.npy')

def convKernel2tfFormat(kernel):
	return np.transpose(kernel,(2, 3, 1, 0))

def concateKernelPrediction(kernel1, kernel2):
	return np.concatenate((kernel1, kernel2), axis=3)

def concateBiasPrediction(bias1, bias2):
	return np.concatenate((bias1, bias2), axis=0)

caffe.set_mode_cpu()
net = caffe.Net(modelProto, modelWeights, caffe.TRAIN)
param = {}
for k, v in net.params.items():
	layer_param = {}
	vNum = len(v)
	if vNum == 2:
		print "layer: %s, weights: %s, biases: %s" % (k, v[0].data.shape, v[1].data.shape)
		_w = convKernel2tfFormat(v[0].data)
		_b = v[1].data
		print "--> (tf)layer: %s, weights: %s, biases: %s" % (k, _w.shape, _b.shape)
		layer_param['weights'] = _w
		layer_param['biases'] = _b
	elif vNum == 1:
		print "layer: %s, gamma: %s" % (k, v[0].data.shape)
		layer_param[k+'_gamma'] = v[0].data
	param[k] = layer_param
output = np.asarray(param)
np.save(outFile, output)
