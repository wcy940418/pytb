import scipy.io as sio
import lmdb
import os
import numpy
import json
import sys
import cv2

def checkImageIsValid(imagePath):
	if imagePath is None:
		return False
	img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
	if img is None:
		return False
	imgH, imgW = img.shape[0], img.shape[1]
	if imgH * imgW == 0:
		return False
	return True

def writeCache(env, cache):
	with env.begin(write=True) as txn:
		for k, v in cache.iteritems():
			txn.put(k, v)

def bbProcess(bb):
	x_top_left = bb[0][0]
	x_bottom_right = bb[0][2]
	y_top_left = bb[1][0]
	y_bottom_right = bb[1][2]
	box = []
	if len(x_top_left.shape) == 0 and len(x_bottom_right.shape) == 0 and len(y_top_left.shape) == 0 and len(y_bottom_right.shape) == 0:
		return box
	assert x_top_left.shape[0] == x_bottom_right.shape[0] and x_top_left.shape[0] == y_top_left.shape[0] and x_top_left.shape[0] == y_bottom_right.shape[0]
	for x1, y1, x2, y2 in zip(x_top_left, y_top_left, x_bottom_right, y_bottom_right):
		x = max(int(x1), 0)
		y = max(int(y1), 0)
		w = max(int(x2 - x1), 0)
		h = max(int(y2 - y1), 0)
		box.append(([x, y, w, h], 0))
	return box


def createDataset(outputPath, configFile, imgDir):
	"""
	Create LMDB dataset for CRNN training.

	ARGS:
		outputPath    : LMDB output path
		imagePathList : list of image path
		labelList     : list of corresponding groundtruth texts
		lexiconList   : (optional) list of lexicon lists
		checkValid    : if true, check the validity of every image
	"""
	env = lmdb.open(outputPath, map_size=1099511627776)
	cache = {}
	cnt = 0
	data = sio.loadmat(configFile)
	print "mat file loaded"
	n = data['imnames'][0].shape[0]
	img_path = data['imnames'][0]
	wordBB = data['wordBB'][0]
	for i in range(n):
		image_path = img_path[i][0]
		img = os.path.join(imgDir, image_path)
		if True:
			path_key = "img-%08d" % cnt
			cache[path_key] = str(image_path)
			bb_key = "bb-%08d" % cnt
			cache[bb_key] = json.dumps(bbProcess(wordBB[i]))
			cnt += 1
			if cnt % 10000 == 0 and cnt != 0:
				writeCache(env, cache)
				cache = {}
				print "Written %d images" % cnt
	nSamples = cnt
	cache['num-samples'] = str(nSamples)
	writeCache(env, cache)
	print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
	'''ro run create_dataset, use command: 
	"python create_dataset.py <config text file path> <image files dir path>" '''
	configFile = sys.argv[1]
	imgDir = sys.argv[2]
	outputPath = './Synth_data'
	createDataset(outputPath, configFile, imgDir)