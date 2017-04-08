import lmdb
import os
import numpy as np
import json
import sys
import cv2
import random

class SynthLmdb:
	def __init__(self, lmdbPath,dataDirPath):
		self.env = lmdb.open(lmdbPath, map_size=1099511627776)
		with self.env.begin() as txn:
			self.nSamples = int(txn.get('num-samples'))
		self.dataDirPath = dataDirPath

	def getNumSamples(self):
		return self.nSamples

	def nextBatch(self, batches):
		imgH = 300
		imgW = 300
		randomIndex = random.sample(range(self.nSamples), batches)
		images = []
		anns = []
		imageList = []
		bbList = []
		with self.env.begin() as txn:
			for i in range(batches):
				idx = randomIndex[i]
				imageKey = 'img-%08d' % idx
				bbKey = 'bb-%08d' % idx
				imagePath = txn.get(imageKey)
				bb = json.loads(txn.get(bbKey))
				imageList.append(imagePath)
				bbList.append(bb)
		for imagePath, bb in zip(imageList, bbList):
			box_list = []
			filePath = os.path.join(self.dataDirPath, imagePath)
			img = cv2.imread(filePath, cv2.IMREAD_COLOR)
			height, width = img.shape[0], img.shape[1]
			resized = cv2.resize(img, (imgW, imgH))
			resized = np.multiply(resized, 1.0/255.0)
			images.append(resized)
			for box in bb:
				coord = box[0]
				x = float(coord[0]) / width
				y = float(coord[1]) / height
				w = float(coord[2]) / width
				h = float(coord[3]) / height
				box_list.append(([x, y, w, h], box[1]))
			anns.append(box_list)
		images = np.asarray(images)
		return (images, anns)

if __name__ == '__main__':
	db  = SynthLmdb("../data/SynthTextLmdb", "../data/SynthText")
	batches, anns = db.nextBatch(2)
	print  batches.shape, anns