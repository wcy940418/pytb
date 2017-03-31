import lmdb
import os
import numpy
import json
import sys
import cv2

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
		randomIndex = random.sample(range(len(datalist)), batches)
		images = []
		anns = []
		imageList = []
		bbList = []
		with self.env.begin() as txn:
			for i in range(batchSize):
				idx = randomIndex[i]
				imageKey = 'img-%08d' % idx
				bbKey = 'bb-%08d' % idx
				imagePath = txn.get(imageKey)
				bb = json.loads(txn.get(bbKey))
				imageList.append(imagePath)
				bbList.append(bb)
		for imagePath, bb in zip(imageList, bbList):
			filePath = os.path.join(self.dataDirPath, imagePath)
			img = cv2.imread(filePath, cv2.IMREAD_COLOR)
			resized = cv2.resize(img, (imgW, imgH))
			resized = np.multiply(resized, 1.0/255.0)
			images.append(resized)
			anns.append(bb)
		images = np.asarray(images)
		return (images, anns)

if __name__ == '__main__':
	db  = SynthLmdb("../data/SynthTextLmdb", "../data/SynthText")
	batches, anns = db.nextBatch(2)
	print  batches.shape, anns