import xml.etree.ElementTree as ET
import cv2
import numpy as np
import random
import os
#All feeded data should be in [x,y,w,h] format(left top based corner box)
class SVT:
	def __init__(self, trainPath=None, testPath=None):
		trainList = None
		testList = None
		if trainPath:
			self.trainList = self.parseTree(trainPath)
		if testPath:
			self.testList = self.parseTree(testPath)
		self.path = os.path.dirname(trainPath)

	def parseTree(self, path):
		dataset = []
		tree = ET.parse(path)
		root = tree.getroot()
		for image in root.findall('image'):
			name = image.find('imageName').text
			rectangles = []
			taggedRectangles = image.find('taggedRectangles')
			resolution = image.find('Resolution')
			width = float(resolution.get('x'))
			height = float(resolution.get('y'))
			for rectangle in taggedRectangles.findall('taggedRectangle'):
				h = float(rectangle.get('height')) / height
				w = float(rectangle.get('width')) / width
				x = float(rectangle.get('x')) / width
				y = float(rectangle.get('y')) / height
				rectangles.append(([x,y,w,h], 0))
			dataset.append((name, rectangles))
		return dataset

	def nextBatch(self, batches, dataset='train'):
		imgH = 300
		imgW = 300
		if dataset == 'train':
			datalist = self.trainList
		if dataset == 'test':
			datalist = self.testList
		randomIndex = random.sample(range(len(datalist)), batches)
		images = []
		anns = []
		for index in randomIndex:
			fileName = os.path.join(self.path, datalist[index][0])
			img = cv2.imread(fileName, cv2.IMREAD_COLOR)
			resized = cv2.resize(img, (imgW, imgH))
			resized = np.multiply(resized, 1.0/255.0)
			images.append(resized)
			anns.append(datalist[index][1])
		images = np.asarray(images)
		return (images, anns)

if __name__ == '__main__':
	loader = SVT('./svt1/train.xml', './svt1/test.xml')
	train_img, train_anns = loader.nextBatch(5,'test')
	for img, anns in zip(train_img, train_anns):
		print anns
		cv2.imshow('output', img)
		cv2.waitKey(0)

