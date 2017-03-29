from matcher import Matcher
import draw
import constants as c
import svt_data_loader
import boxes
import cv2
import numpy as np

svt = svt_data_loader.SVT('./svt1/train.xml', './svt1/test.xml')
c.out_shapes = [[8, 38, 38, 72], [8, 19, 19, 72], [8, 10, 10, 72], [8, 5, 5, 72], [8, 3, 3, 72], [8, 1, 1, 72]]
c.defaults = boxes.default_boxes(c.out_shapes)

box_matcher = Matcher()

imgs, anns = svt.nextBatch(1)

cv2.imshow("original", (imgs[0] * 255.0).astype(np.uint8))
cv2.waitKey(1000)

matches = box_matcher.match_boxes([], anns[0])

draw.draw_matches(imgs[0], c.defaults, matches, anns[0], wait=0)