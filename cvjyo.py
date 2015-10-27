import cv2
import numpy as np
import sys
import math

def markPoints(pts, img):
    for pt in pts:
        cv2.circle(img, tuple((pt[0], pt[1])), 2, 0, -1)

def contourAreaComparator(cnt1, cnt2):
	if cv2.contourArea(cnt1) > cv2.contourArea(cnt2):
		return 1
	else:
		return -1

img = cv2.imread(sys.argv[1], 0)
img = cv2.GaussianBlur(img, (5, 5), 0)
height,width = img.shape

_,otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow("img", otsu); cv2.waitKey(0);

imgAnd = cv2.bitwise_and(img, otsu)
# cv2.imshow("img", imgAnd); cv2.waitKey(0);

contours, hierarchy = cv2.findContours(otsu, 1, 2)

area = []
for cnt in contours:
    area.append(cv2.contourArea(cnt))

area = np.array(area)
idx = np.max(area)
idx = np.where(area==idx)[0][0]
cnt = contours[idx]

hull = cv2.convexHull(cnt, returnPoints = False)
defects = cv2.convexityDefects(cnt, hull)

for d in defects:
    s, e, f, appr = d[0]
    cv2.circle(imgAnd, tuple(cnt[f][0]), 2, 255, -1)

dt = cv2.distanceTransform(otsu, cv2.cv.CV_DIST_L2, 3)
cv2.normalize(dt, dt, 0.0, 1.0, cv2.NORM_MINMAX);
# cv2.imshow("img", dt);cv2.waitKey(0)

idx = np.where(dt==np.max(dt))
pt = (idx[0][0], idx[1][0])

defPts = cnt[defects[:, 0, 2]]
defPts = defPts.reshape(-1,2)

thrDistTop = int(0.4*height)
defPts = defPts[np.where(defPts[:, 1] > thrDistTop)[0]]

# markPoints(defPts, img)
# cv2.imshow("img", img); cv2.waitKey(0)

boundImg = np.zeros((height,width), np.uint8)
cv2.fillPoly(boundImg, [defPts], 255)
imgRoi = cv2.bitwise_and(img, boundImg)
imgRoi = cv2.adaptiveThreshold(imgRoi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
kernel = np.ones((5,5),np.uint8)
boundImg = cv2.erode(boundImg,kernel,iterations = 1)
imgRoi = cv2.bitwise_and(imgRoi, boundImg)
# cv2.imshow("img", imgRoi); cv2.waitKey(0)

imgRoiC = imgRoi.copy()
contours, hierarchy = cv2.findContours(imgRoiC, 1, 2)

contours.sort(contourAreaComparator)
l = len(contours)
ll = np.arange(l-6, l-1)

for idx in ll:
	cv2.drawContours(imgRoi, contours, idx, 127, 3)
	cv2.drawContours(img, contours, idx, 0, 3)

cv2.imshow("img", img); cv2.waitKey(0)