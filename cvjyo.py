import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1], 0)
_,otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("otsu", otsu); cv2.waitKey(0);

img_and = cv2.bitwise_and(img, otsu)
cv2.imshow("otsu", img_and); cv2.waitKey(0);

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS
_, contours, hierarchy = cv2.findContours(otsu, 1, 2)

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
    cv2.circle(img_and, tuple(cnt[f][0]), 2, 255, -1)

dt = cv2.distanceTransform(otsu, cv2.DIST_L2, 3)
cv2.normalize(dt, dt, 0.0, 1.0, cv2.NORM_MINMAX);
cv2.imshow("otsu", dt);cv2.waitKey(0)