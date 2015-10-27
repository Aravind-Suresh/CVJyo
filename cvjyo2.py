cd cvg/CVJyo
python cvjyo.py hand-1/hand.jpg
run cvjyo.py hand-1/hand.jpg
run cvjyo.py hand-1/hand.jpg
np.mean(dt)
mean = np.mean(dt)
def markPoints(pts, img):
    for pt in pts:
        cv2.circle(img, tuple(pt[0], pt[1]), 2, 0, -1)
ptsT = np.where(img > mean)
ptsT
markPoints(ptsT, img)
markPoints(ptsT.T, img)
ptsT
np.transpose(ptsT)
markPoints(np.transpose(ptsT), img)
markPoints(np.transpose(ptsT), img)
markPoints(np.transpose(ptsT[0]), img)
markPoints(np.transpose(ptsT), img)
ptsT
np.transpose(ptsT)
markPoints(np.transpose(ptsT)[0], img)
markPoints(np.transpose(ptsT), img)
np.transpose(ptsT)
for pts in np.transpose(ptsT):
    print pt[0], pt[1]
for pt in np.transpose(ptsT):
    print pt[0], pt[1]
for pt in np.transpose(ptsT):
    print pt[0], pt[1]
for pt in np.transpose(ptsT):
    print pt[0]
print a
print "a"
for pt in np.transpose(ptsT):
    print pt[0]
print
print "hello"
print("hello")
for pt in np.transpose(ptsT):
    print(pt[0], pt[1])
markPoints(np.transpose(ptsT), img)
for pt in np.transpose(ptsT):
    print(tuple(pt[0], pt[1]))
def markPoints(pts, img):
    for pt in pts:
        cv2.circle(img, tuple((pt[0], pt[1])), 2, 0, -1)
markPoints(np.transpose(ptsT), img)
cv2.imshow("otsu", img);cv2.waitKey(0)
ptsT = np.where(img > 1.2*mean)
img = cv2.imread("hand.png", 0)
ptsT = np.where(img > 1.2*mean)
cv2.imshow("otsu", img);cv2.waitKey(0)
cv2.imshow("otsu", img);cv2.waitKey(0)
ptsT
markPoints(np.transpose(ptsT), img)
cv2.imshow("otsu", img);cv2.waitKey(0)
img = cv2.imread("hand.png", 0)
np.max               ^
SyntaxError: invalid syntax
np.max(dt)
idx = np.where(dt==np.max(dt))
idx
idx = np.where(dt==np.max(dt))[0][0]
idx
idx = np.where(dt==np.max(dt))[0]
idx
idx = np.where(dt==np.max(dt))
idx
tuple(idx)
idx = np.where(dt==np.max(dt))[0, 0]
idx
idx = np.where(dt==np.max(dt))
idx
idx = (idx[0][0], idx[1][0])
idx
defects
defects[:, 2]
defects[:]
defects[:, 0]
defects[:, 0, 2]
cnt[defects[:, 0, 2]]
defPts = cnt[defects[:, 0, 2]]
defPts[0]
defPts.reshape(-1,2)
defPts = defPts.reshape(-1,2)
arrD = np.abs(defPts - pt)
arrD
arrD2 = []
for dd in arrD:
    arrD2.append(dd[0]*dd[0] + dd[1]*dd[1])
arrD2
idx2 = np.where(arrD2==np.min(arrD2))[0]
idx2
idx2 = idx2[0]
idx2
rad = sqrt(np.min(arrD2))
import math
rad = math.sqrt(np.min(arrD2))
rad
%history -f cvjyo2.py
