#!/usr/bin/env python3

import os
import cv2
import imutils
import numpy as np
from skimage import measure
from imutils import contours


def find_defect(img):
    """Finds and marks defects of a bearing in a given image."""
    w, h = img.shape[1], img.shape[0]
    width = 960
    height = int(h * (width / w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    orig = bgr.copy()
    blured = cv2.GaussianBlur(img, (11, 11), 0)
    thresh = cv2.threshold(blured, 100, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=10)

    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1.5, 200)
    circles = np.round(circles[0, :]).astype('int')
    bc = circles[0]
    cx, cy, cr = bc
    cv2.circle(bgr, (cx, cy), cr, (255, 0, 0), 2)

    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype='uint8')
    for label in np.unique(labels):
        if label == 0:
            continue

        label_mask = np.zeros(thresh.shape, dtype='uint8')
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)

        if 50000 > num_pixels > 2000:
            mask = cv2.add(mask, label_mask)

    conts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = conts[0] if imutils.is_cv2() else conts[1]
    conts = contours.sort_contours(conts)[0]

    rs, xys = [], []
    for i, c in enumerate(conts):
        (x, y), r = cv2.minEnclosingCircle(c)
        x, y, r = int(x), int(y), int(r)
        rs.append(r)
        xys.append((x, y))

    avg_r = sum(rs) / len(rs)
    for i in range(len(rs)):
        x, y = xys[i][0], xys[i][1]
        if x > cx:
            lx = x - cx
        else:
            lx = cx - x
        if y > cy:
            ly = y - cy
        else:
            ly = cy - y
        cath = int((lx ** 2 + ly ** 2) ** 0.5)
        if cath > cr * 0.5:
            if rs[i] > avg_r * 1.5:
                cv2.line(bgr, (cx, cy), (x, y), (0, 0, 255), 2)
                cv2.circle(bgr, xys[i], rs[i], (0, 0, 255), 2)
            else:
                cv2.line(bgr, (cx, cy), (x, y), (0, 255, 0), 2)
                cv2.circle(bgr, xys[i], rs[i], (0, 255, 0), 2)

    return orig, bgr


def main():
    data_dir = 'data/'
    files = os.listdir(data_dir)
    files.sort()
    files = [data_dir + f for f in files if f.endswith('.jpg')]
    for filename in files:
        img = cv2.imread(filename, 0)
        orig, img = find_defect(img)
        img = np.hstack((orig, img))
        cv2.imshow('Press "b" to break', img)
        # cv2.imwrite('output.jpg', img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('b'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
