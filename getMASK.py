import cv2
import numpy as np

swt_img = r'Stroke_Width_Transformed.png'
img = cv2.imread(swt_img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
mask = np.zeros(img.shape[:2], dtype=np.uint8)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY) # <= 1 -> 0, > 1 -> 255
cv2.imshow('thresh', thresh)
# findcontours需要输入二值化图片
# 创建一个膨胀的核（kernel）
kernel = np.ones((10, 10), np.uint8)  # 核的大小可以根据需要调整
# 在填充后的掩码图像上执行膨胀操作
thresh = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilated_thresh', thresh)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED) # filled会填充内部
cv2.imshow('mask', mask)

# # 创建一个膨胀的核（kernel）
# kernel = np.ones((5, 5), np.uint8)  # 核的大小可以根据需要调整
# # 在填充后的掩码图像上执行膨胀操作
# dilated_mask = cv2.dilate(mask, kernel, iterations=1)
# cv2.imshow('dilated_mask', dilated_mask)
cv2.waitKey(0)