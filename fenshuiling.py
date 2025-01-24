import cv2
import numpy as np

# 读取图像
img = cv2.imread('C:\\Users\\Ccbbj\\Desktop\\origin\\origin.jpg')

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 噪声去除
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 确定背景区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 寻找前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# 寻找未知区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 标记类别
ret, markers = cv2.connectedComponents(sure_fg)

# 为所有的标记加1，保证背景是0而不是1
markers = markers + 1

# 现在让所有的未知区域为0
markers[unknown == 255] = 0

# 应用分水岭算法
markers = cv2.watershed(img, markers)

# 标记边界
img[markers == -1] = [255, 0, 0]

# 显示结果
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()