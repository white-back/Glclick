import cv2
import numpy as np

# 读取图像
img = cv2.imread('C:\\Users\\Ccbbj\\Desktop\\origin\\tile_1_0.png')

# 创建一个和图像大小相同的mask，初始化为0（背景）
mask = np.zeros(img.shape[:2], np.uint8)

# 创建两个数组，用于GrabCut算法的背景和前景模型
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 定义一个矩形区域，包含前景物体
rect = (x, y, w, h)  # x, y是矩形左上角坐标，w和h是矩形的宽和高

# 应用GrabCut算法
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 将mask中的可能前景和确定背景的区域标记为0和1
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# 根据mask得到最终的前景图像
img = img * mask2[:, :, np.newaxis]

# 显示结果
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()