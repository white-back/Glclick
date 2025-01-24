import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image = cv2.imread('C:\\Users\\Ccbbj\\Desktop\\origin\\tile_1_0.png', cv2.IMREAD_GRAYSCALE)  # 确保路径正确

# 检查图像是否正确加载
if image is None:
    print("Error: Image not found.")
else:
    # 使用高斯模糊减少图像噪声
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # 使用Canny算法进行边缘检测
    edges = cv2.Canny(blurred_image, 50, 150)  # 50和150是Canny算法的阈值参数

    # 显示原图和边缘检测后的图像
    plt.figure(figsize=(10, 5))

    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()