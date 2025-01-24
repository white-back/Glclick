import cv2
import numpy as np

# 全局变量
seed_point = None
seed_color = None
threshold = 40  # 预设阈值为30
segmented_mask = None

# 鼠标回调函数，用于获取点击位置的坐标和颜色值
def click_event(event, x, y, flags, param):
    global seed_point, seed_color, segmented_mask
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击事件
        seed_point = (x, y)
        seed_color = img[y, x]  # 获取种子点的BGR颜色值
        print(f"Seed point: ({x}, {y})")
        print(f"Seed color (BGR): {seed_color}")
        
        # 将BGR颜色值转换为RGB
        seed_color_rgb = (seed_color[2], seed_color[1], seed_color[0])
        print(f"Seed color (RGB): {seed_color_rgb}")
        
        # 根据种子点和阈值进行分割
        segmented_mask = segment_image(seed_color_rgb)
        cv2.imshow('Segmented Mask', segmented_mask)

# 根据种子点和阈值进行分割
def segment_image(seed_color_rgb):
    global img, threshold
    # 创建一个与原图大小相同的空白图像，用于显示分割掩码
    mask = np.zeros_like(img)
    # 计算RGB差值
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.all(np.abs(img[i, j].astype(int) - seed_color_rgb) < [threshold, threshold, threshold]):
                mask[i, j] = [255]  # 前景设置为白色
            else:
                mask[i, j] = [0]  # 背景设置为黑色
    return mask

# 读取图像
img = cv2.imread('C:\\Users\\Ccbbj\\Desktop\\fenge-plus\\tile_1_3.png')  # 替换为您的图像路径
if img is None:
    print("Error: Image not found.")
    exit()

# 创建窗口
cv2.namedWindow('Image')
cv2.namedWindow('Segmented Mask')

# 设置鼠标回调函数
cv2.setMouseCallback('Image', click_event)

# 显示图像
cv2.imshow('Image', img)
print("Click on the image to select a seed point. The program will then segment the image and display the mask.")

# 等待直到鼠标点击
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存分割掩码
if segmented_mask is not None:
    cv2.imwrite('C:\\Users\\Ccbbj\\Desktop\\fenge-plus\\tile_1_3_masko.png', segmented_mask)  # 将掩码乘以255以保存为可见的二值图像
    print("Segmented mask saved as 'segmented_mask.png'")
else:
    print("No mask was generated.")