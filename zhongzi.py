import cv2
import numpy as np
from matplotlib import pyplot as plt

# 全局变量来存储种子点
seed_points = []

# 鼠标回调函数来记录种子点
def select_seed(event, x, y, flags, param):
    global seed_points
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        seed_points.append((x, y))
        print(f"Seed point added: ({x}, {y})")

def region_growing(img, seed_points, threshold=3):
    # 将图像转换为浮点数，以便进行更精确的操作
    img_float = np.float32(img)

    # 创建一个与图像大小相同的mask，用于标记已访问的像素
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # 创建一个与图像大小相同的输出图像
    output = np.zeros(img.shape, dtype=np.uint8)

    # 定义颜色阈值
    for seed_point in seed_points:
        seeds = [seed_point]

        while len(seeds) > 0:
            # 取出一个种子点
            point = seeds.pop(0)
            
            # 标记该点已访问
            mask[point[1], point[0]] = 255
            
            # 将该点的颜色值赋给输出图像
            output[point[1], point[0]] = 255

            # 遍历8个邻域点
            for x in range(-1, 2):
                for y in range(-1, 2):
                    # 计算邻域点的坐标
                    x_adj = point[0] + x
                    y_adj = point[1] + y

                    # 检查邻域点是否在图像范围内
                    if x_adj >= 0 and x_adj < img.shape[1] and y_adj >= 0 and y_adj < img.shape[0]:
                        # 检查邻域点是否未被访问
                        if mask[y_adj, x_adj] == 0:
                            # 计算颜色差的绝对值
                            diff = np.sqrt(np.sum((img_float[y_adj, x_adj] - img_float[point[1], point[0]]) ** 2))

                            # 如果颜色差小于阈值，则将邻域点添加到种子点列表
                            if diff < threshold:
                                seeds.append((x_adj, y_adj))
                                mask[y_adj, x_adj] = 255
                                output[y_adj, x_adj] = 255

    return output

# 读取图像
image = cv2.imread('C:\\Users\\Ccbbj\\Desktop\\origin\\tile_1_0.png')  # 确保路径正确，并且读取彩色图像

# 检查图像是否正确加载
if image is None:
    print("Error: Image not found.")
else:
    # 显示图像并等待用户选择种子点
    cv2.namedWindow('Select Seed Points')
    cv2.setMouseCallback('Select Seed Points', select_seed)
    cv2.imshow('Select Seed Points', image)
    print("Click on the image to select seed points. Press 's' to start segmentation and 'q' to quit.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 用户按下's'键开始分割
            break
        elif key == ord('q'):  # 用户按下'q'键退出
            cv2.destroyAllWindows()
            exit()

    # 应用区域生长算法
    # 注意：对于彩色图像，我们需要对每个通道分别进行区域生长
    segmented_image = np.zeros_like(image)
    for channel in range(3):
        channel_image = image[:, :, channel]
        segmented_channel = region_growing(channel_image, seed_points)
        segmented_image[:, :, channel] = segmented_channel

    # 显示原图和分割后的图像
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])

    plt.show()