import cv2

# 鼠标回调函数，用于获取点击位置的坐标和颜色值
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击事件
        # 打印坐标
        print(f"Coordinates: ({x}, {y})")
        # 获取并打印BGR颜色值
        bgr_color = img[y, x]
        print(f"BGR values: {bgr_color}")
        # 获取RGB颜色值（因为OpenCV默认使用BGR格式）
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        print(f"RGB values: {rgb_color}")

# 读取图像
img = cv2.imread('C:\\Users\\Ccbbj\\Desktop\\origin\\tile_1_0.png')  # 替换为您的图像路径

# 检查图像是否正确加载
if img is None:
    print("Error: Image not found.")
else:
    # 创建窗口
    cv2.namedWindow('Image')
    # 设置鼠标回调函数
    cv2.setMouseCallback('Image', click_event)

    # 显示图像
    cv2.imshow('Image', img)
    print("Click on the image to get the coordinates and RGB values. Press 'ESC' to exit.")

    # 等待直到ESC键被按下
    while True:
        if cv2.waitKey(20) == 27:  # 27是ESC键的ASCII码
            break

    # 关闭所有窗口
    cv2.destroyAllWindows()