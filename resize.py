from PIL import Image

def resize_image(input_image_path, output_image_path, size=(448, 448)):
    # 打开图像文件
    with Image.open(input_image_path) as img:
        # 使用ANTIALIAS滤镜来保持图像质量
        img_resized = img.resize(size, Image.Resampling.LANCZOS)
        
        # 保存调整大小后的图像
        img_resized.save(output_image_path)

# 使用示例
input_image_path = 'C:\\Users\\Ccbbj\\Desktop\\origin\\origin.jpg'  # 替换为你的大图像文件路径
output_image_path = 'C:\\Users\\Ccbbj\\Desktop\\origin\\resized_image.jpg'  # 替换为你想要保存的文件路径和文件名
resize_image(input_image_path, output_image_path)