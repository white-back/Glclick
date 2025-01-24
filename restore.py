from PIL import Image

def restore_image(input_image_path, output_image_path, original_size):
    # 打开调整大小后的图像文件
    with Image.open(input_image_path) as img:
        # 使用Resampling.LANCZOS滤镜来保持图像质量
        img_restored = img.resize(original_size, Image.Resampling.LANCZOS)
        
        # 保存恢复后的图像
        img_restored.save(output_image_path)

# 使用示例
input_image_path = 'C:\\Users\\Ccbbj\\Desktop\\mask.png'  # 替换为你调整大小后的图像文件路径
output_image_path = 'C:\\Users\\Ccbbj\\Desktop\\origin\\restored_image1.jpg'  # 替换为你想要保存的文件路径和文件名
original_size = (1920, 1080)  # 替换为原始图像的宽度和高度
restore_image(input_image_path, output_image_path, original_size)