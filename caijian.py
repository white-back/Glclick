from PIL import Image
import os

def crop_images(source_folder, target_folder):
    # 创建目标文件夹如果它不存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 计数器，用于文件命名
    count = 1
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            # 打开图片
            img_path = os.path.join(source_folder, filename)
            with Image.open(img_path) as img:
                # 确保图片是1920*1080分辨率
                if img.size == (1920, 1080):
                    # 裁剪图片
                    for i in range(2):  # 两行
                        for j in range(4):  # 四列
                            # 计算裁剪区域
                            left = j * 448
                            top = i * 448
                            right = left + 448
                            bottom = top + 448
                            crop_area = (left, top, right, bottom)
                            # 裁剪并保存图片
                            cropped_img = img.crop(crop_area)
                            output_filename = f"{count:06d}.jpg"  # 格式化文件名为6位数字
                            cropped_img.save(os.path.join(target_folder, output_filename))
                            count += 1
                else:
                    print(f"图片{filename}不是1920*1080分辨率，已跳过。")

# 使用示例
source_folder = 'C:\\Users\Ccbbj\\Desktop\\aaa'  # 替换为你的源文件夹路径
target_folder = 'C:\\Users\\Ccbbj\\Desktop\\aaa\\bbb'  # 替换为你的目标文件夹路径
crop_images(source_folder, target_folder)