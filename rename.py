import os

# 指定包含图片的文件夹路径
folder_path = 'C:\\Users\\Ccbbj\\Desktop\\biaotu\\biaotu\\2'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件名是否符合'maskxxx'格式
    if filename.startswith('mask') and filename.endswith('.png'):
        # 提取数字部分
        number = filename[4:-4]
        # 生成新的文件名，格式为'mask_000xxx.png'
        new_filename = f'mask_000{number}.png'
        # 构建原始文件和新文件的完整路径
        original_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        # 重命名文件
        os.rename(original_file_path, new_file_path)
        print(f'Renamed "{filename}" to "{new_filename}"')