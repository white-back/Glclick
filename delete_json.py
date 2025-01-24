import os
import json

def delete_json_files(folder_path):
    # 检查文件夹路径是否存在
    if not os.path.exists(folder_path):
        print("指定的文件夹路径不存在。")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                # 删除文件
                os.remove(file_path)
                print(f"已删除文件：{file_path}")
            except Exception as e:
                print(f"删除文件时出错：{e}")

# 使用示例
folder_path = 'D:\\浏览器下载\\曹妃甸海草床标注\\曹妃甸海草床标注'  # 替换为你的文件夹路径
delete_json_files(folder_path)