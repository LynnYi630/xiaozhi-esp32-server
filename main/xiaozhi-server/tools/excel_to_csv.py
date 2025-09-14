import os
import glob
import pandas as pd

def excel_to_csv(excel_file: str):
    """
    将 Excel 文件转换为 CSV 文件。
    例如，将 data/excel/sample.xlsx 转为 data/csv/sample.csv。
    """
    # 提取 Excel 文件的基名，例如 "sample.xlsx"
    base_filename = os.path.basename(excel_file)
    # 分离文件名和扩展名，例如 ("sample", ".xlsx")
    base, _ = os.path.splitext(base_filename)
    
    # 构造 CSV 输出目录（data/csv），若不存在则创建
    output_dir = os.path.join("data", "csv")
    os.makedirs(output_dir, exist_ok=True)
    
    # 构造 CSV 文件完整路径，例如 "data/csv/sample.csv"
    csv_file = os.path.join(output_dir, base + ".csv")
    
    # 读取 Excel 文件，默认读取第一个工作表
    df = pd.read_excel(excel_file)
    # 保存为 CSV 文件，不保存行索引，编码为 UTF-8
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"已将 {excel_file} 转换为 {csv_file}")

if __name__ == '__main__':
    # 查找 data/excel 目录下所有 Excel 文件（支持 .xls 和 .xlsx）
    file_paths = glob.glob("data/excel/*.xls*")
    for file_path in file_paths:
        excel_to_csv(file_path)
