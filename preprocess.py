import os
import glob
import pandas as pd
import numpy as np

# 配置路径
DATA_DIR = '/root/autodl-tmp/battery/data'
SAVE_PATH = '/root/autodl-tmp/battery/processed_data.feather'


def run_preprocess():
    files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
    if not files:
        print("❌ 错误：在指定目录下未找到 .xlsx 文件！")
        return

    all_dfs = []
    print(f"开始转换 {len(files)} 个文件...", flush=True)

    for f in files:
        print(f"正在读取文件: {os.path.basename(f)}", flush=True)
        try:
            # 1. 打开 Excel 对象
            xls = pd.ExcelFile(f, engine='calamine')

            # 2. 遍历 Sheet
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)

                # 3. 强力清洗列名 (去除空格，转字符串)
                df.columns = [str(c).strip() for c in df.columns]

                # 4. 彻底解决重复列名问题 (InvalidIndexError)
                if len(df.columns) != len(set(df.columns)):
                    new_cols = []
                    counts = {}
                    for col in df.columns:
                        if col in counts:
                            counts[col] += 1
                            new_cols.append(f"{col}_{counts[col]}")
                        else:
                            counts[col] = 0
                            new_cols.append(col)
                    df.columns = new_cols

                all_dfs.append(df)

        except Exception as e:
            print(f"  ❌ 读取文件 {os.path.basename(f)} 出错: {e}")

    if not all_dfs:
        print("❌ 错误：没有成功加载任何数据。")
        return

    print("正在合并所有数据中 (这可能需要一分钟)...", flush=True)
    full_df = pd.concat(all_dfs, ignore_index=True)

    print(f"正在保存为极速 Feather 格式...", flush=True)
    full_df.to_feather(SAVE_PATH)
    print(f"✅ 转换全部完成！")
    print(f"文件位置: {SAVE_PATH}")
    print(f"总数据量: {len(full_df)} 行")


if __name__ == "__main__":
    run_preprocess()