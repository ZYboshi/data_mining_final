# 检查数值即可
import pandas as pd

dataset_path = "./dataset/data_label_encoded.csv"
data = pd.read_csv(dataset_path)
print(f"✅ 数据加载完成，形状: {data.shape}")
print(f"   行数: {data.shape[0]}")
print(f"   列数: {data.shape[1]}")

# 2. 查看列名和数据类型
print("\n" + "="*50)
print("步骤2: 查看数据结构和类型")
print("列名和数据类型:")
print(data.dtypes)

# 3. 检查空值
print("\n" + "="*50)
print("步骤3: 检查空值")
null_counts = data.isnull().sum()
null_columns = null_counts[null_counts > 0]
if len(null_columns) == 0:
    print("✅ 所有列都没有空值！")
else:
    print("⚠️ 有以下列存在空值:")
    for col, count in null_columns.items():
        print(f"   {col}: {count}个空值 ({count/len(data)*100:.2f}%)")



