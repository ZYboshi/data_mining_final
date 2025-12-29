# data_process.py
"""
    数据处理：
    1. 获取数据集
    2. 对数据进行预处理
    3. 保存处理后的数据
"""
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
import os   
warnings.filterwarnings('ignore')

def download_dataset():
    """
    下载数据集（如果未下载）
    """
    import os
    from sklearn.datasets import fetch_openml
    
    # 创建 dataset 文件夹（如果不存在）
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
    
    # 下载数据集
    bank = fetch_openml(name='bank-marketing', version=1, as_frame=True)
    
    # 将数据保存为 CSV 文件
    data_path = './dataset/bank_marketing.csv'
    bank.frame.to_csv(data_path, index=False)
    
    print(f"Data saved to {data_path}")
    return data_path

def load_and_rename_data(file_path):
    """
    加载数据并重命名列
    
    参数:
    file_path: 数据文件路径
    
    返回:
    data: 处理后的DataFrame
    """
    # 加载数据
    data = pd.read_csv(file_path)
    print("数据形状:", data.shape)
    
    # 定义列名映射
    column_names = {
        'V1': 'age',
        'V2': 'job',
        'V3': 'marital',
        'V4': 'education',
        'V5': 'default',
        'V6': 'balance',
        'V7': 'housing',
        'V8': 'loan',
        'V9': 'contact',
        'V10': 'day',
        'V11': 'month',
        'V12': 'duration',
        'V13': 'campaign',
        'V14': 'pdays',
        'V15': 'previous',
        'V16': 'poutcome',
        'Class': 'deposit'
    }
    
    # 重命名列
    data = data.rename(columns=column_names)
    print("重命名后的列名:", list(data.columns))
    
    return data

def explore_data(data):
    """
    数据探索分析
    
    参数:
    data: 要探索的DataFrame
    
    返回:
    categorical_cols: 类别型列列表
    """
    # 基本信息
    print("\n数据类型:")
    print(data.dtypes)
    
    print("\n数据统计描述:")
    print(data.describe())
    
    # 类别型变量
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                        'loan', 'contact', 'month', 'poutcome', 'deposit']
    
    print("\n类别型变量统计:")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(data[col].value_counts())
    
    print("\n缺失值统计:")
    print(data.isnull().sum())
    
    return categorical_cols

def generate_column_info(data, save_path='./dataset/column_info.json'):
    """
    生成列信息并保存为JSON文件
    
    参数:
    data: 输入的DataFrame
    save_path: JSON文件保存路径
    """
    data_info = {
        "dataset_shape": {
            "rows": int(data.shape[0]),
            "columns": int(data.shape[1])
        },
        "columns": {}
    }
    
    # 为每一列收集信息
    for column in data.columns:
        col_type = str(data[column].dtype)
        
        # 数值型列
        if col_type in ['int64', 'float64']:
            data_info["columns"][column] = {
                "type": "numerical",
                "values": "numerical values"
            }
        
        # 类别型列
        else:
            unique_values = data[column].unique().tolist()
            unique_values_str = [str(val) for val in unique_values]
            
            data_info["columns"][column] = {
                "type": "categorical",
                "values": unique_values_str,
                "count": len(unique_values)
            }
    
    # 保存到JSON文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_info, f, indent=2, ensure_ascii=False)
    
    print(f"列信息已保存到: {save_path}")
    return data_info


def analyze_missing_data(df, missing_values=['unknown', 'Unknown', '?', '', 'NaN', 'N/A', None]):
    """
    专门分析数据集中的缺失值情况
    
    Args:
        df: pandas DataFrame
        missing_values: 常见的缺失值标记列表
        
    Returns:
        missing_report: 缺失值统计报告（字典）
    """
    
    missing_report = {
        "dataset_shape": {
            "rows": int(len(df)),
            "columns": int(len(df.columns))
        },
        "missing_analysis": {},
        "recommendations": []
    }
    
    total_rows = len(df)
    columns_with_missing = []
    
    for col in df.columns:
        col_analysis = {
            "column_name": col,
            "dtype": str(df[col].dtype),  # 确保是字符串类型
            "is_categorical": str(df[col].dtype) == 'object',  # 转换为字符串比较
            "total_missing": 0,
            "missing_percentage": 0.0,
            "missing_types": {}
        }
        
        # 1. 统计NaN值
        nan_count = int(df[col].isna().sum())  # 转为整数
        col_analysis["missing_types"]["NaN"] = {
            "count": nan_count,
            "percentage": round(float(nan_count) / float(total_rows) * 100, 2)
        }
        col_analysis["total_missing"] += nan_count
        
        # 2. 统计特殊标记的缺失值（如'unknown'等）
        if col_analysis["is_categorical"]:
            for missing_val in missing_values:
                if missing_val is not None and str(missing_val) in df[col].astype(str).values:
                    # 安全地统计缺失值
                    try:
                        count = int((df[col] == missing_val).sum())
                    except:
                        # 如果是None，使用isna()
                        count = int(df[col].isna().sum())
                    
                    col_analysis["missing_types"][str(missing_val)] = {
                        "count": count,
                        "percentage": round(float(count) / float(total_rows) * 100, 2)
                    }
                    col_analysis["total_missing"] += count
        
        # 计算总缺失比例
        col_analysis["missing_percentage"] = round(float(col_analysis["total_missing"]) / float(total_rows) * 100, 2)
        col_analysis["total_missing"] = int(col_analysis["total_missing"])  # 确保是整数
        
        if col_analysis["total_missing"] > 0:
            columns_with_missing.append(col)
            
            # 添加处理建议
            missing_pct = col_analysis["missing_percentage"]
            if missing_pct < 1:
                suggestion = f"{col}: 删除缺失行（缺失率<1%）"
            elif missing_pct < 10:
                suggestion = f"{col}: 用众数/平均数填充"
            else:
                suggestion = f"{col}: 需要建模填充或删除该列（缺失率≥10%）"
            
            missing_report["recommendations"].append(suggestion)
        
        missing_report["missing_analysis"][col] = col_analysis
    
    # 保存报告
    output_file = './dataset/missing_value_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(missing_report, f, indent=2, ensure_ascii=False)
    
    print("=" * 50)
    print(f"缺失值分析报告已保存到: {output_file}")
    print(f"包含缺失值的列数: {len(columns_with_missing)}/{len(df.columns)}")
    
    # 显示关键信息
    if columns_with_missing:
        print("\n关键缺失信息:")
        for col in columns_with_missing:
            info = missing_report["missing_analysis"][col]
            print(f"  {col}: {info['total_missing']}个缺失 ({info['missing_percentage']}%)")
    else:
        print("\n无缺失值!")
    
    return missing_report

def handle_missing_values(data):
    """
    处理缺失值（基于missing_value_analysis.json的建议）
    
    参数:
    data: 输入的DataFrame
    
    返回:
    data: 处理后的DataFrame
    """
    
    # job: 删除缺失行
    print("处理 job 列 (删除 'unknown' 值)...")
    original_rows = len(data)
    data = data[data['job'] != 'unknown'].copy()
    removed_job_rows = original_rows - len(data)
    print(f"  删除了 {removed_job_rows} 行 'unknown' 值")
    
    # education: 用众数填充
    print("处理 education 列 (用众数填充 'unknown')...")
    # 计算众数（排除unknown）
    education_mode = data[data['education'] != 'unknown']['education'].mode()
    if len(education_mode) > 0:
        edu_fill_value = education_mode[0]
        edu_missing_count = (data['education'] == 'unknown').sum()
        data['education'] = data['education'].replace('unknown', edu_fill_value)
        print(f"  将 {edu_missing_count} 个 'unknown' 替换为 '{edu_fill_value}'")
    else:
        print("  警告：无法找到合适的填充值")
    
    # contact: 保留 'unknown' 为单独类别
    print("处理 contact 列 (保留 'unknown' 为单独类别)...")
    contact_missing = (data['contact'] == 'unknown').sum()
    print(f"  保留 {contact_missing} 个 'unknown' 作为分类值")
    
    # poutcome: 删除该列
    print("处理 poutcome 列 (删除整列)...")
    if 'poutcome' in data.columns:
        data = data.drop('poutcome', axis=1)
        print(f"  已删除 poutcome 列")
    else:
        print("  该列不存在")
    
    # 显示处理结果
    print(f"\n处理前: {original_rows} 行, 17 列")
    print(f"处理后: {len(data)} 行, {len(data.columns)} 列")
    print(f"删除了 {original_rows - len(data)} 行数据")
    
    return data

def simple_label_encoding(data):
    """
    简单标签编码，直接转换所有类别列
    保存编码器和处理后的数据到dataset文件夹
    """
    import pickle
    import os
    
    categorical_cols = ['job', 'marital', 'education', 'default', 
                       'housing', 'loan', 'contact', 'month']
    
    # 创建编码器字典
    label_encoders = {}
    
    print("开始标签编码...")
    for col in categorical_cols:
        # 确保数据为字符串类型
        data[col] = data[col].astype(str)
        
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
        
        print(f"  {col}: 编码完成 ({len(le.classes_)}个类别)")
    
    # ==================== 保存到dataset文件夹 ====================
    dataset_dir = './dataset'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # 1. 保存编码后的数据为CSV
    processed_data_path = os.path.join(dataset_dir, 'data_label_encoded.csv')
    data.to_csv(processed_data_path, index=False)
    print(f"\n✅ 编码后的数据已保存到: {processed_data_path}")
    
    # 2. 保存编码器对象供后续使用
    encoders_path = os.path.join(dataset_dir, 'label_encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"✅ 标签编码器已保存到: {encoders_path}")
    
    # 3. 创建简单的映射信息文件（可选，用于查看）
    mapping_info = {}
    for col, le in label_encoders.items():
        mapping_info[col] = {
            'classes': le.classes_.tolist(),
            'indices': list(range(len(le.classes_)))
        }
    
    mapping_path = os.path.join(dataset_dir, 'encoding_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_info, f, indent=2, ensure_ascii=False)
    print(f"✅ 编码映射信息已保存到: {mapping_path}")
    
    # 显示前几行的编码示例
    print("\n📊 编码示例 (前3行):")
    sample_cols = ['job', 'education', 'contact']
    for col in sample_cols:
        print(f"  {col} 原始值 -> 编码值:")
        for i in range(3):
            original = label_encoders[col].inverse_transform([data[col].iloc[i]])[0]
            encoded = data[col].iloc[i]
            print(f"    第{i+1}行: '{original}' -> {encoded}")
    
    return data, label_encoders


def load_processed_data():
    """
    从dataset文件夹加载处理好的数据
    """
    import pickle
    
    dataset_dir = './dataset'
    
    # 加载编码后的数据
    data_path = os.path.join(dataset_dir, 'data_label_encoded.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到处理后的数据文件: {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"加载处理后的数据: {data.shape[0]}行 × {data.shape[1]}列")
    
    # 加载编码器
    encoders_path = os.path.join(dataset_dir, 'label_encoders.pkl')
    if os.path.exists(encoders_path):
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        print(f"加载标签编码器: {len(label_encoders)}个类别列")
    else:
        label_encoders = None
        print("警告: 未找到标签编码器文件")
    
    return data, label_encoders


def decode_columns(data, label_encoders):
    """
    将编码后的列解码回原始类别（仅用于查看）
    """
    if label_encoders is None:
        print("无法解码: 未找到编码器")
        return data
    
    categorical_cols = ['job', 'marital', 'education', 'default', 
                       'housing', 'loan', 'contact', 'month']
    
    decoded_data = data.copy()
    
    for col in categorical_cols:
        if col in label_encoders:
            decoded_data[f'{col}_decoded'] = label_encoders[col].inverse_transform(data[col])
    
    return decoded_data


def main():
    """
    主函数：执行完整数据处理流程
    """
    # 步骤1：加载数据
    print("步骤1: 加载数据...")
    data_path = './dataset/bank_marketing.csv'
    data = load_and_rename_data(data_path)
    
    # 步骤2：数据探索
    print("\n步骤2: 数据探索...")
    explore_data(data)
    
    # 步骤3：生成列信息
    print("\n步骤3: 生成列信息...")
    generate_column_info(data)
    
    # 步骤4：缺失值分析
    print("\n步骤4: 缺失值处理...")
    analyze_missing_data(data)
    
    # 步骤5：缺失值处理
    print("\n步骤5: 缺失值处理...")
    data = handle_missing_values(data)
    generate_column_info(data, save_path="./dataset/column_info_aftermissing.json")
    
    # 步骤6：标签编码
    print("\n步骤6: 标签编码...")
    data, label_encoders = simple_label_encoding(data)  # 现在会保存到dataset文件夹
    
    print("\n🎉 数据处理管道完成！")
    

    


    

def run_full_pipeline(include_download=False):
    """
    运行完整数据处理管道
    
    参数:
    include_download: 是否包含下载数据集步骤
    """
    if include_download:
        print("下载数据集...")
        download_dataset()
    
    # 运行主流程
    return main()

if __name__ == "__main__":
    # 运行完整管道（不包含下载，假设数据已存在）
    processed_data = run_full_pipeline(include_download=False)
    print("\n数据处理完成!")
