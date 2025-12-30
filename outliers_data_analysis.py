import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys

# 创建必要的目录
def create_directories():
    """创建保存结果所需的目录结构"""
    base_dir = './preprocess_dataset'
    outliers_dir = f'{base_dir}/outliers'
    
    # 创建目录
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(outliers_dir, exist_ok=True)
    
    print(f"✅ 已创建目录结构:")
    print(f"   - {base_dir}")
    print(f"   - {outliers_dir}")
    
    return outliers_dir

def load_data_and_info(dataset_dir='./dataset'):
    """加载数据和列信息"""
    try:
        # 构建完整的文件路径
        data_path = os.path.join(dataset_dir, 'bank_marketing_aftermissing.csv')
        info_path = os.path.join(dataset_dir, 'column_info_aftermissing.json')
        
        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在: {data_path}")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"列信息文件不存在: {info_path}")
        
        # 加载数据
        print(f"📁 正在加载数据集: {data_path}")
        data = pd.read_csv(data_path)
        
        # 加载列信息
        print(f"📁 正在加载列信息: {info_path}")
        with open(info_path, 'r') as f:
            column_info = json.load(f)
        
        print(f"✅ 数据加载成功!")
        print(f"   数据形状: {data.shape}")
        print(f"   列信息: {len(column_info['columns'])} 个特征")
        
        # 检查列名一致性
        data_columns = set(data.columns)
        info_columns = set(column_info['columns'].keys())
        
        if not data_columns.issubset(info_columns):
            missing_in_info = data_columns - info_columns
            if missing_in_info:
                print(f"⚠️  警告: 数据中的以下列在列信息中未找到: {missing_in_info}")
        
        return data, column_info
    
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("请检查文件路径或文件是否存在")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("❌ 错误: 数据集文件为空")
        sys.exit(1)
    except json.JSONDecodeError:
        print("❌ 错误: 列信息文件不是有效的JSON格式")
        sys.exit(1)

def detect_and_handle_outliers(data, column_info, outliers_dir):
    """检测和处理异常值"""
    
    print("\n" + "=" * 60)
    print("开始异常值检测与处理")
    print("=" * 60)
    
    # 创建副本，不修改原始数据
    data_clean = data.copy()
    outlier_report = {
        'processing_steps': {},
        'summary': {},
        'detailed_stats': {}
    }
    
    # 记录原始数据形状
    original_rows = data.shape[0]
    original_cols = data.shape[1]
    
    # ===============================
    # 1. 业务规则过滤（硬性规则）
    # ===============================
    
    print("\n📋 步骤1: 业务规则过滤")
    
    # 1.1 age: 年龄范围过滤（15-100岁）
    if 'age' in data_clean.columns:
        age_mask = data_clean['age'].between(15, 100)
        age_outliers = data_clean[~age_mask].shape[0]
        data_clean = data_clean[age_mask]
        outlier_report['processing_steps']['age_business_rule'] = {
            'type': 'business_rule',
            'condition': 'age between 15 and 100',
            'records_removed': age_outliers
        }
        print(f"  ✅ age: 删除 {age_outliers} 条记录（年龄<15或>100）")
    
    # 1.2 balance: 账户余额范围（避免极端值影响）
    if 'balance' in data_clean.columns:
        balance_mask = data_clean['balance'].between(-100000, 1000000)
        balance_outliers = data_clean[~balance_mask].shape[0]
        data_clean = data_clean[balance_mask]
        outlier_report['processing_steps']['balance_business_rule'] = {
            'type': 'business_rule',
            'condition': 'balance between -100,000 and 1,000,000',
            'records_removed': balance_outliers
        }
        print(f"  ✅ balance: 删除 {balance_outliers} 条记录（余额<-100,000或>1,000,000）")
    
    # 1.3 duration: 通话时长必须非负
    if 'duration' in data_clean.columns:
        duration_mask = data_clean['duration'] >= 0
        duration_outliers = data_clean[~duration_mask].shape[0]
        data_clean = data_clean[duration_mask]
        outlier_report['processing_steps']['duration_business_rule'] = {
            'type': 'business_rule',
            'condition': 'duration >= 0',
            'records_removed': duration_outliers
        }
        print(f"  ✅ duration: 删除 {duration_outliers} 条记录（通话时长<0）")
    
    # 1.4 campaign: 当前营销联系次数必须为正数
    if 'campaign' in data_clean.columns:
        campaign_mask = data_clean['campaign'] > 0
        campaign_outliers = data_clean[~campaign_mask].shape[0]
        data_clean = data_clean[campaign_mask]
        outlier_report['processing_steps']['campaign_business_rule'] = {
            'type': 'business_rule',
            'condition': 'campaign > 0',
            'records_removed': campaign_outliers
        }
        print(f"  ✅ campaign: 删除 {campaign_outliers} 条记录（营销次数≤0）")
    
    # 1.5 pdays: 上一次联系的天数（特殊值-1表示未联系过）
    if 'pdays' in data_clean.columns:
        # pdays的特殊情况：-1表示从未联系
        pdays_mask = data_clean['pdays'] >= -1
        pdays_outliers = data_clean[~pdays_mask].shape[0]
        data_clean = data_clean[pdays_mask]
        outlier_report['processing_steps']['pdays_business_rule'] = {
            'type': 'business_rule',
            'condition': 'pdays >= -1',
            'records_removed': pdays_outliers
        }
        print(f"  ✅ pdays: 删除 {pdays_outliers} 条记录（pdays < -1）")
    
    # 1.6 previous: 之前联系次数必须非负
    if 'previous' in data_clean.columns:
        previous_mask = data_clean['previous'] >= 0
        previous_outliers = data_clean[~previous_mask].shape[0]
        data_clean = data_clean[previous_mask]
        outlier_report['processing_steps']['previous_business_rule'] = {
            'type': 'business_rule',
            'condition': 'previous >= 0',
            'records_removed': previous_outliers
        }
        print(f"  ✅ previous: 删除 {previous_outliers} 条记录（之前联系次数<0）")
    
    # 1.7 day: 日期必须在1-31之间
    if 'day' in data_clean.columns:
        day_mask = data_clean['day'].between(1, 31)
        day_outliers = data_clean[~day_mask].shape[0]
        data_clean = data_clean[day_mask]
        outlier_report['processing_steps']['day_business_rule'] = {
            'type': 'business_rule',
            'condition': 'day between 1 and 31',
            'records_removed': day_outliers
        }
        print(f"  ✅ day: 删除 {day_outliers} 条记录（日期<1或>31）")
    
    # ===============================
    # 2. 统计截断处理（Winsorization）
    # ===============================
    
    print("\n📈 步骤2: 统计截断处理（温和处理）")
    # 选择需要进行缩尾处理的数值列
    numerical_cols_for_winsor = ['balance', 'duration', 'campaign', 'pdays', 'previous']
    
    for col in numerical_cols_for_winsor:
        if col in data_clean.columns:
            try:
                # 计算1%和99%分位数
                q1 = data_clean[col].quantile(0.01)
                q99 = data_clean[col].quantile(0.99)
                
                # 统计截断前的异常值数量
                before_outliers = data_clean[(data_clean[col] < q1) | (data_clean[col] > q99)].shape[0]
                
                if before_outliers > 0:
                    # 应用Winsorization：将极端值缩尾
                    clipped_col = np.clip(data_clean[col], q1, q99)
                    data_clean[col] = clipped_col
                    
                    outlier_report['processing_steps'][f'{col}_winsorization'] = {
                        'type': 'winsorization',
                        'lower_bound': float(q1),
                        'upper_bound': float(q99),
                        'records_affected': int(before_outliers)
                    }
                    print(f"  ✅ {col}: 缩尾处理 {before_outliers} 个极端值（1%-99%范围）")
                    print(f"     下界: {q1:.2f}, 上界: {q99:.2f}")
            except Exception as e:
                print(f"  ⚠️  {col}: 处理失败 - {e}")
    
    # ===============================
    # 3. 目标变量检查
    # ===============================
    
    print("\n🎯 步骤3: 目标变量检查")
    if 'deposit' in data_clean.columns:
        unique_values = sorted(data_clean['deposit'].unique())
        print(f"  deposit 的唯一值: {unique_values}")
        
        # 记录目标变量信息
        deposit_counts = data_clean['deposit'].value_counts().to_dict()
        outlier_report['target_variable'] = {
            'unique_values': [int(val) for val in unique_values],
            'value_counts': {int(k): int(v) for k, v in deposit_counts.items()}
        }
        
        print(f"  目标变量分布:")
        total = len(data_clean)
        for val, count in data_clean['deposit'].value_counts().items():
            percentage = count / total * 100
            print(f"    {val}: {count:,} ({percentage:.1f}%)")
    else:
        print("  ⚠️  未找到目标变量 'deposit'")
    
    # ===============================
    # 4. 类别变量检查
    # ===============================
    
    print("\n📊 步骤4: 类别变量检查")
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month']
    
    category_report = {}
    for col in categorical_cols:
        if col in data_clean.columns:
            if col in column_info['columns']:
                try:
                    expected_values = column_info['columns'][col]['values']
                    actual_values = list(data_clean[col].astype(str).unique())
                    
                    # 检查是否有意外值
                    unexpected = list(set(actual_values) - set(expected_values))
                    
                    category_report[col] = {
                        'expected_values': expected_values,
                        'actual_values': actual_values,
                        'unexpected_values': unexpected,
                        'has_unexpected': len(unexpected) > 0
                    }
                    
                    if unexpected:
                        print(f"  ⚠️  {col}: 发现意外值 {unexpected[:5]}")  # 只显示前5个
                    else:
                        print(f"  ✅ {col}: 所有值都在预期范围内")
                except Exception as e:
                    print(f"  ❌ {col}: 检查失败 - {e}")
            else:
                print(f"  ⚠️  {col}: 未在列信息中找到")
    
    outlier_report['categorical_check'] = category_report
    
    # ===============================
    # 5. 汇总统计信息
    # ===============================
    
    cleaned_rows = data_clean.shape[0]
    rows_removed = original_rows - cleaned_rows
    retention_rate = cleaned_rows / original_rows * 100
    
    # 记录详细的统计信息
    outlier_report['summary'] = {
        'original_rows': original_rows,
        'original_columns': original_cols,
        'cleaned_rows': cleaned_rows,
        'cleaned_columns': data_clean.shape[1],
        'rows_removed': rows_removed,
        'retention_rate': retention_rate,
        'removal_rate': 100 - retention_rate
    }
    
    # 数值特征的描述性统计
    numerical_cols = [col for col in data_clean.columns 
                     if column_info['columns'][col]['type'] == 'numerical' 
                     if col in column_info['columns']]
    
    descriptive_stats = {}
    for col in numerical_cols:
        if col in data_clean.columns:
            stats = data_clean[col].describe().to_dict()
            descriptive_stats[col] = {
                'mean': float(stats.get('mean', 0)),
                'std': float(stats.get('std', 0)),
                'min': float(stats.get('min', 0)),
                '25%': float(stats.get('25%', 0)),
                '50%': float(stats.get('50%', 0)),
                '75%': float(stats.get('75%', 0)),
                'max': float(stats.get('max', 0))
            }
    
    outlier_report['descriptive_statistics'] = descriptive_stats
    
    # ===============================
    # 6. 结果展示
    # ===============================
    
    print("\n" + "=" * 60)
    print("✅ 异常值处理完成！")
    print("=" * 60)
    print(f"📊 处理摘要:")
    print(f"   原始数据行数: {original_rows:,}")
    print(f"   处理后数据行数: {cleaned_rows:,}")
    print(f"   删除的行数: {rows_removed:,}")
    print(f"   数据保留比例: {retention_rate:.1f}%")
    print(f"   删除比例: {100 - retention_rate:.1f}%")
    print(f"   数据形状变化: {original_rows}×{original_cols} → {cleaned_rows}×{data_clean.shape[1]}")
    
    return data_clean, outlier_report

def visualize_outliers(data_before, data_after, outliers_dir):
    """可视化处理前后的异常值变化"""
    # 数值特征列表（仅显示有异常值的特征）
    numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    
    # 创建子图
    fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(16, 5*len(numerical_cols)))
    fig.suptitle('异常值处理前后对比', fontsize=16, y=1.02)
    
    for idx, col in enumerate(numerical_cols):
        if col not in data_before.columns or col not in data_after.columns:
            continue
        
        # 处理前的箱线图
        ax1 = axes[idx, 0]
        bp1 = ax1.boxplot(data_before[col].dropna(), vert=True, patch_artist=True)
        # 设置颜色
        bp1['boxes'][0].set_facecolor('lightcoral')
        ax1.set_title(f'{col} - 处理前', fontsize=12, fontweight='bold')
        ax1.set_ylabel('数值')
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_before = data_before[col].describe()
        text1 = f"均值: {stats_before['mean']:.2f}\n标准差: {stats_before['std']:.2f}\n异常值: {len([x for x in data_before[col] if x < stats_before['25%'] - 1.5*(stats_before['75%']-stats_before['25%']) or x > stats_before['75%'] + 1.5*(stats_before['75%']-stats_before['25%'])])}"
        ax1.text(0.02, 0.98, text1, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 处理后的箱线图  
        ax2 = axes[idx, 1]
        bp2 = ax2.boxplot(data_after[col].dropna(), vert=True, patch_artist=True)
        bp2['boxes'][0].set_facecolor('lightgreen')
        ax2.set_title(f'{col} - 处理后', fontsize=12, fontweight='bold')
        ax2.set_ylabel('数值')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_after = data_after[col].describe()
        text2 = f"均值: {stats_after['mean']:.2f}\n标准差: {stats_after['std']:.2f}\n异常值: {len([x for x in data_after[col] if x < stats_after['25%'] - 1.5*(stats_after['75%']-stats_after['25%']) or x > stats_after['75%'] + 1.5*(stats_after['75%']-stats_after['25%'])])}"
        ax2.text(0.02, 0.98, text2, transform=ax2.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(outliers_dir, 'outlier_handling_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 对比图已保存: {output_path}")
    plt.show()
    
    # 单独绘制balance（通常异常值最多）
    if 'balance' in data_before.columns and 'balance' in data_after.columns:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig2.suptitle('账户余额异常值处理对比', fontsize=14)
        
        # 处理前
        bp1 = ax1.boxplot(data_before['balance'].dropna(), vert=True, patch_artist=True)
        bp1['boxes'][0].set_facecolor('lightcoral')
        ax1.set_title('处理前', fontsize=12)
        ax1.set_ylabel('余额')
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_before = data_before['balance'].describe()
        outliers_before = len([x for x in data_before['balance'] 
                              if x < stats_before['25%'] - 1.5*(stats_before['75%']-stats_before['25%']) 
                              or x > stats_before['75%'] + 1.5*(stats_before['75%']-stats_before['25%'])])
        ax1.text(0.05, 0.95, f"异常值数量: {outliers_before:,}", 
                transform=ax1.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 处理后
        bp2 = ax2.boxplot(data_after['balance'].dropna(), vert=True, patch_artist=True)
        bp2['boxes'][0].set_facecolor('lightgreen')
        ax2.set_title('处理后', fontsize=12)
        ax2.set_ylabel('余额')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_after = data_after['balance'].describe()
        outliers_after = len([x for x in data_after['balance'] 
                             if x < stats_after['25%'] - 1.5*(stats_after['75%']-stats_after['25%']) 
                             or x > stats_after['75%'] + 1.5*(stats_after['75%']-stats_after['25%'])])
        ax2.text(0.05, 0.95, f"异常值数量: {outliers_after:,}", 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        output_path2 = os.path.join(outliers_dir, 'balance_outlier_comparison.png')
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"📈 Balance对比图已保存: {output_path2}")
        plt.show()

def save_results(data_clean, outlier_report, outliers_dir, column_info):
    """保存处理结果"""
    
    print("\n💾 开始保存处理结果...")
    
    # 1. 保存清理后的数据
    output_data_path = os.path.join(outliers_dir, 'bank_marketing_outliers_cleaned.csv')
    data_clean.to_csv(output_data_path, index=False)
    print(f"✅ 清理后的数据已保存: {output_data_path}")
    print(f"   文件大小: {os.path.getsize(output_data_path) / 1024:.1f} KB")
    
    # 2. 保存异常值处理报告（详细版）
    report_path = os.path.join(outliers_dir, 'outlier_handling_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(outlier_report, f, indent=2, ensure_ascii=False)
    print(f"✅ 详细处理报告已保存: {report_path}")
    
    # 3. 保存异常值处理报告（简版）
    summary_report = {
        '基本信息': {
            '原始数据行数': outlier_report['summary']['original_rows'],
            '处理后数据行数': outlier_report['summary']['cleaned_rows'],
            '删除行数': outlier_report['summary']['rows_removed'],
            '保留比例(%)': round(outlier_report['summary']['retention_rate'], 2),
            '删除比例(%)': round(outlier_report['summary']['removal_rate'], 2)
        },
        '处理步骤': {
            '业务规则过滤': {},
            '统计缩尾处理': {}
        }
    }
    
    # 从详细报告中提取关键信息到简版
    for key, value in outlier_report['processing_steps'].items():
        if 'business_rule' in key:
            col_name = key.replace('_business_rule', '')
            summary_report['处理步骤']['业务规则过滤'][col_name] = {
                '删除记录数': value['records_removed']
            }
        elif 'winsorization' in key:
            col_name = key.replace('_winsorization', '')
            summary_report['处理步骤']['统计缩尾处理'][col_name] = {
                '影响记录数': value['records_affected'],
                '下界': value['lower_bound'],
                '上界': value['upper_bound']
            }
    
    # 目标变量信息
    if 'target_variable' in outlier_report:
        summary_report['目标变量'] = outlier_report['target_variable']
    
    summary_path = os.path.join(outliers_dir, 'outlier_handling_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    print(f"✅ 简洁处理总结已保存: {summary_path}")
    
    # 4. 保存处理后的数据描述性统计
    descriptive_stats = outlier_report.get('descriptive_statistics', {})
    if descriptive_stats:
        stats_df = pd.DataFrame(descriptive_stats).T
        stats_csv_path = os.path.join(outliers_dir, 'descriptive_statistics_after_outliers.csv')
        stats_df.to_csv(stats_csv_path)
        print(f"✅ 描述性统计已保存: {stats_csv_path}")
    
    # 5. 生成文本报告
    txt_report_path = os.path.join(outliers_dir, 'outlier_handling_summary.txt')
    with open(txt_report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("             异常值处理总结报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. 数据概况\n")
        f.write("-" * 40 + "\n")
        f.write(f"原始数据形状: {outlier_report['summary']['original_rows']} 行 × {outlier_report['summary']['original_columns']} 列\n")
        f.write(f"处理后数据形状: {outlier_report['summary']['cleaned_rows']} 行 × {outlier_report['summary']['cleaned_columns']} 列\n")
        f.write(f"删除记录数: {outlier_report['summary']['rows_removed']}\n")
        f.write(f"数据保留率: {outlier_report['summary']['retention_rate']:.2f}%\n\n")
        
        f.write("2. 业务规则过滤结果\n")
        f.write("-" * 40 + "\n")
        for key, value in outlier_report['processing_steps'].items():
            if 'business_rule' in key:
                col_name = key.replace('_business_rule', '')
                f.write(f"  {col_name}: 删除 {value['records_removed']} 条记录\n")
        
        f.write("\n3. 统计缩尾处理结果\n")
        f.write("-" * 40 + "\n")
        for key, value in outlier_report['processing_steps'].items():
            if 'winsorization' in key:
                col_name = key.replace('_winsorization', '')
                f.write(f"  {col_name}: 处理 {value['records_affected']} 个极端值 ({value['lower_bound']:.2f} - {value['upper_bound']:.2f})\n")
        
        f.write("\n4. 目标变量分布\n")
        f.write("-" * 40 + "\n")
        if 'target_variable' in outlier_report:
            total = sum(outlier_report['target_variable']['value_counts'].values())
            for val, count in outlier_report['target_variable']['value_counts'].items():
                percentage = count / total * 100
                f.write(f"  {val}: {count} ({percentage:.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("处理完成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ 文本报告已保存: {txt_report_path}")
    
    return {
        'cleaned_data_path': output_data_path,
        'detailed_report_path': report_path,
        'summary_report_path': summary_path,
        'text_report_path': txt_report_path
    }

def generate_readme(outliers_dir, file_paths):
    """生成README文件，说明处理过程和结果"""
    readme_path = os.path.join(outliers_dir, 'README.md')
    
    readme_content = """# 异常值处理记录

## 📋 处理概述
本文件夹包含了银行营销数据集的异常值检测与处理结果。

## 🚀 处理方法
采用了两种主要的异常值处理策略：

### 1. 业务规则过滤
基于业务知识设定合理的值域范围：
- **年龄 (age)**: 15-100岁
- **余额 (balance)**: -100,000 ~ 1,000,000
- **通话时长 (duration)**: ≥0
- **当前联系次数 (campaign)**: >0
- **上次联系天数 (pdays)**: ≥-1 (-1表示未联系过)
- **之前联系次数 (previous)**: ≥0
- **日期 (day)**: 1-31

超出上述范围的记录被直接删除。

### 2. 统计缩尾处理（Winsorization）
对以下特征的极端值进行缩尾处理：
- balance, duration, campaign, pdays, previous

处理方式：将小于1%分位数和大于99%分位数的值截断到相应边界。

## 📁 文件说明

### 主要文件
| 文件名称 | 说明 |
|----------|------|
| `bank_marketing_outliers_cleaned.csv` | 处理后的完整数据集 |
| `outlier_handling_report.json` | 详细的处理报告（JSON格式） |
| `outlier_handling_summary.json` | 简洁的处理总结报告 |
| `outlier_handling_summary.txt` | 文本格式的处理总结 |

### 可视化文件
| 文件名称 | 说明 |
|----------|------|
| `outlier_handling_comparison.png` | 所有数值特征处理前后对比图 |
| `balance_outlier_comparison.png` | 账户余额的详细对比图 |

### 统计文件
| 文件名称 | 说明 |
|----------|------|
| `descriptive_statistics_after_outliers.csv` | 处理后数值特征的描述性统计 |

## 🔧 使用说明
1. 主要分析数据：使用 `bank_marketing_outliers_cleaned.csv`
2. 查看处理详情：查看 `outlier_handling_summary.json` 或 `outlier_handling_summary.txt`
3. 可视化结果：查看 `.png` 格式的对比图
4. 如需复现处理过程，参考详细报告 `outlier_handling_report.json`

## 📊 关键指标
处理前后的关键指标对比可参考文本报告或JSON总结文件。

---

*生成时间：""" + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "*"

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📘 README文件已生成: {readme_path}")
    return readme_path

def show_final_data_preview(data_clean, column_info):
    """显示最终数据预览"""
    print("\n" + "=" * 60)
    print("📊 最终数据集预览")
    print("=" * 60)
    
    # 显示基本信息
    print(f"数据形状: {data_clean.shape[0]:,} 行 × {data_clean.shape[1]} 列")
    print(f"内存使用: {data_clean.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # 显示列信息
    print("\n数据列:")
    for i, col in enumerate(data_clean.columns, 1):
        col_type = column_info['columns'][col]['type'] if col in column_info['columns'] else 'unknown'
        unique_count = data_clean[col].nunique()
        print(f"  {i:2d}. {col:<15} ({col_type:<12}) - {unique_count} 个唯一值")
    
    # 显示前几行数据
    print("\n前5行数据:")
    print(data_clean.head())
    
    # 显示基本统计信息
    print("\n数值特征的描述性统计:")
    numerical_cols = [col for col in data_clean.columns 
                     if column_info['columns'][col]['type'] == 'numerical' 
                     if col in column_info['columns']]
    
    if numerical_cols:
        stats_df = data_clean[numerical_cols].describe().round(2)
        print(stats_df)

# ===============================
# 主程序：执行异常值处理
# ===============================
if __name__ == "__main__":
    print("🚀 开始运行异常值检测与处理程序")
    print("=" * 60)
    try:
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    try:
        # 1. 创建目录
        print("\n📂 步骤1: 创建输出目录")
        outliers_dir = create_directories()
        
        # 2. 加载数据
        print("\n📂 步骤2: 加载数据")
        data, column_info = load_data_and_info()
        
        # 3. 备份原始数据用于可视化对比
        data_before = data.copy()
        
        # 4. 检测和处理异常值
        print("\n🔧 步骤3: 异常值检测与处理")
        data_clean, outlier_report = detect_and_handle_outliers(data, column_info, outliers_dir)
        
        # 5. 可视化比较
        print("\n📊 步骤4: 生成可视化对比")
        visualize_outliers(data_before, data_clean, outliers_dir)
        
        # 6. 保存结果
        print("\n💾 步骤5: 保存处理结果")
        file_paths = save_results(data_clean, outlier_report, outliers_dir, column_info)
        
        # 7. 生成README
        print("\n📘 步骤6: 生成说明文档")
        readme_path = generate_readme(outliers_dir, file_paths)
        
        # 8. 显示最终数据预览
        print("\n👀 步骤7: 显示最终数据预览")
        show_final_data_preview(data_clean, column_info)
        
        # 9. 完成
        print("\n" + "=" * 60)
        print("🎉 异常值处理流程完成！")
        print("=" * 60)
        print(f"\n📁 所有结果已保存在: {outliers_dir}")
        print(f"\n📋 生成的文件:")
        for file_name in os.listdir(outliers_dir):
            file_path = os.path.join(outliers_dir, file_name)
            file_size = os.path.getsize(file_path) / 1024
            print(f"  • {file_name:<40} ({file_size:.1f} KB)")
        
        print(f"\n✅ 流程完成！清理后的数据已准备好用于后续分析。")
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
