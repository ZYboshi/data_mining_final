# basic_analysis_final.py
"""
基础统计分析 - 最终版
所有结果存储在 ./preprocess_dataset/basic_data_analysis 文件夹
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

class BasicDataAnalysis:
    def __init__(self):
        """初始化分析类，设置输出路径"""
        # 设置主输出文件夹
        self.output_dir = './preprocess_dataset/basic_data_analysis'
        
        # 创建文件夹结构
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'tables'), exist_ok=True)
        
        print(f"📁 所有分析结果将保存到: {self.output_dir}")
    
    def load_data(self):
        """加载数据集"""
        # 查找数据文件的优先级
        possible_paths = [
            './preprocess_dataset/bank_marketing_renamed.csv',  # 优先使用预处理文件
            './dataset/bank_marketing_renamed.csv',
            './dataset/bank_marketing.csv'
        ]
        
        data = None
        data_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ 找到数据文件: {path}")
                data_path = path
                break
        
        if data_path is None:
            print("❌ 错误: 未找到任何数据文件")
            print("请先运行 data_process.py 或确保有以下文件之一:")
            for path in possible_paths:
                print(f"  - {path}")
            return None
        
        # 加载数据
        data = pd.read_csv(data_path)
        print(f"✅ 数据加载完成: {data.shape[0]}行 × {data.shape[1]}列")
        
        # 如果加载的是原始文件，可能需要重命名
        if 'bank_marketing.csv' in data_path:
            print("检测到原始数据，尝试重命名列...")
            
            # 查找目标列
            if 'Class' in data.columns:
                target_col_name = 'Class'
            else:
                target_col_name = 'deposit'
            
            # 创建列名映射
            column_mapping = {}
            for i, col in enumerate(data.columns):
                if col == target_col_name:
                    column_mapping[col] = 'deposit'
                elif col.startswith('V'):
                    num = int(col[1:]) if col[1:].isdigit() else i
                    standard_names = [
                        'age', 'job', 'marital', 'education', 'default',
                        'balance', 'housing', 'loan', 'contact', 'day',
                        'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'
                    ]
                    if num <= len(standard_names):
                        column_mapping[col] = standard_names[num-1]
                else:
                    column_mapping[col] = col
            
            # 应用重命名
            data = data.rename(columns=column_mapping)
        
        return data
    
    def get_basic_stats(self, data):
        """获取基础统计信息"""
        print("\n" + "="*50)
        print("📊 基础统计信息")
        print("="*50)
        
        # 1. 数据集规模
        stats = {
            "dataset_info": {
                "samples": int(data.shape[0]),
                "features": int(data.shape[1]),
                "columns": list(data.columns)
            }
        }
        
        print(f"📦 数据集规模: {stats['dataset_info']['samples']:,} 样本 × {stats['dataset_info']['features']} 特征")
        
        # 2. 数据类型分布
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        stats["data_type_distribution"] = {
            "numeric": {
                "count": len(numeric_cols),
                "percentage": round(len(numeric_cols) / data.shape[1] * 100, 1),
                "features": numeric_cols
            },
            "categorical": {
                "count": len(categorical_cols),
                "percentage": round(len(categorical_cols) / data.shape[1] * 100, 1),
                "features": categorical_cols
            }
        }
        
        print(f"🎨 数据类型分布:")
        print(f"  - 数值型: {stats['data_type_distribution']['numeric']['count']}个 "
              f"({stats['data_type_distribution']['numeric']['percentage']}%)")
        print(f"  - 类别型: {stats['data_type_distribution']['categorical']['count']}个 "
              f"({stats['data_type_distribution']['categorical']['percentage']}%)")
        
        # 3. 目标变量分布
        target_candidates = ['deposit', 'Class', 'class', 'target', 'y']
        target_col = None
        
        for candidate in target_candidates:
            if candidate in data.columns:
                target_col = candidate
                break
        
        if target_col:
            stats["target_info"] = {
                "column_name": target_col,
                "data_type": str(data[target_col].dtype)
            }
            
            counts = data[target_col].value_counts()
            percentages = (data[target_col].value_counts(normalize=True) * 100).round(2)
            
            stats["target_distribution"] = {}
            for val in counts.index:
                stats["target_distribution"][str(val)] = {
                    "count": int(counts[val]),
                    "percentage": float(percentages[val]),
                    "label": str(val)
                }
            
            print(f"\n🎯 目标变量 '{target_col}' 分布:")
            for val, cnt in counts.items():
                pct = percentages[val]
                print(f"  - {val}: {cnt:,} 个 ({pct}%)")
            
            # 计算不平衡比例（如果是二分类）
            if len(counts) == 2:
                cnt_values = counts.values
                if cnt_values[0] > 0 and cnt_values[1] > 0:
                    ratio = max(cnt_values) / min(cnt_values)
                    stats["target_info"]["imbalance_ratio"] = float(ratio.round(2))
                    print(f"  ⚠️  类别不平衡比例: {ratio:.2f}:1")
        else:
            print(f"\n⚠️  警告: 未找到目标变量列")
            stats["target_info"] = {"found": False}
        
        # 保存统计结果为JSON
        stats_path = os.path.join(self.output_dir, 'tables', 'basic_statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 统计结果已保存到: {stats_path}")
        
        return stats
    
    def create_feature_type_chart(self, data):
        """创建特征类型分布图表"""
        numeric_count = len(data.select_dtypes(include=[np.number]).columns)
        categorical_count = len(data.select_dtypes(include=['object']).columns)
        
        # 创建饼图
        plt.figure(figsize=(10, 8))
        
        sizes = [numeric_count, categorical_count]
        labels = [f'数值型 ({numeric_count}个)', f'类别型 ({categorical_count}个)']
        colors = ['#FF9999', '#66B3FF']
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                colors=colors, startangle=90, explode=(0.05, 0))
        
        plt.title('数据集特征类型分布', fontsize=16, fontweight='bold', pad=20)
        
        # 保存图表
        chart_path = os.path.join(self.output_dir, 'figures', 'feature_type_distribution.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 特征类型分布图已保存到: {chart_path}")
    
    def create_target_distribution_chart(self, data):
        """创建目标变量分布图表"""
        # 查找目标列
        target_col = None
        for col in ['deposit', 'Class', 'class']:
            if col in data.columns:
                target_col = col
                break
        
        if not target_col:
            print("⚠️  未找到目标变量，跳过图表生成")
            return
        
        counts = data[target_col].value_counts()
        
        # 创建条形图
        plt.figure(figsize=(10, 6))
        
        x_pos = np.arange(len(counts))
        colors = ['#4ECDC4', '#FF6B6B', '#95E1D3', '#F38181'][:len(counts)]
        
        bars = plt.bar(x_pos, counts.values, color=colors, alpha=0.8, edgecolor='black')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, counts.values)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + max(counts.values)*0.01,
                    f'{value:,}\n({value/sum(counts.values)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=11)
        
        plt.xticks(x_pos, counts.index.astype(str), fontsize=12)
        plt.title(f'目标变量 "{target_col}" 分布', fontsize=16, fontweight='bold')
        plt.ylabel('样本数量', fontsize=14)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 保存图表
        chart_path = os.path.join(self.output_dir, 'figures', 'target_variable_distribution.png')
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 目标变量分布图已保存到: {chart_path}")
    
    def generate_statistics_tables(self, data):
        """生成统计表格"""
        print("\n" + "-"*50)
        print("📋 生成统计表格")
        print("-"*50)
        
        # 1. 数值特征统计表
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            numeric_stats = data[numeric_cols].describe().round(2)
            numeric_stats.loc['missing'] = data[numeric_cols].isnull().sum()
            numeric_stats.loc['missing_pct'] = (data[numeric_cols].isnull().sum() / len(data) * 100).round(2)
            
            # 保存为CSV
            numeric_path = os.path.join(self.output_dir, 'tables', 'numeric_features_statistics.csv')
            numeric_stats.to_csv(numeric_path)
            
            print(f"📊 数值特征统计表:")
            print(f"  共 {len(numeric_cols)} 个数值特征")
            print(f"  已保存到: {numeric_path}")
        
        # 2. 类别特征统计表
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            cat_stats = []
            for col in categorical_cols:
                unique_vals = data[col].unique()
                missing_count = data[col].isnull().sum()
                
                if len(data[col].mode()) > 0:
                    top_value = data[col].mode().iloc[0]
                    top_freq = (data[col] == top_value).sum()
                    top_pct = (top_freq / len(data) * 100).round(2)
                else:
                    top_value = 'N/A'
                    top_freq = 0
                    top_pct = 0
                
                cat_stats.append({
                    'feature': col,
                    'unique_values': len(unique_vals),
                    'missing': missing_count,
                    'missing_pct': (missing_count / len(data) * 100).round(2),
                    'most_common': str(top_value),
                    'most_common_count': top_freq,
                    'most_common_pct': top_pct
                })
            
            # 保存为CSV
            cat_stats_df = pd.DataFrame(cat_stats)
            cat_path = os.path.join(self.output_dir, 'tables', 'categorical_features_statistics.csv')
            cat_stats_df.to_csv(cat_path, index=False)
            
            print(f"📊 类别特征统计表:")
            print(f"  共 {len(categorical_cols)} 个类别特征")
            print(f"  已保存到: {cat_path}")
    
    def create_summary_report(self, stats, data):
        """创建简要总结报告"""
        print("\n" + "-"*50)
        print("📝 生成总结报告")
        print("-"*50)
        
        report_content = f"""数据集基础统计分析报告
========================================

一、数据集基本信息
------------------
- 总样本数: {stats['dataset_info']['samples']:,}
- 总特征数: {stats['dataset_info']['features']}
- 数值型特征: {stats['data_type_distribution']['numeric']['count']}个
- 类别型特征: {stats['data_type_distribution']['categorical']['count']}个

二、目标变量信息
---------------
"""
        
        if 'target_info' in stats and stats['target_info'].get('found', True):
            target_col = stats['target_info'].get('column_name', 'N/A')
            report_content += f"- 目标变量: {target_col}\n"
            
            for label, info in stats.get('target_distribution', {}).items():
                report_content += f"  - {label}: {info['count']:,} 个 ({info['percentage']}%)\n"
            
            if stats['target_info'].get('imbalance_ratio'):
                report_content += f"- 类别不平衡比例: {stats['target_info']['imbalance_ratio']}:1\n"
        else:
            report_content += "- 未检测到目标变量\n"
        
        report_content += f"""
三、结果文件
-----------
所有分析结果保存在: {self.output_dir}

├── figures/
│   ├── feature_type_distribution.png    # 特征类型分布图
│   └── target_variable_distribution.png  # 目标变量分布图
└── tables/
    ├── basic_statistics.json            # 基础统计信息
    ├── numeric_features_statistics.csv   # 数值特征统计
    └── categorical_features_statistics.csv # 类别特征统计

分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'analysis_summary.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 总结报告已保存到: {report_path}")
        
        # 在控制台也显示报告
        print("\n" + report_content)
    
    def create_all_feature_distributions(self, data, max_categories=20):
        """创建所有特征的分布柱状图，每行3个"""
        print("\n" + "="*60)
        print("📊 生成所有特征分布图 (每行3个)")
        print("="*60)
        
        # 分离数值型和类别型特征
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        # 排除目标变量（如果存在）
        target_col = None
        for col in ['deposit', 'Class', 'class', 'target', 'y']:
            if col in data.columns:
                target_col = col
                break
        
        if target_col:
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            if target_col in categorical_cols:
                categorical_cols.remove(target_col)
        
        all_features = categorical_cols + numeric_cols
        
        if not all_features:
            print("⚠️  没有找到可绘制的特征")
            return
        
        print(f"📈 共 {len(all_features)} 个特征需要绘图")
        print(f"  - 类别型: {len(categorical_cols)} 个")
        print(f"  - 数值型: {len(numeric_cols)} 个")
        
        # 计算需要的行数（每行3个）
        n_features = len(all_features)
        n_rows = (n_features + 2) // 3  # 向上取整
        
        # 设置画布大小
        fig_width = 15
        fig_height = 5 * n_rows
        
        # 创建大图
        fig, axes = plt.subplots(n_rows, 3, figsize=(fig_width, fig_height))
        fig.suptitle('所有特征分布图', fontsize=18, fontweight='bold', y=0.995)
        
        # 如果只有一行，axes不是二维数组，需要转换
        if n_rows == 1:
            axes = axes.reshape(1, -1) if hasattr(axes, 'reshape') else np.array([axes])
        
        # 扁平化axes便于迭代
        axes_flat = axes.flatten()
        
        # 遍历所有特征并绘制
        for idx, feature in enumerate(all_features):
            ax = axes_flat[idx]
            
            # 处理类别型特征
            if feature in categorical_cols:
                value_counts = data[feature].value_counts().head(max_categories)
                
                # 如果类别太多，分组显示
                if len(data[feature].unique()) > max_categories:
                    value_counts = data[feature].value_counts().head(max_categories)
                    title_suffix = f" (Top {max_categories})"
                else:
                    title_suffix = ""
                
                bars = ax.bar(range(len(value_counts)), value_counts.values, 
                            color=plt.cm.Set3(idx % 12), alpha=0.8, edgecolor='black')
                
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index.astype(str), rotation=45, ha='right', fontsize=8)
                
                # 添加数值标签
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
                
                ax.set_title(f'{feature}{title_suffix}', fontsize=11, fontweight='bold')
                ax.set_ylabel('频数', fontsize=9)
                ax.tick_params(axis='both', labelsize=8)
                
            # 处理数值型特征
            else:
                # 使用直方图
                data_values = data[feature].dropna()
                
                # 检查是否有足够的唯一值
                unique_vals = data_values.nunique()
                if unique_vals > 50:
                    # 使用直方图
                    ax.hist(data_values, bins=30, color=plt.cm.Set3(idx % 12), 
                        alpha=0.8, edgecolor='black')
                    ax.set_title(f'{feature} (直方图)', fontsize=11, fontweight='bold')
                else:
                    # 使用条形图显示分布
                    value_counts = data_values.value_counts().head(20)
                    bars = ax.bar(range(len(value_counts)), value_counts.values,
                                color=plt.cm.Set3(idx % 12), alpha=0.8, edgecolor='black')
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index.astype(str), rotation=45, ha='right', fontsize=8)
                    ax.set_title(f'{feature} (离散值)', fontsize=11, fontweight='bold')
                
                ax.set_ylabel('频数', fontsize=9)
                ax.tick_params(axis='both', labelsize=8)
            
            # 添加网格
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 隐藏多余的子图
        for idx in range(len(all_features), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = os.path.join(self.output_dir, 'figures', 'all_features_distribution.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 所有特征分布图已保存到: {chart_path}")
        print(f"   预览: {n_rows} 行 × 3 列 = {n_rows * 3} 个子图位置，实际使用 {len(all_features)} 个")

    
    def run_full_analysis(self):
        """运行完整的分析流程"""
        print("="*60)
        print("🚀 开始数据集基础统计分析")
        print("="*60)
        
        # 1. 加载数据
        data = self.load_data()
        if data is None:
            return None
        
        # 2. 显示数据基本信息
        print("\n📄 数据预览 (前5行):")
        print(data.head())
        print(f"\n📋 所有特征: {', '.join(data.columns.tolist())}")
        
        # 3. 获取基础统计
        stats = self.get_basic_stats(data)
        
        # 4. 创建可视化图表
        print("\n" + "="*60)
        print("🎨 创建可视化图表")
        print("="*60)
        self.create_feature_type_chart(data)
        self.create_target_distribution_chart(data)
        
        # 🆕 新增：创建所有特征分布图
        self.create_all_feature_distributions(data)
        
        # 5. 生成统计表格
        self.generate_statistics_tables(data)
        
        # 6. 创建总结报告
        self.create_summary_report(stats, data)
        
        print("\n" + "="*60)
        print("✅ 分析完成！")
        print("="*60)
        print(f"📁 所有结果已保存到: {self.output_dir}")
        
        return stats

def run_analysis():
    """运行分析的函数"""
    analyzer = BasicDataAnalysis()
    return analyzer.run_full_analysis()

def main():
    """主函数"""
    results = run_analysis()
    
    if results:
        print("\n🎉 基础统计分析成功完成！")
        print(f"请查看文件夹: ./preprocess_dataset/basic_data_analysis")
    else:
        print("\n❌ 分析失败，请检查数据文件")

if __name__ == "__main__":
    # 设置matplotlib支持中文显示
    try:
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except:
        pass
    main()
