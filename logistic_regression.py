# logistic_regression_model.py
"""
逻辑回归模型建模
使用十折交叉验证
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib支持中文显示
try:
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass

def create_directories():
    """创建必要的文件夹"""
    directories = [
        './result/logistic',
        './model_checkpoint/logistic',
        './result',
        './model_checkpoint'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ 创建目录: {directory}")

def load_and_split_data():
    """加载并分割数据"""
    print("📂 加载数据...")
    data = pd.read_csv('./dataset/data_label_encoded.csv')
    
    # 分离特征和目标
    X = data.drop('deposit', axis=1)  # 特征
    y = data['deposit']  # 目标变量
    
    # 查看数据基本情况
    print(f"数据集形状: {data.shape}")
    print(f"目标变量分布:\n{y.value_counts()}")
    print(f"类别比例 [是/否]: {y.mean():.2%} / {(1-y.mean()):.2%}")
    
    return X, y

def standardize_features(X):
    """标准化数值特征"""
    print("\n⚙️ 标准化特征...")
    
    # 数值特征列
    numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    
    # 只选择存在的列
    existing_cols = [col for col in numerical_cols if col in X.columns]
    print(f"标准化的数值特征: {existing_cols}")
    
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[existing_cols] = scaler.fit_transform(X[existing_cols])
    
    return X_scaled

def train_with_cross_validation(X, y):
    """
    使用十折交叉验证训练逻辑回归
    """
    print("\n🎯 开始十折交叉验证逻辑回归...")
    
    # 1. 创建模型
    model = LogisticRegression(
        max_iter=1000,  # 增加迭代次数确保收敛
        random_state=42,
        C=1.0  # 正则化强度，默认值
    )
    
    # 2. 创建分层10折交叉验证（保持类别比例）
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    print("正在进行10折交叉验证...")
    
    # 3. 计算交叉验证得分
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print(f"\n📊 交叉验证结果:")
    print(f"  平均准确率: {cv_scores.mean():.4f}")
    print(f"  准确率标准差: {cv_scores.std():.4f}")
    print(f"  每折准确率: {cv_scores.round(4)}")
    
    return model, cv, cv_scores

def detailed_cv_analysis(model, X, y, cv):
    """详细的交叉验证分析"""
    print("\n🔍 详细交叉验证分析...")
    
    # 收集每折的预测结果
    cv_metrics = {
        'fold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    # 用于存储所有预测
    y_all_pred = []
    y_all_true = []
    y_all_proba = []
    
    fold_num = 1
    for train_idx, val_idx in cv.split(X, y):
        # 分割数据
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # 计算指标
        cv_metrics['fold'].append(fold_num)
        cv_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        cv_metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        cv_metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        cv_metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
        cv_metrics['auc'].append(roc_auc_score(y_val, y_proba))
        
        # 收集所有预测
        y_all_true.extend(y_val)
        y_all_pred.extend(y_pred)
        y_all_proba.extend(y_proba)
        
        fold_num += 1
    
    # 创建指标DataFrame
    metrics_df = pd.DataFrame(cv_metrics)
    
    print("\n📈 每折详细指标:")
    print(metrics_df.round(4))
    
    print("\n🌟 平均指标:")
    print(f"  准确率: {metrics_df['accuracy'].mean():.4f} ± {metrics_df['accuracy'].std():.4f}")
    print(f"  精确率: {metrics_df['precision'].mean():.4f} ± {metrics_df['precision'].std():.4f}")
    print(f"  召回率: {metrics_df['recall'].mean():.4f} ± {metrics_df['recall'].std():.4f}")
    print(f"  F1分数: {metrics_df['f1'].mean():.4f} ± {metrics_df['f1'].std():.4f}")
    print(f"  AUC: {metrics_df['auc'].mean():.4f} ± {metrics_df['auc'].std():.4f}")
    
    return np.array(y_all_true), np.array(y_all_pred), np.array(y_all_proba), metrics_df

def analyze_feature_importance(model, X):
    """分析特征重要性"""
    print("\n🔬 特征重要性分析:")
    
    feature_names = X.columns
    coefficients = model.coef_[0]
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    })
    
    # 按绝对系数排序
    importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
    
    print("\nFeature Importance (sorted by impact):")
    print(importance_df[['feature', 'coefficient']].round(4).head(10))
    
    # 可视化 - 使用英文避免中文字体问题
    plt.figure(figsize=(10, 6))
    top_10 = importance_df.head(10)
    colors = ['red' if coef < 0 else 'green' for coef in top_10['coefficient']]
    plt.barh(range(len(top_10)), top_10['abs_coefficient'], color=colors)
    plt.yticks(range(len(top_10)), top_10['feature'])
    plt.xlabel('Coefficient Absolute Value')
    plt.title('Logistic Regression - Top 10 Most Important Features')
    plt.tight_layout()
    
    # 保存图片（确保文件夹存在）
    save_path = './result/logistic/logistic_feature_importance.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"✅ 特征重要性图已保存到: {save_path}")
    plt.show()
    
    return importance_df

def final_train_and_evaluate(X, y):
    """最终训练和评估模型"""
    print("\n🎯 最终模型训练和评估...")
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 训练最终模型
    final_model = LogisticRegression(max_iter=1000, random_state=42)
    final_model.fit(X_train, y_train)
    
    # 预测
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]
    
    # 评估
    print("\n📋 测试集评估结果:")
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:\n{cm}")
    
    # 可视化混淆矩阵 - 用简单英文
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # 保存图片
    save_path = './result/logistic/logistic_confusion_matrix.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"✅ 混淆矩阵图已保存到: {save_path}")
    plt.show()
    
    return final_model

def save_model_and_metrics(model, metrics_df, X_columns):
    """保存模型和指标"""
    print("\n💾 保存模型和指标...")
    
    # 1. 保存模型
    model_path = './model_checkpoint/logistic/logistic_regression_model.pkl'
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ 模型已保存到: {model_path}")
    
    # 2. 保存指标
    metrics_path = './result/logistic/logistic_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ 指标已保存到: {metrics_path}")
    
    # 3. 保存特征名（用于后续预测）
    features_path = './model_checkpoint/logistic/logistic_features.npy'
    np.save(features_path, X_columns)
    print(f"✅ 特征名已保存到: {features_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("Logistic Regression Modeling (10-Fold CV)")
    print("=" * 60)
    
    # 0. 创建必要的文件夹
    create_directories()
    
    # 1. 加载数据
    X, y = load_and_split_data()
    
    # 2. 标准化特征
    X_scaled = standardize_features(X)
    
    # 3. 交叉验证训练
    model, cv, cv_scores = train_with_cross_validation(X_scaled, y)
    
    # 4. 详细分析
    y_all_true, y_all_pred, y_all_proba, metrics_df = detailed_cv_analysis(
        model, X_scaled, y, cv
    )
    
    # 5. 重新在整个数据集上训练以分析特征重要性
    print("\n" + "=" * 40)
    print("Training final model on full dataset...")
    final_model = LogisticRegression(max_iter=1000, random_state=42)
    final_model.fit(X_scaled, y)
    
    # 6. 分析特征重要性
    importance_df = analyze_feature_importance(final_model, X_scaled)
    
    # 7. 训练最终模型并评估
    final_model = final_train_and_evaluate(X_scaled, y)
    
    # 8. 保存所有内容
    save_model_and_metrics(final_model, metrics_df, X_scaled.columns)
    
    print("\n✅ Logistic Regression Modeling Completed!")
    print(f"Average Cross-Validation Accuracy: {cv_scores.mean():.4f}")
    print(f"Files saved in ./result/logistic/ and ./model_checkpoint/logistic/")
    
    return final_model, metrics_df, importance_df

if __name__ == "__main__":
    try:
        model, metrics_df, importance_df = main()
        print("\n🎉 所有操作成功完成!")
        print(f"查看结果文件夹: './result/logistic/'")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n检查以下事项:")
        print("1. 确保 './dataset/data_label_encoded.csv' 文件存在")
        print("2. 确保有写入权限")
        print("3. 尝试运行: mkdir -p result/logistic model_checkpoint/logistic")
