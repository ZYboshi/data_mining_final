# naive_bayes_model.py
"""
朴素贝叶斯模型建模
使用十折交叉验证
"""
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib
try:
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass

def create_directories():
    """创建必要的文件夹"""
    directories = [
        './result/naive_bayes',
        './model_checkpoint/naive_bayes'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ 创建目录: {directory}")

def load_and_split_data():
    """加载数据"""
    print("📂 加载数据...")
    data = pd.read_csv('./dataset/data_label_encoded.csv')
    
    X = data.drop('deposit', axis=1)
    y = data['deposit']
    
    print(f"数据集形状: {data.shape}")
    print(f"特征数量: {X.shape[1]}")
    print(f"类别分布:\n{y.value_counts()}")
    print(f"正类比例 (订阅定期存款): {y.mean():.2%}")
    
    return X, y

def preprocess_for_naive_bayes(X):
    """
    为朴素贝叶斯预处理数据
    """
    print("\n⚙️ 数据预处理...")
    
    # 1. 数值特征标准化（高斯朴素贝叶斯需要）
    numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    existing_numerical = [col for col in numerical_cols if col in X.columns]
    
    if len(existing_numerical) > 0:
        print(f"数值特征 (标准化): {existing_numerical}")
        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[existing_numerical] = scaler.fit_transform(X[existing_numerical])
    else:
        X_scaled = X.copy()
    
    return X_scaled

def train_naive_bayes_model(X, y, bayes_type='gaussian'):
    """
    训练朴素贝叶斯模型
    bayes_type: 'gaussian', 'bernoulli', 'multinomial'
    """
    print(f"\n🔮 训练{bayes_type}朴素贝叶斯模型...")
    
    if bayes_type == 'gaussian':
        model = GaussianNB(var_smoothing=1e-9)
    elif bayes_type == 'bernoulli':
        model = BernoulliNB()
    elif bayes_type == 'multinomial':
        model = MultinomialNB()
    else:
        model = GaussianNB()
    
    return model

def cross_validation_analysis(model, X, y, bayes_type='gaussian'):
    """十折交叉验证分析"""
    print("\n📊 十折交叉验证分析...")
    
    # 转换回pandas DataFrame以确保正确索引
    if isinstance(X, pd.DataFrame):
        X_df = X
    else:
        X_df = pd.DataFrame(X)
    
    # 创建分层10折交叉验证
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    cv_metrics = {
        'fold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
        'train_time': [],
        'pred_time': []
    }
    
    y_all_true = []
    y_all_pred = []
    y_all_proba = []
    
    fold_num = 1
    total_start_time = time.time()
    
    for train_idx, val_idx in cv.split(X_df, y):
        fold_start_time = time.time()
        
        # 正确使用iloc
        X_train = X_df.iloc[train_idx]
        X_val = X_df.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        # 训练
        model.fit(X_train, y_train)
        train_time = time.time() - fold_start_time
        
        # 预测
        pred_start_time = time.time()
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        pred_time = time.time() - pred_start_time
        
        # 计算指标
        cv_metrics['fold'].append(fold_num)
        cv_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        cv_metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        cv_metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        cv_metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
        cv_metrics['auc'].append(roc_auc_score(y_val, y_proba))
        cv_metrics['train_time'].append(train_time)
        cv_metrics['pred_time'].append(pred_time)
        
        # 收集所有预测
        y_all_true.extend(y_val)
        y_all_pred.extend(y_pred)
        y_all_proba.extend(y_proba)
        
        print(f"  第{fold_num}折: 准确率={accuracy_score(y_val, y_pred):.4f}, "
              f"AUC={roc_auc_score(y_val, y_proba):.4f}")
        
        fold_num += 1
    
    total_time = time.time() - total_start_time
    
    # 创建指标DataFrame
    metrics_df = pd.DataFrame(cv_metrics)
    
    print("\n" + "="*50)
    print(f"📈 {bayes_type}朴素贝叶斯交叉验证结果汇总:")
    print("="*50)
    
    print("\n各折详细指标:")
    print(metrics_df[['fold', 'accuracy', 'precision', 'recall', 'f1', 'auc']].round(4))
    
    print(f"\n🏆 平均指标 (±标准差):")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        mean_val = metrics_df[metric].mean()
        std_val = metrics_df[metric].std()
        print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"\n⏱️  时间统计:")
    print(f"  平均每折训练时间: {metrics_df['train_time'].mean():.4f}秒")
    print(f"  平均每折预测时间: {metrics_df['pred_time'].mean():.4f}秒")
    print(f"  总运行时间: {total_time:.2f}秒")
    
    return metrics_df, np.array(y_all_true), np.array(y_all_pred), np.array(y_all_proba), cv

def analyze_model_probabilities(model, X_df, y, bayes_type='gaussian'):
    """
    分析模型的概率分布
    """
    print(f"\n🔍 分析{bayes_type}朴素贝叶斯的概率分布...")
    
    # 预测概率
    model.fit(X_df, y)
    y_proba = model.predict_proba(X_df)
    
    # 可视化概率分布
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 概率分布直方图
    ax1 = axes[0]
    ax1.hist(y_proba[:, 1], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('预测概率 (订阅定期存款)')
    ax1.set_ylabel('频数')
    ax1.set_title(f'预测概率分布 - {bayes_type}朴素贝叶斯')
    ax1.grid(True, alpha=0.3)
    
    # 2. 按真实类别分组的概率分布
    ax2 = axes[1]
    colors = ['red', 'green']
    labels = ['未订阅', '已订阅']
    
    for i in [0, 1]:
        mask = (y == i)
        ax2.hist(y_proba[mask, 1], bins=30, alpha=0.6, 
                color=colors[i], label=labels[i])
    
    ax2.set_xlabel('预测概率')
    ax2.set_ylabel('频数')
    ax2.set_title(f'按真实类别分组的概率分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'./result/naive_bayes/probability_distribution_{bayes_type}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"✅ 概率分布图已保存到: {save_path}")
    plt.show()
    
    # 输出模型参数
    print(f"\n📊 {bayes_type}朴素贝叶斯参数:")
    if hasattr(model, 'class_prior_'):
        print(f"类先验概率: {model.class_prior_}")
    
    if bayes_type == 'gaussian' and hasattr(model, 'theta_'):
        print(f"\n前5个特征的类条件均值:")
        for i in range(min(5, len(model.theta_[0]))):
            print(f"  特征 {i}: 类0={model.theta_[0][i]:.4f}, 类1={model.theta_[1][i]:.4f}")

def evaluate_final_model(model, X_df, y, bayes_type='gaussian'):
    """评估最终模型"""
    print(f"\n🎯 评估{bayes_type}朴素贝叶斯最终模型...")
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 训练最终模型
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"模型训练完成，耗时: {train_time:.4f}秒")
    
    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 评估
    print("\n📋 测试集评估结果:")
    print(classification_report(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"AUC Score: {auc_score:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:\n{cm}")
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Deposit', 'Deposit'],
                yticklabels=['No Deposit', 'Deposit'])
    plt.title(f'{bayes_type.capitalize()} Naive Bayes - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = f'./result/naive_bayes/confusion_matrix_{bayes_type}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"✅ 混淆矩阵图已保存到: {save_path}")
    plt.show()
    
    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'{bayes_type.capitalize()} NB (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {bayes_type.capitalize()} Naive Bayes')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = f'./result/naive_bayes/roc_curve_{bayes_type}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"✅ ROC曲线图已保存到: {save_path}")
    plt.show()
    
    # Precision-Recall曲线
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'g-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {bayes_type.capitalize()} NB')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = f'./result/naive_bayes/precision_recall_curve_{bayes_type}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"✅ Precision-Recall曲线图已保存到: {save_path}")
    plt.show()
    
    return model, X_test, y_test, y_pred, y_proba

def compare_different_nb_types_simple(X_df, y):
    """简单比较不同类型的朴素贝叶斯"""
    print("\n⚖️ 比较不同类型的朴素贝叶斯...")
    
    nb_types = ['gaussian', 'bernoulli', 'multinomial']
    results = {}
    
    for nb_type in nb_types:
        print(f"\n测试 {nb_type} 朴素贝叶斯...")
        
        # 复制数据避免修改原始数据
        X_temp = X_df.copy()
        
        if nb_type == 'multinomial':
            # 多项朴素贝叶斯需要非负特征
            scaler = MinMaxScaler()
            X_temp = pd.DataFrame(scaler.fit_transform(X_temp), 
                                 columns=X_temp.columns)
        
        # 创建模型
        model = train_naive_bayes_model(X_temp, y, nb_type)
        
        # 使用5折交叉验证快速评估
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []
        accuracy_scores = []
        
        for train_idx, val_idx in cv.split(X_temp, y):
            X_train = X_temp.iloc[train_idx]
            X_val = X_temp.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            auc_scores.append(roc_auc_score(y_val, y_proba))
            accuracy_scores.append(accuracy_score(y_val, y_pred))
        
        results[nb_type] = {
            'mean_auc': np.mean(auc_scores),
            'std_auc': np.std(auc_scores),
            'mean_accuracy': np.mean(accuracy_scores),
            'std_accuracy': np.std(accuracy_scores)
        }
        
        print(f"  AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
        print(f"  准确率: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    
    # 可视化比较
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # AUC比较
    ax1 = axes[0]
    types = list(results.keys())
    auc_means = [results[t]['mean_auc'] for t in types]
    auc_stds = [results[t]['std_auc'] for t in types]
    
    bars1 = ax1.bar(types, auc_means, yerr=auc_stds, capsize=10, 
                   color=['blue', 'green', 'orange'], alpha=0.7)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (AUC=0.5)')
    ax1.set_xlabel('Naive Bayes Type')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('AUC Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率比较
    ax2 = axes[1]
    acc_means = [results[t]['mean_accuracy'] for t in types]
    acc_stds = [results[t]['std_accuracy'] for t in types]
    
    bars2 = ax2.bar(types, acc_means, yerr=acc_stds, capsize=10, 
                   color=['cyan', 'lime', 'gold'], alpha=0.7)
    ax2.set_xlabel('Naive Bayes Type')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值
    for bars, ax, means in zip([bars1, bars2], [ax1, ax2], [auc_means, acc_means]):
        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = './result/naive_bayes/nb_type_comparison.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"✅ 朴素贝叶斯类型比较图已保存到: {save_path}")
    plt.show()
    
    # 确定最佳类型
    best_type_auc = max(results.items(), key=lambda x: x[1]['mean_auc'])
    best_type_acc = max(results.items(), key=lambda x: x[1]['mean_accuracy'])
    
    print(f"\n🌟 基于AUC的最佳类型: {best_type_auc[0]} (AUC = {best_type_auc[1]['mean_auc']:.4f})")
    print(f"🌟 基于准确率的最佳类型: {best_type_acc[0]} (准确率 = {best_type_acc[1]['mean_accuracy']:.4f})")
    
    # 选择AUC最佳的类型
    return best_type_auc[0], results

def save_results(model, metrics_df, bayes_type='gaussian'):
    """保存模型和结果"""
    print(f"\n💾 保存{bayes_type}朴素贝叶斯结果...")
    
    import pickle
    import json
    
    # 保存模型
    model_path = f'./model_checkpoint/naive_bayes/{bayes_type}_naive_bayes_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ 模型已保存到: {model_path}")
    
    # 保存指标
    metrics_path = f'./result/naive_bayes/{bayes_type}_cross_validation_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ 交叉验证指标已保存到: {metrics_path}")
    
    # 保存模型参数
    params_path = f'./result/naive_bayes/{bayes_type}_model_params.json'
    params_dict = model.get_params()
    
    if hasattr(model, 'class_prior_'):
        params_dict['class_prior'] = model.class_prior_.tolist()
    if hasattr(model, 'theta_'):
        params_dict['class_0_means'] = model.theta_[0].tolist()
        params_dict['class_1_means'] = model.theta_[1].tolist()
    
    with open(params_path, 'w') as f:
        json.dump(params_dict, f, indent=2)
    
    print(f"✅ 模型参数已保存到: {params_path}")
    
    # 保存性能总结
    summary_path = f'./result/naive_bayes/{bayes_type}_performance_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"{bayes_type.capitalize()} Naive Bayes Performance Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Average Accuracy: {metrics_df['accuracy'].mean():.4f} ± {metrics_df['accuracy'].std():.4f}\n")
        f.write(f"Average Precision: {metrics_df['precision'].mean():.4f} ± {metrics_df['precision'].std():.4f}\n")
        f.write(f"Average Recall: {metrics_df['recall'].mean():.4f} ± {metrics_df['recall'].std():.4f}\n")
        f.write(f"Average F1 Score: {metrics_df['f1'].mean():.4f} ± {metrics_df['f1'].std():.4f}\n")
        f.write(f"Average AUC: {metrics_df['auc'].mean():.4f} ± {metrics_df['auc'].std():.4f}\n")
        f.write(f"Average Training Time: {metrics_df['train_time'].mean():.4f}秒\n")
        f.write(f"Average Prediction Time: {metrics_df['pred_time'].mean():.4f}秒\n")
    
    print(f"✅ 性能总结已保存到: {summary_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("🔮 Naive Bayes Modeling (10-Fold CV)")
    print("=" * 60)
    
    # 0. 创建目录
    create_directories()
    
    # 1. 加载数据
    X, y = load_and_split_data()
    
    # 2. 预处理
    X_processed = preprocess_for_naive_bayes(X)
    print(f"处理后的数据形状: {X_processed.shape}")
    
    # 3. 简单比较不同的朴素贝叶斯类型
    best_type, comparison_results = compare_different_nb_types_simple(X_processed, y)
    
    # 4. 使用最佳类型进行详细分析
    print(f"\n{'='*60}")
    print(f"🎯 详细分析最佳类型: {best_type}")
    print('='*60)
    
    # 根据不同类型可能需要不同的预处理
    X_final = X_processed.copy()
    if best_type == 'multinomial':
        # 多项朴素贝叶斯需要非负特征
        scaler = MinMaxScaler()
        X_final = pd.DataFrame(scaler.fit_transform(X_final), columns=X_final.columns)
    
    # 创建模型
    model = train_naive_bayes_model(X_final, y, best_type)
    
    # 5. 十折交叉验证
    metrics_df, y_all_true, y_all_pred, y_all_proba, cv = cross_validation_analysis(
        model, X_final, y, best_type
    )
    
    # 6. 分析概率分布
    analyze_model_probabilities(model, X_final, y, best_type)
    
    # 7. 训练最终模型并完整评估
    final_model, X_test, y_test, y_pred, y_proba = evaluate_final_model(
        model, X_final, y, best_type
    )
    
    # 8. 保存结果
    save_results(final_model, metrics_df, best_type)
    
    print("\n" + "="*60)
    print(f"✅ {best_type.capitalize()} Naive Bayes Modeling Completed!")
    print("="*60)
    
    print(f"\n📊 性能总结:")
    print(f"  平均准确率: {metrics_df['accuracy'].mean():.4f}")
    print(f"  平均AUC: {metrics_df['auc'].mean():.4f}")
    print(f"  平均F1分数: {metrics_df['f1'].mean():.4f}")
    print(f"  平均召回率: {metrics_df['recall'].mean():.4f}")
    print(f"  平均精确率: {metrics_df['precision'].mean():.4f}")
    
    print(f"\n⏱️  时间效率:")
    print(f"  平均每折训练时间: {metrics_df['train_time'].mean():.4f}秒")
    print(f"  平均每折预测时间: {metrics_df['pred_time'].mean():.4f}秒")
    
    print(f"\n📁 结果文件保存在:")
    print(f"  ./result/naive_bayes/ - 包含所有图片和CSV文件")
    print(f"  ./model_checkpoint/naive_bayes/ - 包含训练好的模型")
    
    return final_model, metrics_df, comparison_results

if __name__ == "__main__":
    try:
        start_time = time.time()
        model, metrics_df, results = main()
        total_time = time.time() - start_time
        
        print(f"\n⏱️  总运行时间: {total_time:.2f}秒")
        print("🎉 朴素贝叶斯建模完成!")
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到: {e}")
        print("请确保 './dataset/data_label_encoded.csv' 文件存在")
        print("尝试运行: python data_preprocessing.py 先进行数据预处理")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 如果问题持续，可以:")
        print("1. 检查数据文件路径")
        print("2. 确保所有必要的包已安装")
        print("3. 尝试重启Python环境")
