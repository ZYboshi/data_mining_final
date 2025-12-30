# random_forest_model.py
"""
随机森林模型建模
使用十折交叉验证
"""
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score, roc_curve)
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
        './result/random_forest',
        './model_checkpoint/random_forest'
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
    print(f"类别分布:\n{y.value_counts()}")
    print(f"正类比例: {y.mean():.2%}")
    
    return X, y

def train_random_forest_base(X, y):
    """训练基础的随机森林模型"""
    print("\n🎲 训练随机森林模型...")
    
    # 随机森林参数设置
    rf_model = RandomForestClassifier(
        n_estimators=100,      # 树的数量
        max_depth=None,        # 树的最大深度
        min_samples_split=2,   # 内部节点分裂所需最小样本数
        min_samples_leaf=1,    # 叶节点所需最小样本数
        max_features='sqrt',   # 每棵树考虑的最大特征数
        random_state=42,
        n_jobs=-1,             # 使用所有CPU核心
        class_weight='balanced_subsample'  # 处理类别不平衡
    )
    
    return rf_model

def cross_validation_analysis(model, X, y):
    """十折交叉验证分析"""
    print("\n📊 十折交叉验证分析...")
    
    # 创建分层10折交叉验证
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    cv_metrics = {
        'fold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
        'train_time': []
    }
    
    y_all_true = []
    y_all_pred = []
    y_all_proba = []
    
    fold_num = 1
    total_start_time = time.time()
    
    for train_idx, val_idx in cv.split(X, y):
        fold_start_time = time.time()
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # 记录时间
        train_time = time.time() - fold_start_time
        
        # 计算指标
        cv_metrics['fold'].append(fold_num)
        cv_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        cv_metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        cv_metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        cv_metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
        cv_metrics['auc'].append(roc_auc_score(y_val, y_proba))
        cv_metrics['train_time'].append(train_time)
        
        # 收集所有预测
        y_all_true.extend(y_val)
        y_all_pred.extend(y_pred)
        y_all_proba.extend(y_proba)
        
        print(f"  第{fold_num}折: 准确率={accuracy_score(y_val, y_pred):.4f}, "
              f"训练时间={train_time:.2f}秒")
        
        fold_num += 1
    
    total_time = time.time() - total_start_time
    
    # 创建指标DataFrame
    metrics_df = pd.DataFrame(cv_metrics)
    
    print("\n" + "="*50)
    print("📈 交叉验证结果汇总:")
    print("="*50)
    
    print("\n各折详细指标:")
    print(metrics_df[['fold', 'accuracy', 'precision', 'recall', 'f1', 'auc']].round(4))
    
    print(f"\n🏆 平均指标 (±标准差):")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        mean_val = metrics_df[metric].mean()
        std_val = metrics_df[metric].std()
        print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"\n⏱️  时间统计:")
    print(f"  平均每折训练时间: {metrics_df['train_time'].mean():.2f}秒")
    print(f"  总训练时间: {total_time:.2f}秒")
    
    return metrics_df, np.array(y_all_true), np.array(y_all_pred), np.array(y_all_proba), cv

def analyze_feature_importance(model, X, top_n=15):
    """分析特征重要性"""
    print("\n🔬 特征重要性分析...")
    
    # 获取特征重要性
    feature_importance = model.feature_importances_
    
    # 创建DataFrame
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    })
    
    # 排序
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} 最重要的特征:")
    print(importance_df.head(top_n).round(4))
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    
    top_features = importance_df.head(top_n)
    colors = plt.cm.viridis(np.linspace(0.3, 1, len(top_features)))
    
    plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance Score')
    plt.title(f'Random Forest - Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()  # 最重要的在顶部
    plt.tight_layout()
    
    # 保存图片
    save_path = './result/random_forest/feature_importance.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"✅ 特征重要性图已保存到: {save_path}")
    plt.show()
    
    return importance_df

def train_final_model(X, y):
    """训练最终的随机森林模型"""
    print("\n🎯 训练最终模型...")
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 最终模型 - 可以调整参数
    final_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )
    
    # 训练
    start_time = time.time()
    final_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"模型训练完成，耗时: {train_time:.2f}秒")
    
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
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Deposit', 'Deposit'],
                yticklabels=['No Deposit', 'Deposit'])
    plt.title('Random Forest - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = './result/random_forest/confusion_matrix.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"✅ 混淆矩阵图已保存到: {save_path}")
    plt.show()
    
    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', label=f'Random Forest (AUC = {roc_auc_score(y_test, y_proba):.3f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = './result/random_forest/roc_curve.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"✅ ROC曲线图已保存到: {save_path}")
    plt.show()
    
    return final_model, X_test, y_test, y_pred, y_proba

def hyperparameter_tuning(X, y):
    """随机森林超参数调优"""
    print("\n⚙️  超参数调优（网格搜索）...")
    
    from sklearn.model_selection import GridSearchCV
    
    # 简化版的网格搜索（避免耗时过长）
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # 使用较小的网格搜索
    grid_search = GridSearchCV(
        rf_base, 
        param_grid, 
        cv=3,  # 用3折减少时间
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("正在进行网格搜索（这可能需要几分钟）...")
    grid_search.fit(X, y)
    
    print(f"\n✨ 最佳参数: {grid_search.best_params_}")
    print(f"✨ 最佳交叉验证AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def save_results(model, metrics_df, importance_df):
    """保存模型和结果"""
    print("\n💾 保存结果...")
    
    import pickle
    
    # 保存模型
    model_path = './model_checkpoint/random_forest/random_forest_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ 模型已保存到: {model_path}")
    
    # 保存指标
    metrics_path = './result/random_forest/cross_validation_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ 交叉验证指标已保存到: {metrics_path}")
    
    # 保存特征重要性
    importance_path = './result/random_forest/feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"✅ 特征重要性已保存到: {importance_path}")
    
    # 保存参数
    params_path = './result/random_forest/model_params.txt'
    with open(params_path, 'w') as f:
        f.write(f"Model Parameters:\n")
        for key, value in model.get_params().items():
            f.write(f"{key}: {value}\n")
    print(f"✅ 模型参数已保存到: {params_path}")

def compare_with_logistic(logistic_model=None):
    """与逻辑回归模型比较"""
    print("\n⚖️  模型性能比较...")
    
    # 这里可以添加逻辑回归对比
    # 如果提供了逻辑回归模型，可以进行比较
    if logistic_model:
        print("需要加载逻辑回归模型进行比较...")
        # 对比代码可以根据需要添加

def main():
    """主函数"""
    print("=" * 60)
    print("🌲 Random Forest Modeling (10-Fold CV)")
    print("=" * 60)
    
    # 0. 创建目录
    create_directories()
    
    # 1. 加载数据
    X, y = load_and_split_data()
    
    # 2. 训练基础模型并进行交叉验证
    rf_model = train_random_forest_base(X, y)
    metrics_df, y_all_true, y_all_pred, y_all_proba, cv = cross_validation_analysis(rf_model, X, y)
    
    # 3. 在整个数据集上训练以获取特征重要性
    print("\n" + "="*40)
    print("Training model on full dataset...")
    rf_model_full = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model_full.fit(X, y)
    
    # 4. 特征重要性分析
    importance_df = analyze_feature_importance(rf_model_full, X, top_n=15)
    
    # 5. 训练最终模型
    final_model, X_test, y_test, y_pred, y_proba = train_final_model(X, y)
    
    # 6. （可选）超参数调优
    want_tuning = input("\n是否进行超参数调优？(y/n, 可能需要几分钟): ").lower()
    if want_tuning == 'y':
        try:
            best_model, best_params = hyperparameter_tuning(X, y)
            final_model = best_model
            print(f"使用调优后的模型进行最终评估...")
        except Exception as e:
            print(f"调优过程中出错: {e}")
            print("继续使用基础模型...")
    
    # 7. 保存所有结果
    save_results(final_model, metrics_df, importance_df)
    
    print("\n" + "="*60)
    print("✅ Random Forest Modeling Completed Successfully!")
    print("="*60)
    print("\n📁 生成的文件:")
    print(f"  - ./result/random_forest/")
    print(f"     • feature_importance.png (特征重要性图)")
    print(f"     • confusion_matrix.png (混淆矩阵)")
    print(f"     • roc_curve.png (ROC曲线)")
    print(f"     • cross_validation_metrics.csv (交叉验证指标)")
    print(f"     • feature_importance.csv (特征重要性数据)")
    print(f"  - ./model_checkpoint/random_forest/")
    print(f"     • random_forest_model.pkl (训练好的模型)")
    print(f"\n📊 性能总结:")
    print(f"  平均准确率: {metrics_df['accuracy'].mean():.4f}")
    print(f"  平均AUC: {metrics_df['auc'].mean():.4f}")
    print(f"  平均F1分数: {metrics_df['f1'].mean():.4f}")
    
    return final_model, metrics_df, importance_df

if __name__ == "__main__":
    try:
        start_time = time.time()
        model, metrics_df, importance_df = main()
        total_time = time.time() - start_time
        
        print(f"\n⏱️  总运行时间: {total_time:.2f}秒")
        print("🎉 随机森林建模完成!")
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到: {e}")
        print("请确保 './dataset/data_label_encoded.csv' 文件存在")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
