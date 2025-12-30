1. 环境加载：是哟个environment.yml文件，使用conda create -f environment.yml命令创建环境。
2. 数据集加载:直接运行data_processing.py文件即可
3. 数据分析：
    * 基础特征：运行basic_data_analysis.py文件
    * 异常分析：运行outliers_data_analysis.py文件
    * e.g:其中很大一部分已经集成在data_processing.py文件中了，单独写这两个文件是因为这两部分是我在昨晚全部试验后后知后觉需要做这些步骤才另写的。
4. 各个模型运行:
    * 朴素贝叶斯模型：运行bayes.py文件
    * 逻辑回归模型：运行logistic_regression.py文件
    * 随机森林模型：运行random_forest.py文件
