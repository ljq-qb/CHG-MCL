import numpy as np

class Metric_fun(object):
    def __init__(self):
        super(Metric_fun).__init__()


    # 计算模型的性能指标
    def cv_mat_model_evaluate(self, association_mat, predict_mat):

        real_score = np.mat(association_mat.detach().cpu().numpy().flatten())
        predict_score = np.mat(predict_mat.detach().cpu().numpy().flatten())

        return self.get_metrics(real_score, predict_score)


    def get_metrics(self, real_score, predict_score):
        # 第一步：构造候选阈值
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))

        sorted_predict_score_num = len(sorted_predict_score)
        thresholds = sorted_predict_score[           # 构造 thresholds（阈值）,选取 1000 个等间隔的阈值（从 sorted_predict_score 中）。这些阈值用于计算 不同的二分类结果，以绘制 ROC 和 PR 曲线。
            (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
        thresholds = np.mat(thresholds)
        thresholds_num = thresholds.shape[1]      # 扩展预测分数矩阵，以便后续基于多个阈值进行二值化处理。
        predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))    # np.tile(A, (m, n)) 用于 扩展矩阵 A，重复 m 行、n 列。沿行方向（垂直）重复 thresholds_num 次，每一行都和 predict_score 相同。

        # 第二步：二值化预测分数
        negative_index = np.where(predict_score_matrix < thresholds.T)
        positive_index = np.where(predict_score_matrix >= thresholds.T)
        predict_score_matrix[negative_index] = 0     # 小于阈值 的置为 0（负类）
        predict_score_matrix[positive_index] = 1     # 大于等于阈值 的置为 1（正类）

        # 第三步：计算混淆矩阵元素
        TP = predict_score_matrix * real_score.T
        FP = predict_score_matrix.sum(axis=1) - TP
        FN = real_score.sum() - TP
        TN = len(real_score.T) - TP - FP - FN

        # 第四步：计算 ROC 曲线和 AUC
        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)
        ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
        ROC_dot_matrix.T[0] = [0, 0]
        ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
        x_ROC = ROC_dot_matrix[0].T
        y_ROC = ROC_dot_matrix[1].T

        auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])   # 使用梯形面积法计算 AUC（ROC 曲线下面积）

        # 第五步：计算 PR 曲线和 AUPR
        recall_list = tpr
        precision_list = TP / (TP + FP)
        PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
        PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
        PR_dot_matrix.T[0] = [0, 1]
        PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
        x_PR = PR_dot_matrix[0].T
        y_PR = PR_dot_matrix[1].T
        aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

        # 第六步：计算 F1-score, Accuracy, Specificity
        f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
        accuracy_list = (TP + TN) / len(real_score.T)
        specificity_list = TN / (TN + FP)

        max_index = np.argmax(f1_score_list)
        f1_score = f1_score_list[max_index, 0]
        accuracy = accuracy_list[max_index, 0]
        specificity = specificity_list[max_index, 0]
        recall = recall_list[max_index, 0]
        precision = precision_list[max_index, 0]

        results = [
            round(auc[0, 0], 5),
            round(aupr[0, 0], 5),
            round(f1_score, 5),
            round(accuracy, 5),
            round(recall, 5),
            round(specificity, 5),
            round(precision, 5)
        ]
        return results

