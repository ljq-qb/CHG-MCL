# # 移除对比学习模块
# import torch
# from prepareData import prepare_data
# import numpy as np
# from torch import optim
# from param import parameter_parser
# from Module import HGCLAMIR
# from utils import get_L2reg, Myloss
# from Calculate_Metrics import Metric_fun
# from trainData import Dataset
# import ConstructHW
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.impute import SimpleImputer
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import silhouette_score
# import warnings
#
# warnings.filterwarnings('ignore')
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# USE_GRID_SEARCH = True
#
#
# # 数据1、2
# def find_best_dbscan_params(data, eps_values=[0.3, 0.5, 0.7], min_samples_values=[5, 10, 15]):
#     best_eps, best_min_samples = None, None
#     best_score = -1
#     for eps in eps_values:
#         for min_samples in min_samples_values:
#             clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
#             labels = clustering.labels_
#             if len(set(labels)) > 1:
#                 score = silhouette_score(data, labels)
#                 if score > best_score:
#                     best_score = score
#                     best_eps, best_min_samples = eps, min_samples
#     return best_eps, best_min_samples if best_eps is not None else (0.5, 10)
#
#
# def find_optimal_eps(data, min_samples=10, quantile=0.95):
#     neighbors = NearestNeighbors(n_neighbors=min_samples)
#     neighbors.fit(data)
#     distances, _ = neighbors.kneighbors(data)
#     distances = np.sort(distances[:, min_samples - 1], axis=0)
#     return distances[int(len(distances) * quantile)]
#
#
# def train_epoch(model, train_data, optim, opt):
#     model.train()
#     regression_crit = Myloss()
#
#     one_index = train_data[2][0].to(device).t().tolist()
#     zero_index = train_data[2][1].to(device).t().tolist()
#
#     dis_sim_integrate_tensor = train_data[0].to(device)
#     mi_sim_integrate_tensor = train_data[1].to(device)
#
#     concat_miRNA = np.hstack([train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])
#     concat_mi_tensor = torch.FloatTensor(concat_miRNA).to(device)
#
#     if USE_GRID_SEARCH:
#         best_eps_mi, best_min_samples_mi = find_best_dbscan_params(
#             concat_mi_tensor.cpu().numpy(),
#             eps_values=np.linspace(0.1, 1.0, 10),
#             min_samples_values=[5, 10, 15]
#         )
#     else:
#         best_eps_mi = find_optimal_eps(concat_mi_tensor.cpu().numpy(), min_samples=10, quantile=0.95)
#         best_min_samples_mi = 10
#
#     G_mi_Kn = ConstructHW.constructHW_knn(concat_mi_tensor.cpu().numpy(), K_neigs=[13], is_probH=False)
#     G_mi_Km = ConstructHW.constructHW_dbscan(
#         concat_mi_tensor.cpu().numpy(), eps=best_eps_mi, min_samples=best_min_samples_mi, is_probH=False
#     )
#     G_mi_Kn, G_mi_Km = G_mi_Kn.to(device), G_mi_Km.to(device)
#
#     concat_dis = np.hstack([train_data[4].numpy().T, dis_sim_integrate_tensor.detach().cpu().numpy()])
#     concat_dis_tensor = torch.FloatTensor(concat_dis).to(device)
#
#     if USE_GRID_SEARCH:
#         best_eps_dis, best_min_samples_dis = find_best_dbscan_params(
#             concat_dis_tensor.cpu().numpy(),
#             eps_values=np.linspace(0.1, 1.0, 10),
#             min_samples_values=[5, 10, 15]
#         )
#     else:
#         best_eps_dis = find_optimal_eps(concat_dis_tensor.cpu().numpy(), min_samples=10, quantile=0.95)
#         best_min_samples_dis = 10
#
#     G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.cpu().numpy(), K_neigs=[13], is_probH=False)
#     G_dis_Km = ConstructHW.constructHW_dbscan(
#         concat_dis_tensor.cpu().numpy(), eps=best_eps_dis, min_samples=best_min_samples_dis, is_probH=False
#     )
#     G_dis_Kn, G_dis_Km = G_dis_Kn.to(device), G_dis_Km.to(device)
#
#     for epoch in range(1, opt.epoch + 1):
#         # 模型现在只返回score，不再返回mi_cl_loss和dis_cl_loss
#         score = model(concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
#         recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)
#         reg_loss = get_L2reg(model.parameters())
#         # 移除对比学习损失
#         tol_loss = recover_loss + 0.00001 * reg_loss
#         optim.zero_grad()
#         tol_loss.backward()
#         optim.step()
#
#     true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(
#         model, train_data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km
#     )
#
#     return true_value_one, true_value_zero, pre_value_one, pre_value_zero
#
#
# def test(model, data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km):
#     model.eval()
#     # 模型现在只返回score
#     score = model(concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
#     test_one_index = data[3][0].t().tolist()
#     test_zero_index = data[3][1].t().tolist()
#     true_one = data[5][test_one_index]
#     true_zero = data[5][test_zero_index]
#
#     pre_one = score[test_one_index]
#     pre_zero = score[test_zero_index]
#
#     # 检查预测值是否包含NaN或inf
#     if torch.isnan(pre_one).any() or torch.isinf(pre_one).any():
#         print("警告: pre_one包含NaN或inf，将其替换为0")
#         pre_one = torch.nan_to_num(pre_one, nan=0.0, posinf=0.0, neginf=0.0)
#
#     if torch.isnan(pre_zero).any() or torch.isinf(pre_zero).any():
#         print("警告: pre_zero包含NaN或inf，将其替换为0")
#         pre_zero = torch.nan_to_num(pre_zero, nan=0.0, posinf=0.0, neginf=0.0)
#
#     return true_one, true_zero, pre_one, pre_zero
#
#
# def evaluate(true_one, true_zero, pre_one, pre_zero):
#     Metric = Metric_fun()
#     metrics_tensor = np.zeros((1, 7))
#     valid_runs = 0  # 记录有效运行次数
#
#     for seed in range(10):
#         test_po_num = true_one.shape[0]
#
#         # 检查true_zero中是否有足够的0值
#         zero_indices = np.where(true_zero == 0)[0]
#         if len(zero_indices) < test_po_num:
#             print(f"警告: 种子 {seed}: true_zero中没有足够的0值，跳过此次评估")
#             continue
#
#         # 随机选择负样本
#         np.random.seed(seed)
#         np.random.shuffle(zero_indices)
#         test_ne_index = zero_indices[:test_po_num]
#
#         eval_true_zero = true_zero[test_ne_index]
#         eval_true_data = torch.cat([true_one, eval_true_zero])
#
#         eval_pre_zero = pre_zero[test_ne_index]
#         eval_pre_data = torch.cat([pre_one, eval_pre_zero])
#
#         # 检查合并后的数据是否包含NaN或inf
#         if torch.isnan(eval_true_data).any() or torch.isinf(eval_true_data).any():
#             print(f"警告: 种子 {seed}: eval_true_data包含NaN或inf，跳过此次评估")
#             continue
#
#         if torch.isnan(eval_pre_data).any() or torch.isinf(eval_pre_data).any():
#             print(f"警告: 种子 {seed}: eval_pre_data包含NaN或inf，跳过此次评估")
#             continue
#
#         # 计算指标
#         metrics = Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data)
#
#         # 检查计算结果是否有效
#         if np.isnan(metrics).any() or np.isinf(metrics).any():
#             print(f"警告: 种子 {seed}: 计算的指标包含NaN或inf，跳过此次评估")
#             continue
#
#         metrics_tensor += metrics
#         valid_runs += 1
#
#     # 避免除以0
#     if valid_runs > 0:
#         metrics_tensor_avg = metrics_tensor / valid_runs
#     else:
#         print("警告: 没有有效的评估运行，返回全0指标")
#         metrics_tensor_avg = np.zeros((1, 7))
#
#     # 检查最终结果
#     if np.isnan(metrics_tensor_avg).any():
#         print("警告: 最终指标包含NaN，将其替换为0")
#         metrics_tensor_avg = np.nan_to_num(metrics_tensor_avg)
#
#     return metrics_tensor_avg
#
#
# # 修改后的随机森林训练函数 - 添加数据验证和错误处理
# def train_random_forest(pre_one, pre_zero, labels):
#     # 将张量转换为numpy数组
#     pre_one_np = pre_one.detach().cpu().numpy()
#     pre_zero_np = pre_zero.detach().cpu().numpy()
#
#     # 验证数据形状
#     print(f"pre_one shape: {pre_one_np.shape}")
#     print(f"pre_zero shape: {pre_zero_np.shape}")
#
#     # 确保数据至少有一个维度
#     if pre_one_np.size == 0 or pre_zero_np.size == 0:
#         print("警告: pre_one或pre_zero为空数组，跳过随机森林训练")
#         return None
#
#     # 重塑数据为二维数组
#     pre_one_reshaped = pre_one_np.reshape(-1, 1)
#     pre_zero_reshaped = pre_zero_np.reshape(-1, 1)
#
#     # 合并数据
#     X = np.concatenate([pre_one_reshaped, pre_zero_reshaped], axis=0)
#
#     # 验证合并后的数据形状
#     print(f"合并后的X形状: {X.shape}")
#
#     # 检查是否有NaN值
#     if np.isnan(X).any():
#         # 使用均值填充NaN值
#         imputer = SimpleImputer(strategy='mean')
#         X = imputer.fit_transform(X)
#
#     # 检查特征维度是否大于0
#     if X.shape[1] == 0:
#         print("警告: 特征维度为0，跳过随机森林训练")
#         return None
#
#     clf = RandomForestClassifier(random_state=42)
#     clf.fit(X, labels)
#     return clf
#
#
# # 修改后的随机森林评估函数 - 添加数据验证和错误处理
# def evaluate_random_forest(rf_model, pre_one, pre_zero, labels):
#     if rf_model is None:
#         return 0.0
#
#     # 将张量转换为numpy数组
#     pre_one_np = pre_one.detach().cpu().numpy()
#     pre_zero_np = pre_zero.detach().cpu().numpy()
#
#     # 确保数据至少有一个维度
#     if pre_one_np.size == 0 or pre_zero_np.size == 0:
#         print("警告: pre_one或pre_zero为空数组，返回0准确率")
#         return 0.0
#
#     # 重塑数据为二维数组
#     pre_one_reshaped = pre_one_np.reshape(-1, 1)
#     pre_zero_reshaped = pre_zero_np.reshape(-1, 1)
#
#     # 合并数据
#     X = np.concatenate([pre_one_reshaped, pre_zero_reshaped], axis=0)
#
#     # 检查是否有NaN值
#     if np.isnan(X).any():
#         # 使用均值填充NaN值
#         imputer = SimpleImputer(strategy='mean')
#         X = imputer.fit_transform(X)
#
#     preds = rf_model.predict(X)
#     return accuracy_score(labels, preds)
#
#
# def main(opt):
#     dataset = prepare_data(opt)
#     train_data = Dataset(opt, dataset)
#     metrics_cross = np.zeros((1, 7))
#
#     for i in range(opt.validation):
#         print(f"\n===== 验证轮次 {i + 1}/{opt.validation} =====")
#
#         hidden_list = [256, 256]
#         num_proj_hidden = 64
#
#         model = HGCLAMIR(args.mi_num, args.dis_num, hidden_list, num_proj_hidden, args)
#         model.to(device)
#
#         optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
#
#         print("开始训练...")
#         true_score_one, true_score_zero, pre_score_one, pre_score_zero = train_epoch(
#             model, train_data[i], optimizer, opt)
#
#         print("训练完成，开始评估...")
#
#         # 检查训练数据是否有效
#         if true_score_one.numel() == 0 or true_score_zero.numel() == 0:
#             print("警告: true_score_one或true_score_zero为空，跳过此轮评估")
#             continue
#
#         if pre_score_one.numel() == 0 or pre_score_zero.numel() == 0:
#             print("警告: pre_score_one或pre_score_zero为空，跳过此轮评估")
#             continue
#
#         train_labels = np.hstack([
#             np.ones(true_score_one.shape[0]),
#             np.zeros(true_score_zero.shape[0])
#         ])
#
#         rf_model = train_random_forest(pre_score_one, pre_score_zero, train_labels)
#         if rf_model is not None:
#             rf_accuracy = evaluate_random_forest(rf_model, pre_score_one, pre_score_zero, train_labels)
#             print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
#         else:
#             print("随机森林模型未训练，跳过评估")
#
#         metrics_value = evaluate(true_score_one, true_score_zero, pre_score_one, pre_score_zero)
#         print(f"此轮指标: {metrics_value}")
#
#         # 检查metrics_value是否有效
#         if np.isnan(metrics_value).all():
#             print("警告: 此轮指标全为NaN，跳过累加")
#             continue
#
#         metrics_cross += metrics_value
#
#     # 计算平均指标
#     if np.sum(metrics_cross) != 0:  # 检查是否有有效数据
#         metrics_cross_avg = metrics_cross / opt.validation
#     else:
#         print("警告: 所有轮次的指标均无效，返回全0结果")
#         metrics_cross_avg = np.zeros((1, 7))
#
#     print('metrics_avg:', metrics_cross_avg)
#
#
# if __name__ == '__main__':
#     args = parameter_parser()
#     main(args)



# # 移除 HGCN_Attention_mechanism 模块
# import torch
# from prepareData import prepare_data
# import numpy as np
# from torch import optim
# from param import parameter_parser
# from Module import HGCLAMIR
# from utils import get_L2reg, Myloss
# from Calculate_Metrics import Metric_fun
# from trainData import Dataset
# import ConstructHW
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import silhouette_score
# import warnings
#
# warnings.filterwarnings('ignore')
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# USE_GRID_SEARCH = True
#
# # 数据1、2
# def find_best_dbscan_params(data, eps_values=[0.3, 0.5, 0.7], min_samples_values=[5, 10, 15]):
#     best_eps, best_min_samples = None, None
#     best_score = -1
#     for eps in eps_values:
#         for min_samples in min_samples_values:
#             clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
#             labels = clustering.labels_
#             if len(set(labels)) > 1:
#                 score = silhouette_score(data, labels)
#                 if score > best_score:
#                     best_score = score
#                     best_eps, best_min_samples = eps, min_samples
#     return best_eps, best_min_samples if best_eps is not None else (0.5, 10)
#
# def find_optimal_eps(data, min_samples=10, quantile=0.95):
#     neighbors = NearestNeighbors(n_neighbors=min_samples)
#     neighbors.fit(data)
#     distances, _ = neighbors.kneighbors(data)
#     distances = np.sort(distances[:, min_samples - 1], axis=0)
#     return distances[int(len(distances) * quantile)]
#
# def train_epoch(model, train_data, optim, opt):
#     model.train()
#     regression_crit = Myloss()
#
#     one_index = train_data[2][0].to(device).t().tolist()
#     zero_index = train_data[2][1].to(device).t().tolist()
#
#     dis_sim_integrate_tensor = train_data[0].to(device)
#     mi_sim_integrate_tensor = train_data[1].to(device)
#
#     concat_miRNA = np.hstack([train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])
#     concat_mi_tensor = torch.FloatTensor(concat_miRNA).to(device)
#
#     if USE_GRID_SEARCH:
#         best_eps_mi, best_min_samples_mi = find_best_dbscan_params(
#             concat_mi_tensor.cpu().numpy(),
#             eps_values=np.linspace(0.1, 1.0, 10),
#             min_samples_values=[5, 10, 15]
#         )
#     else:
#         best_eps_mi = find_optimal_eps(concat_mi_tensor.cpu().numpy(), min_samples=10, quantile=0.95)
#         best_min_samples_mi = 10
#
#     G_mi_Kn = ConstructHW.constructHW_knn(concat_mi_tensor.cpu().numpy(), K_neigs=[13], is_probH=False)
#     G_mi_Km = ConstructHW.constructHW_dbscan(
#         concat_mi_tensor.cpu().numpy(), eps=best_eps_mi, min_samples=best_min_samples_mi, is_probH=False
#     )
#     G_mi_Kn, G_mi_Km = G_mi_Kn.to(device), G_mi_Km.to(device)
#
#     concat_dis = np.hstack([train_data[4].numpy().T, dis_sim_integrate_tensor.detach().cpu().numpy()])
#     concat_dis_tensor = torch.FloatTensor(concat_dis).to(device)
#
#     if USE_GRID_SEARCH:
#         best_eps_dis, best_min_samples_dis = find_best_dbscan_params(
#             concat_dis_tensor.cpu().numpy(),
#             eps_values=np.linspace(0.1, 1.0, 10),
#             min_samples_values=[5, 10, 15]
#         )
#     else:
#         best_eps_dis = find_optimal_eps(concat_dis_tensor.cpu().numpy(), min_samples=10, quantile=0.95)
#         best_min_samples_dis = 10
#
#     G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.cpu().numpy(), K_neigs=[13], is_probH=False)
#     G_dis_Km = ConstructHW.constructHW_dbscan(
#         concat_dis_tensor.cpu().numpy(), eps=best_eps_dis, min_samples=best_min_samples_dis, is_probH=False
#     )
#     G_dis_Kn, G_dis_Km = G_dis_Kn.to(device), G_dis_Km.to(device)
#
#     for epoch in range(1, opt.epoch + 1):
#         score, mi_cl_loss, dis_cl_loss = model(concat_mi_tensor, concat_dis_tensor,
#                                                G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
#         recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)
#         reg_loss = get_L2reg(model.parameters())
#         tol_loss = recover_loss + mi_cl_loss + dis_cl_loss + 0.00001 * reg_loss
#         optim.zero_grad()
#         tol_loss.backward()
#         optim.step()
#
#     true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(
#         model, train_data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km
#     )
#
#     return true_value_one, true_value_zero, pre_value_one, pre_value_zero
#
# def test(model, data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km):
#     model.eval()
#     score, _, _ = model(concat_mi_tensor, concat_dis_tensor,
#                         G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
#     test_one_index = data[3][0].t().tolist()
#     test_zero_index = data[3][1].t().tolist()
#     true_one = data[5][test_one_index]
#     true_zero = data[5][test_zero_index]
#
#     pre_one = score[test_one_index]
#     pre_zero = score[test_zero_index]
#
#     return true_one, true_zero, pre_one, pre_zero
#
# def evaluate(true_one, true_zero, pre_one, pre_zero):
#     Metric = Metric_fun()
#     metrics_tensor = np.zeros((1, 7))
#
#     for seed in range(10):
#         test_po_num = true_one.shape[0]
#         test_index = np.array(np.where(true_zero == 0))
#         np.random.seed(seed)
#         np.random.shuffle(test_index.T)
#         test_ne_index = tuple(test_index[:, :test_po_num])
#
#         eval_true_zero = true_zero[test_ne_index]
#         eval_true_data = torch.cat([true_one, eval_true_zero])
#
#         eval_pre_zero = pre_zero[test_ne_index]
#         eval_pre_data = torch.cat([pre_one, eval_pre_zero])
#
#         metrics_tensor += Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data)
#
#     metrics_tensor_avg = metrics_tensor / 10
#     return metrics_tensor_avg
#
# # 修改后的随机森林训练函数
# def train_random_forest(pre_one, pre_zero, labels):
#     X = np.concatenate([
#         pre_one.detach().cpu().numpy().reshape(-1, 1),
#         pre_zero.detach().cpu().numpy().reshape(-1, 1)
#     ], axis=0)
#     clf = RandomForestClassifier(random_state=42)
#     clf.fit(X, labels)
#     return clf
#
# # 修改后的随机森林评估函数
# def evaluate_random_forest(rf_model, pre_one, pre_zero, labels):
#     X = np.concatenate([
#         pre_one.detach().cpu().numpy().reshape(-1, 1),
#         pre_zero.detach().cpu().numpy().reshape(-1, 1)
#     ], axis=0)
#     preds = rf_model.predict(X)
#     return accuracy_score(labels, preds)
#
# def main(opt):
#     dataset = prepare_data(opt)
#     train_data = Dataset(opt, dataset)
#     metrics_cross = np.zeros((1, 7))
#
#     for i in range(opt.validation):
#         hidden_list = [256, 256]
#         num_proj_hidden = 64
#
#         model = HGCLAMIR(args.mi_num, args.dis_num, hidden_list, num_proj_hidden, args)
#         model.to(device)
#
#         optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
#
#         true_score_one, true_score_zero, pre_score_one, pre_score_zero = train_epoch(
#             model, train_data[i], optimizer, opt)
#
#         train_labels = np.hstack([
#             np.ones(true_score_one.shape[0]),
#             np.zeros(true_score_zero.shape[0])
#         ])
#
#         rf_model = train_random_forest(pre_score_one, pre_score_zero, train_labels)
#         rf_accuracy = evaluate_random_forest(rf_model, pre_score_one, pre_score_zero, train_labels)
#         print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
#
#         metrics_value = evaluate(true_score_one, true_score_zero, pre_score_one, pre_score_zero)
#         metrics_cross += metrics_value
#
#     metrics_cross_avg = metrics_cross / opt.validation
#     print('metrics_avg:', metrics_cross_avg)
#
# if __name__ == '__main__':
#     args = parameter_parser()
#     main(args)




# # 移除 TransformerEncoder 模块
# import torch
# from prepareData import prepare_data
# import numpy as np
# from torch import optim
# from param import parameter_parser
# from Module import HGCLAMIR  # 导入修改后的HGCLAMIR类
# from utils import get_L2reg, Myloss
# from Calculate_Metrics import Metric_fun
# from trainData import Dataset
# import ConstructHW
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import silhouette_score
# import warnings
#
# warnings.filterwarnings('ignore')
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# USE_GRID_SEARCH = True
#
# # 数据1、2
# def find_best_dbscan_params(data, eps_values=[0.3, 0.5, 0.7], min_samples_values=[5, 10, 15]):
#     best_eps, best_min_samples = None, None
#     best_score = -1
#     for eps in eps_values:
#         for min_samples in min_samples_values:
#             clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
#             labels = clustering.labels_
#             if len(set(labels)) > 1:
#                 score = silhouette_score(data, labels)
#                 if score > best_score:
#                     best_score = score
#                     best_eps, best_min_samples = eps, min_samples
#     return best_eps, best_min_samples if best_eps is not None else (0.5, 10)
#
# def find_optimal_eps(data, min_samples=10, quantile=0.95):
#     neighbors = NearestNeighbors(n_neighbors=min_samples)
#     neighbors.fit(data)
#     distances, _ = neighbors.kneighbors(data)
#     distances = np.sort(distances[:, min_samples - 1], axis=0)
#     return distances[int(len(distances) * quantile)]
#
# def train_epoch(model, train_data, optim, opt):
#     model.train()
#     regression_crit = Myloss()
#
#     one_index = train_data[2][0].to(device).t().tolist()
#     zero_index = train_data[2][1].to(device).t().tolist()
#
#     dis_sim_integrate_tensor = train_data[0].to(device)
#     mi_sim_integrate_tensor = train_data[1].to(device)
#
#     concat_miRNA = np.hstack([train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])
#     concat_mi_tensor = torch.FloatTensor(concat_miRNA).to(device)
#
#     if USE_GRID_SEARCH:
#         best_eps_mi, best_min_samples_mi = find_best_dbscan_params(
#             concat_mi_tensor.cpu().numpy(),
#             eps_values=np.linspace(0.1, 1.0, 10),
#             min_samples_values=[5, 10, 15]
#         )
#     else:
#         best_eps_mi = find_optimal_eps(concat_mi_tensor.cpu().numpy(), min_samples=10, quantile=0.95)
#         best_min_samples_mi = 10
#
#     G_mi_Kn = ConstructHW.constructHW_knn(concat_mi_tensor.cpu().numpy(), K_neigs=[13], is_probH=False)
#     G_mi_Km = ConstructHW.constructHW_dbscan(
#         concat_mi_tensor.cpu().numpy(), eps=best_eps_mi, min_samples=best_min_samples_mi, is_probH=False
#     )
#     G_mi_Kn, G_mi_Km = G_mi_Kn.to(device), G_mi_Km.to(device)
#
#     concat_dis = np.hstack([train_data[4].numpy().T, dis_sim_integrate_tensor.detach().cpu().numpy()])
#     concat_dis_tensor = torch.FloatTensor(concat_dis).to(device)
#
#     if USE_GRID_SEARCH:
#         best_eps_dis, best_min_samples_dis = find_best_dbscan_params(
#             concat_dis_tensor.cpu().numpy(),
#             eps_values=np.linspace(0.1, 1.0, 10),
#             min_samples_values=[5, 10, 15]
#         )
#     else:
#         best_eps_dis = find_optimal_eps(concat_dis_tensor.cpu().numpy(), min_samples=10, quantile=0.95)
#         best_min_samples_dis = 10
#
#     G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.cpu().numpy(), K_neigs=[13], is_probH=False)
#     G_dis_Km = ConstructHW.constructHW_dbscan(
#         concat_dis_tensor.cpu().numpy(), eps=best_eps_dis, min_samples=best_min_samples_dis, is_probH=False
#     )
#     G_dis_Kn, G_dis_Km = G_dis_Kn.to(device), G_dis_Km.to(device)
#
#     for epoch in range(1, opt.epoch + 1):
#         score, mi_cl_loss, dis_cl_loss = model(concat_mi_tensor, concat_dis_tensor,
#                                                G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
#         recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)
#         reg_loss = get_L2reg(model.parameters())
#         tol_loss = recover_loss + mi_cl_loss + dis_cl_loss + 0.00001 * reg_loss
#         optim.zero_grad()
#         tol_loss.backward()
#         optim.step()
#
#     true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(
#         model, train_data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km
#     )
#
#     return true_value_one, true_value_zero, pre_value_one, pre_value_zero
#
# def test(model, data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km):
#     model.eval()
#     score, _, _ = model(concat_mi_tensor, concat_dis_tensor,
#                         G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
#     test_one_index = data[3][0].t().tolist()
#     test_zero_index = data[3][1].t().tolist()
#     true_one = data[5][test_one_index]
#     true_zero = data[5][test_zero_index]
#
#     pre_one = score[test_one_index]
#     pre_zero = score[test_zero_index]
#
#     return true_one, true_zero, pre_one, pre_zero
#
# def evaluate(true_one, true_zero, pre_one, pre_zero):
#     Metric = Metric_fun()
#     metrics_tensor = np.zeros((1, 7))
#
#     for seed in range(10):
#         test_po_num = true_one.shape[0]
#         test_index = np.array(np.where(true_zero == 0))
#         np.random.seed(seed)
#         np.random.shuffle(test_index.T)
#         test_ne_index = tuple(test_index[:, :test_po_num])
#
#         eval_true_zero = true_zero[test_ne_index]
#         eval_true_data = torch.cat([true_one, eval_true_zero])
#
#         eval_pre_zero = pre_zero[test_ne_index]
#         eval_pre_data = torch.cat([pre_one, eval_pre_zero])
#
#         metrics_tensor += Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data)
#
#     metrics_tensor_avg = metrics_tensor / 10
#     return metrics_tensor_avg
#
# # 修改后的随机森林训练函数
# def train_random_forest(pre_one, pre_zero, labels):
#     X = np.concatenate([
#         pre_one.detach().cpu().numpy().reshape(-1, 1),
#         pre_zero.detach().cpu().numpy().reshape(-1, 1)
#     ], axis=0)
#     clf = RandomForestClassifier(random_state=42)
#     clf.fit(X, labels)
#     return clf
#
# # 修改后的随机森林评估函数
# def evaluate_random_forest(rf_model, pre_one, pre_zero, labels):
#     X = np.concatenate([
#         pre_one.detach().cpu().numpy().reshape(-1, 1),
#         pre_zero.detach().cpu().numpy().reshape(-1, 1)
#     ], axis=0)
#     preds = rf_model.predict(X)
#     return accuracy_score(labels, preds)
#
# def main(opt):
#     dataset = prepare_data(opt)
#     train_data = Dataset(opt, dataset)
#     metrics_cross = np.zeros((1, 7))
#
#     for i in range(opt.validation):
#         hidden_list = [256, 256]  # 保持与修改前一致
#         num_proj_hidden = 64      # 保持与修改前一致
#
#         model = HGCLAMIR(args.mi_num, args.dis_num, hidden_list, num_proj_hidden, args)
#         model.to(device)
#
#         optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
#
#         true_score_one, true_score_zero, pre_score_one, pre_score_zero = train_epoch(
#             model, train_data[i], optimizer, opt)
#
#         train_labels = np.hstack([
#             np.ones(true_score_one.shape[0]),
#             np.zeros(true_score_zero.shape[0])
#         ])
#
#         rf_model = train_random_forest(pre_score_one, pre_score_zero, train_labels)
#         rf_accuracy = evaluate_random_forest(rf_model, pre_score_one, pre_score_zero, train_labels)
#         print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
#
#         metrics_value = evaluate(true_score_one, true_score_zero, pre_score_one, pre_score_zero)
#         metrics_cross += metrics_value
#
#     metrics_cross_avg = metrics_cross / opt.validation
#     print('metrics_avg:', metrics_cross_avg)
#
# if __name__ == '__main__':
#     args = parameter_parser()
#     main(args)





# # 源代码1
# import torch
# from prepareData import prepare_data
# import numpy as np
# from torch import optim
# from param import parameter_parser
# from Module import HGCLAMIR
# from utils import get_L2reg, Myloss
# from Calculate_Metrics import Metric_fun
# from trainData import Dataset
# import ConstructHW
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import silhouette_score
# import warnings
#
# warnings.filterwarnings('ignore')
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# USE_GRID_SEARCH = True
#
# # 数据1、2
# def find_best_dbscan_params(data, eps_values=[0.3, 0.5, 0.7], min_samples_values=[5, 10, 15]):
#     best_eps, best_min_samples = None, None
#     best_score = -1
#     for eps in eps_values:
#         for min_samples in min_samples_values:
#             clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
#             labels = clustering.labels_
#             if len(set(labels)) > 1:
#                 score = silhouette_score(data, labels)
#                 if score > best_score:
#                     best_score = score
#                     best_eps, best_min_samples = eps, min_samples
#     return best_eps, best_min_samples if best_eps is not None else (0.5, 10)
#
#
# # # 数据3、4
# # def find_best_dbscan_params(data, eps_values=[0.3, 0.5, 0.7], min_samples_values=[5, 10, 15]):
# #     best_eps, best_min_samples = None, None
# #     best_score = -1
# #     for eps in eps_values:
# #         for min_samples in min_samples_values:
# #             clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
# #             labels = clustering.labels_
# #             if len(set(labels)) > 1 and -1 not in set(labels):  # 可选增强条件：排除噪声簇
# #                 score = silhouette_score(data, labels)
# #                 if score > best_score:
# #                     best_score = score
# #                     best_eps, best_min_samples = eps, min_samples
# #     if best_eps is not None and best_min_samples is not None:
# #         return best_eps, best_min_samples
# #     else:
# #         return 0.5, 10  # 返回默认值，避免None传入DBSCAN
#
#
#
#
# def find_optimal_eps(data, min_samples=10, quantile=0.95):
#     neighbors = NearestNeighbors(n_neighbors=min_samples)
#     neighbors.fit(data)
#     distances, _ = neighbors.kneighbors(data)
#     distances = np.sort(distances[:, min_samples - 1], axis=0)
#     return distances[int(len(distances) * quantile)]
#
#
# def train_epoch(model, train_data, optim, opt):
#     model.train()
#     regression_crit = Myloss()
#
#     one_index = train_data[2][0].to(device).t().tolist()
#     zero_index = train_data[2][1].to(device).t().tolist()
#
#     dis_sim_integrate_tensor = train_data[0].to(device)
#     mi_sim_integrate_tensor = train_data[1].to(device)
#
#     concat_miRNA = np.hstack([train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])
#     concat_mi_tensor = torch.FloatTensor(concat_miRNA).to(device)
#
#     if USE_GRID_SEARCH:
#         best_eps_mi, best_min_samples_mi = find_best_dbscan_params(
#             concat_mi_tensor.cpu().numpy(),
#             eps_values=np.linspace(0.1, 1.0, 10),
#             min_samples_values=[5, 10, 15]
#         )
#     else:
#         best_eps_mi = find_optimal_eps(concat_mi_tensor.cpu().numpy(), min_samples=10, quantile=0.95)
#         best_min_samples_mi = 10
#
#     G_mi_Kn = ConstructHW.constructHW_knn(concat_mi_tensor.cpu().numpy(), K_neigs=[13], is_probH=False)
#     G_mi_Km = ConstructHW.constructHW_dbscan(
#         concat_mi_tensor.cpu().numpy(), eps=best_eps_mi, min_samples=best_min_samples_mi, is_probH=False
#     )
#     G_mi_Kn, G_mi_Km = G_mi_Kn.to(device), G_mi_Km.to(device)
#
#     concat_dis = np.hstack([train_data[4].numpy().T, dis_sim_integrate_tensor.detach().cpu().numpy()])
#     concat_dis_tensor = torch.FloatTensor(concat_dis).to(device)
#
#     if USE_GRID_SEARCH:
#         best_eps_dis, best_min_samples_dis = find_best_dbscan_params(
#             concat_dis_tensor.cpu().numpy(),
#             eps_values=np.linspace(0.1, 1.0, 10),
#             min_samples_values=[5, 10, 15]
#         )
#     else:
#         best_eps_dis = find_optimal_eps(concat_dis_tensor.cpu().numpy(), min_samples=10, quantile=0.95)
#         best_min_samples_dis = 10
#
#     G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.cpu().numpy(), K_neigs=[13], is_probH=False)
#     G_dis_Km = ConstructHW.constructHW_dbscan(
#         concat_dis_tensor.cpu().numpy(), eps=best_eps_dis, min_samples=best_min_samples_dis, is_probH=False
#     )
#     G_dis_Kn, G_dis_Km = G_dis_Kn.to(device), G_dis_Km.to(device)
#
#     for epoch in range(1, opt.epoch + 1):
#         score, mi_cl_loss, dis_cl_loss = model(concat_mi_tensor, concat_dis_tensor,
#                                                G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
#         recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)
#         reg_loss = get_L2reg(model.parameters())
#         tol_loss = recover_loss + mi_cl_loss + dis_cl_loss + 0.00001 * reg_loss
#         optim.zero_grad()
#         tol_loss.backward()
#         optim.step()
#
#     true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(
#         model, train_data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km
#     )
#
#     return true_value_one, true_value_zero, pre_value_one, pre_value_zero
#
#
# def test(model, data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km):
#     model.eval()
#     score, _, _ = model(concat_mi_tensor, concat_dis_tensor,
#                         G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
#     test_one_index = data[3][0].t().tolist()
#     test_zero_index = data[3][1].t().tolist()
#     true_one = data[5][test_one_index]
#     true_zero = data[5][test_zero_index]
#
#     pre_one = score[test_one_index]
#     pre_zero = score[test_zero_index]
#
#     return true_one, true_zero, pre_one, pre_zero
#
#
# def evaluate(true_one, true_zero, pre_one, pre_zero):
#     Metric = Metric_fun()
#     metrics_tensor = np.zeros((1, 7))
#
#     for seed in range(10):
#         test_po_num = true_one.shape[0]
#         test_index = np.array(np.where(true_zero == 0))
#         np.random.seed(seed)
#         np.random.shuffle(test_index.T)
#         test_ne_index = tuple(test_index[:, :test_po_num])
#
#         eval_true_zero = true_zero[test_ne_index]
#         eval_true_data = torch.cat([true_one, eval_true_zero])
#
#         eval_pre_zero = pre_zero[test_ne_index]
#         eval_pre_data = torch.cat([pre_one, eval_pre_zero])
#
#         metrics_tensor += Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data)
#
#     metrics_tensor_avg = metrics_tensor / 10
#     return metrics_tensor_avg
#
#
# #  修改后的随机森林训练函数
# def train_random_forest(pre_one, pre_zero, labels):
#     X = np.concatenate([
#         pre_one.detach().cpu().numpy().reshape(-1, 1),
#         pre_zero.detach().cpu().numpy().reshape(-1, 1)
#     ], axis=0)
#     clf = RandomForestClassifier(random_state=42)
#     clf.fit(X, labels)
#     return clf
#
#
# #  修改后的随机森林评估函数
# def evaluate_random_forest(rf_model, pre_one, pre_zero, labels):
#     X = np.concatenate([
#         pre_one.detach().cpu().numpy().reshape(-1, 1),
#         pre_zero.detach().cpu().numpy().reshape(-1, 1)
#     ], axis=0)
#     preds = rf_model.predict(X)
#     return accuracy_score(labels, preds)
#
#
# def main(opt):
#     dataset = prepare_data(opt)
#     train_data = Dataset(opt, dataset)
#     metrics_cross = np.zeros((1, 7))
#
#     for i in range(opt.validation):
#         hidden_list = [256, 256]
#         num_proj_hidden = 64
#
#         model = HGCLAMIR(args.mi_num, args.dis_num, hidden_list, num_proj_hidden, args)
#         model.to(device)
#
#         optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
#
#         true_score_one, true_score_zero, pre_score_one, pre_score_zero = train_epoch(
#             model, train_data[i], optimizer, opt)
#
#         train_labels = np.hstack([
#             np.ones(true_score_one.shape[0]),
#             np.zeros(true_score_zero.shape[0])
#         ])
#
#         rf_model = train_random_forest(pre_score_one, pre_score_zero, train_labels)
#         rf_accuracy = evaluate_random_forest(rf_model, pre_score_one, pre_score_zero, train_labels)
#         print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
#
#         metrics_value = evaluate(true_score_one, true_score_zero, pre_score_one, pre_score_zero)
#         metrics_cross += metrics_value
#
#     metrics_cross_avg = metrics_cross / opt.validation
#     print('metrics_avg:', metrics_cross_avg)
#
#
# if __name__ == '__main__':
#     args = parameter_parser()
#     main(args)




# 原代码2
import torch
from prepareData import prepare_data
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from param import parameter_parser
from Module import HGCLAMIR
from utils import get_L2reg, Myloss
from Calculate_Metrics import Metric_fun
from trainData import Dataset
import ConstructHW
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import warnings
import copy
import os

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_GRID_SEARCH = True
EARLY_STOPPING_PATIENCE = 10  # 早停机制：如果验证集性能在10个epoch内没有提升，则停止训练


# 增强型DBSCAN参数搜索，使用更广泛的参数范围和评分方法
def find_best_dbscan_params(data, eps_min=0.1, eps_max=2.0, eps_steps=20, min_samples_min=3, min_samples_max=20):
    best_eps, best_min_samples = None, None
    best_score = -1
    best_labels = None

    # 对数据进行标准化，有助于DBSCAN聚类
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    eps_values = np.linspace(eps_min, eps_max, eps_steps)
    min_samples_values = range(min_samples_min, min_samples_max + 1)

    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                # 执行DBSCAN聚类
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data_scaled)
                labels = clustering.labels_

                # 计算聚类数量（排除噪声点）
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                # 跳过无效的聚类结果
                if n_clusters <= 1:
                    continue

                # 计算噪声点比例
                noise_ratio = list(labels).count(-1) / len(labels)

                # 如果噪声点比例过高，认为聚类效果不佳
                if noise_ratio > 0.5:
                    continue

                # 计算轮廓系数
                score = silhouette_score(data_scaled, labels)

                # 记录最佳参数
                if score > best_score:
                    best_score = score
                    best_eps, best_min_samples = eps, min_samples
                    best_labels = labels
            except Exception as e:
                print(f"DBSCAN参数 {eps}, {min_samples} 出错: {e}")
                continue

    # 如果没有找到合适的参数，使用启发式方法确定
    if best_eps is None:
        print("警告: 未能找到合适的DBSCAN参数，使用启发式方法...")
        best_eps = find_optimal_eps(data_scaled)
        best_min_samples = 10

    print(f"最佳DBSCAN参数: eps={best_eps}, min_samples={best_min_samples}, 轮廓系数={best_score:.4f}")
    return best_eps, best_min_samples


# 使用k-距离图来估计最佳eps参数
def find_optimal_eps(data, min_samples=10):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(data)
    distances, _ = neighbors.kneighbors(data)
    distances = np.sort(distances[:, min_samples - 1], axis=0)

    # 寻找曲线的"拐点"
    diff = np.diff(distances)
    max_diff_idx = np.argmax(diff)
    eps = distances[max_diff_idx]

    print(f"估计的最佳eps: {eps:.4f}")
    return eps


# 标准化数据函数
def standardize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


# 训练函数 - 增强版
def train_epoch(model, train_data, optimizer, opt, scheduler=None, val_data=None):
    model.train()
    regression_crit = Myloss()

    # 数据准备
    one_index = train_data[2][0].to(device).t().tolist()
    zero_index = train_data[2][1].to(device).t().tolist()

    dis_sim_integrate_tensor = train_data[0].to(device)
    mi_sim_integrate_tensor = train_data[1].to(device)

    # 标准化特征
    concat_miRNA = np.hstack([train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])
    concat_miRNA_std = standardize_data(concat_miRNA)
    concat_mi_tensor = torch.FloatTensor(concat_miRNA_std).to(device)

    # 优化DBSCAN参数搜索
    if USE_GRID_SEARCH:
        best_eps_mi, best_min_samples_mi = find_best_dbscan_params(
            concat_mi_tensor.cpu().numpy()
        )
    else:
        best_eps_mi = find_optimal_eps(concat_mi_tensor.cpu().numpy())
        best_min_samples_mi = 10

    # 构建异构图
    G_mi_Kn = ConstructHW.constructHW_knn(concat_mi_tensor.cpu().numpy(), K_neigs=[13], is_probH=False)
    # G_mi_Kn = ConstructHW.constructHW_kmean(concat_mi_tensor.detach().cpu().numpy(), clusters=[9])
    # G_mi_Kn = ConstructHW.constructHW_kmedoids(concat_mi_tensor.cpu().numpy(), n_clusters=13, is_probH=False)
    G_mi_Km = ConstructHW.constructHW_dbscan(
        concat_mi_tensor.cpu().numpy(), eps=best_eps_mi, min_samples=best_min_samples_mi, is_probH=False
    )
    G_mi_Kn, G_mi_Km = G_mi_Kn.to(device), G_mi_Km.to(device)

    # 处理疾病数据
    concat_dis = np.hstack([train_data[4].numpy().T, dis_sim_integrate_tensor.detach().cpu().numpy()])
    concat_dis_std = standardize_data(concat_dis)
    concat_dis_tensor = torch.FloatTensor(concat_dis_std).to(device)

    # 优化DBSCAN参数搜索
    if USE_GRID_SEARCH:
        best_eps_dis, best_min_samples_dis = find_best_dbscan_params(
            concat_dis_tensor.cpu().numpy()
        )
    else:
        best_eps_dis = find_optimal_eps(concat_dis_tensor.cpu().numpy())
        best_min_samples_dis = 10

    # 构建异构图
    G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.cpu().numpy(), K_neigs=[13], is_probH=False)
    # G_dis_Kn = ConstructHW.constructHW_kmean(concat_dis_tensor.detach().cpu().numpy(), clusters=[9])
    # G_dis_Kn = ConstructHW.constructHW_kmedoids(concat_dis_tensor.cpu().numpy(), n_clusters=13, is_probH=False)
    G_dis_Km = ConstructHW.constructHW_dbscan(
        concat_dis_tensor.cpu().numpy(), eps=best_eps_dis, min_samples=best_min_samples_dis, is_probH=False
    )
    G_dis_Kn, G_dis_Km = G_dis_Kn.to(device), G_dis_Km.to(device)

    # 早停机制
    best_loss = float('inf')
    best_model = None
    patience = EARLY_STOPPING_PATIENCE

    for epoch in range(1, opt.epoch + 1):
        model.train()

        # 前向传播
        score, mi_cl_loss, dis_cl_loss = model(concat_mi_tensor, concat_dis_tensor,
                                               G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)

        # 计算损失
        recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)
        reg_loss = get_L2reg(model.parameters())

        # 确保对比学习损失有效
        if torch.isnan(mi_cl_loss) or torch.isinf(mi_cl_loss):
            mi_cl_loss = torch.tensor(0.0, device=device)

        if torch.isnan(dis_cl_loss) or torch.isinf(dis_cl_loss):
            dis_cl_loss = torch.tensor(0.0, device=device)

        # 总损失 - 恢复损失 + 对比学习损失 + 正则化
        tol_loss = recover_loss + 0.1 * (mi_cl_loss + dis_cl_loss) + 0.00001 * reg_loss

        # 反向传播和优化
        optimizer.zero_grad()
        tol_loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 学习率调度
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(tol_loss)
            else:
                scheduler.step()

        # 打印训练信息
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{opt.epoch}, 恢复损失: {recover_loss.item():.6f}, "
                  f"对比学习损失: {mi_cl_loss.item():.6f} + {dis_cl_loss.item():.6f}, "
                  f"正则化损失: {reg_loss.item():.6f}, 总损失: {tol_loss.item():.6f}")

        # 早停机制
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_score, _, _ = model(concat_mi_tensor, concat_dis_tensor,
                                        G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
                val_loss = regression_crit(one_index, zero_index, train_data[4].to(device), val_score)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model)
                patience = EARLY_STOPPING_PATIENCE
            else:
                patience -= 1
                if patience <= 0:
                    print(f"早停触发: 在第 {epoch} 轮停止训练")
                    model = best_model
                    break

    # 最终评估
    model.eval()
    with torch.no_grad():
        score, _, _ = model(concat_mi_tensor, concat_dis_tensor,
                            G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)

    true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(
        model, train_data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km
    )

    return true_value_one, true_value_zero, pre_value_one, pre_value_zero


def test(model, data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km):
    model.eval()
    with torch.no_grad():
        score, _, _ = model(concat_mi_tensor, concat_dis_tensor,
                            G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)

    test_one_index = data[3][0].t().tolist()
    test_zero_index = data[3][1].t().tolist()
    true_one = data[5][test_one_index]
    true_zero = data[5][test_zero_index]

    pre_one = score[test_one_index]
    pre_zero = score[test_zero_index]

    # 检查预测值是否包含NaN或inf
    if torch.isnan(pre_one).any() or torch.isinf(pre_one).any():
        print("警告: pre_one包含NaN或inf，将其替换为0")
        pre_one = torch.nan_to_num(pre_one, nan=0.0, posinf=0.0, neginf=0.0)

    if torch.isnan(pre_zero).any() or torch.isinf(pre_zero).any():
        print("警告: pre_zero包含NaN或inf，将其替换为0")
        pre_zero = torch.nan_to_num(pre_zero, nan=0.0, posinf=0.0, neginf=0.0)

    return true_one, true_zero, pre_one, pre_zero


# 增强版评估函数 - 计算更多指标
def evaluate(true_one, true_zero, pre_one, pre_zero):
    Metric = Metric_fun()
    metrics_tensor = np.zeros((1, 7))
    auc_scores = []
    aupr_scores = []

    valid_runs = 0  # 记录有效运行次数

    for seed in range(10):
        test_po_num = true_one.shape[0]

        # 检查true_zero中是否有足够的0值
        zero_indices = np.where(true_zero == 0)[0]
        if len(zero_indices) < test_po_num:
            print(f"警告: 种子 {seed}: true_zero中没有足够的0值，跳过此次评估")
            continue

        # 随机选择负样本
        np.random.seed(seed)
        np.random.shuffle(zero_indices)
        test_ne_index = zero_indices[:test_po_num]

        eval_true_zero = true_zero[test_ne_index]
        eval_true_data = torch.cat([true_one, eval_true_zero])

        eval_pre_zero = pre_zero[test_ne_index]
        eval_pre_data = torch.cat([pre_one, eval_pre_zero])

        # 检查合并后的数据是否包含NaN或inf
        if torch.isnan(eval_true_data).any() or torch.isinf(eval_true_data).any():
            print(f"警告: 种子 {seed}: eval_true_data包含NaN或inf，跳过此次评估")
            continue

        if torch.isnan(eval_pre_data).any() or torch.isinf(eval_pre_data).any():
            print(f"警告: 种子 {seed}: eval_pre_data包含NaN或inf，跳过此次评估")
            continue

        # 计算指标
        metrics = Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data)

        # 计算AUC和AUPR
        try:
            y_true = eval_true_data.cpu().numpy()
            y_score = eval_pre_data.cpu().numpy()
            auc_score = roc_auc_score(y_true, y_score)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            aupr_score = auc(recall, precision)

            auc_scores.append(auc_score)
            aupr_scores.append(aupr_score)

            print(f"种子 {seed}: AUC={auc_score:.4f}, AUPR={aupr_score:.4f}")
        except Exception as e:
            print(f"计算AUC/AUPR出错: {e}")
            auc_scores.append(0.0)
            aupr_scores.append(0.0)

        # 检查计算结果是否有效
        if np.isnan(metrics).any() or np.isinf(metrics).any():
            print(f"警告: 种子 {seed}: 计算的指标包含NaN或inf，跳过此次评估")
            continue

        metrics_tensor += metrics
        valid_runs += 1

    # 避免除以0
    if valid_runs > 0:
        metrics_tensor_avg = metrics_tensor / valid_runs
        avg_auc = np.mean(auc_scores)
        avg_aupr = np.mean(aupr_scores)
        print(f"平均AUC: {avg_auc:.4f}, 平均AUPR: {avg_aupr:.4f}")
    else:
        print("警告: 没有有效的评估运行，返回全0指标")
        metrics_tensor_avg = np.zeros((1, 7))
        avg_auc = 0.0
        avg_aupr = 0.0

    # 检查最终结果
    if np.isnan(metrics_tensor_avg).any():
        print("警告: 最终指标包含NaN，将其替换为0")
        metrics_tensor_avg = np.nan_to_num(metrics_tensor_avg)

    return metrics_tensor_avg


# 增强版随机森林训练函数 - 添加数据验证和特征工程
def train_random_forest(pre_one, pre_zero, labels):
    # 将张量转换为numpy数组
    pre_one_np = pre_one.detach().cpu().numpy()
    pre_zero_np = pre_zero.detach().cpu().numpy()

    # 验证数据形状
    print(f"pre_one shape: {pre_one_np.shape}")
    print(f"pre_zero shape: {pre_zero_np.shape}")

    # 确保数据至少有一个维度
    if pre_one_np.size == 0 or pre_zero_np.size == 0:
        print("警告: pre_one或pre_zero为空数组，跳过随机森林训练")
        return None

    # 重塑数据为二维数组
    pre_one_reshaped = pre_one_np.reshape(-1, 1)
    pre_zero_reshaped = pre_zero_np.reshape(-1, 1)

    # 合并数据
    X = np.concatenate([pre_one_reshaped, pre_zero_reshaped], axis=0)

    # 验证合并后的数据形状
    print(f"合并后的X形状: {X.shape}")

    # 检查是否有NaN值
    if np.isnan(X).any():
        print("警告: 数据包含NaN值，使用中位数填充")
        # 使用中位数填充NaN值（比均值更鲁棒）
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 检查特征维度是否大于0
    if X_scaled.shape[1] == 0:
        print("警告: 特征维度为0，跳过随机森林训练")
        return None

    # 使用更优的随机森林参数
    clf = RandomForestClassifier(
        n_estimators=500,  # 增加树的数量
        max_depth=10,  # 限制树的深度，防止过拟合
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # 使用所有CPU核心
    )

    clf.fit(X_scaled, labels)
    return clf


# 增强版随机森林评估函数
def evaluate_random_forest(rf_model, pre_one, pre_zero, labels):
    if rf_model is None:
        return 0.0

    # 将张量转换为numpy数组
    pre_one_np = pre_one.detach().cpu().numpy()
    pre_zero_np = pre_zero.detach().cpu().numpy()

    # 确保数据至少有一个维度
    if pre_one_np.size == 0 or pre_zero_np.size == 0:
        print("警告: pre_one或pre_zero为空数组，返回0准确率")
        return 0.0

    # 重塑数据为二维数组
    pre_one_reshaped = pre_one_np.reshape(-1, 1)
    pre_zero_reshaped = pre_zero_np.reshape(-1, 1)

    # 合并数据
    X = np.concatenate([pre_one_reshaped, pre_zero_reshaped], axis=0)

    # 检查是否有NaN值
    if np.isnan(X).any():
        print("警告: 数据包含NaN值，使用中位数填充")
        # 使用中位数填充NaN值
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)

    # 特征标准化（使用训练时的scaler）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 计算预测概率而不是硬分类
    if hasattr(rf_model, "predict_proba"):
        probas = rf_model.predict_proba(X_scaled)[:, 1]
        # 使用AUC作为评估指标，比准确率更适合不平衡数据
        auc_score = roc_auc_score(labels, probas)
        print(f"随机森林AUC: {auc_score:.4f}")
        return auc_score
    else:
        preds = rf_model.predict(X_scaled)
        return accuracy_score(labels, preds)


def main(opt):
    # 创建保存模型的目录
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    dataset = prepare_data(opt)
    train_data = Dataset(opt, dataset)
    metrics_cross = np.zeros((1, 7))

    for i in range(opt.validation):
        print(f"\n===== 验证轮次 {i + 1}/{opt.validation} =====")

        hidden_list = [256, 256]  # 模型隐藏层大小
        num_proj_hidden = 64  # 投影头隐藏层大小

        model = HGCLAMIR(args.mi_num, args.dis_num, hidden_list, num_proj_hidden, args)
        model.to(device)

        # 优化器 - 使用更优的学习率和权重衰减
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)

        # 学习率调度 - 当指标停滞时降低学习率
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )

        print("开始训练...")
        true_score_one, true_score_zero, pre_score_one, pre_score_zero = train_epoch(
            model, train_data[i], optimizer, opt, scheduler
        )

        print("训练完成，开始评估...")

        # 检查训练数据是否有效
        if true_score_one.numel() == 0 or true_score_zero.numel() == 0:
            print("警告: true_score_one或true_score_zero为空，跳过此轮评估")
            continue

        if pre_score_one.numel() == 0 or pre_score_zero.numel() == 0:
            print("警告: pre_score_one或pre_score_zero为空，跳过此轮评估")
            continue

        # 训练标签
        train_labels = np.hstack([
            np.ones(true_score_one.shape[0]),
            np.zeros(true_score_zero.shape[0])
        ])

        # 训练随机森林模型
        rf_model = train_random_forest(pre_score_one, pre_score_zero, train_labels)

        if rf_model is not None:
            rf_auc = evaluate_random_forest(rf_model, pre_score_one, pre_score_zero, train_labels)
            print(f"Random Forest AUC: {rf_auc * 100:.2f}%")
        else:
            print("随机森林模型未训练，跳过评估")

        # 评估模型
        metrics_value = evaluate(true_score_one, true_score_zero, pre_score_one, pre_score_zero)
        print(f"此轮指标: {metrics_value}")

        # 检查metrics_value是否有效
        if np.isnan(metrics_value).all():
            print("警告: 此轮指标全为NaN，跳过累加")
            continue

        metrics_cross += metrics_value

        # 保存模型
        model_path = f'saved_models/model_fold_{i + 1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至 {model_path}")

    # 计算平均指标
    if np.sum(metrics_cross) != 0:  # 检查是否有有效数据
        metrics_cross_avg = metrics_cross / opt.validation
    else:
        print("警告: 所有轮次的指标均无效，返回全0结果")
        metrics_cross_avg = np.zeros((1, 7))

    print('metrics_avg:', metrics_cross_avg)


if __name__ == '__main__':
    args = parameter_parser()
    main(args)





# import torch
# from prepareData import prepare_data
# import numpy as np
# from torch import optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from param import parameter_parser
# from Module import HGCLAMIR
# from utils import get_L2reg, Myloss
# from Calculate_Metrics import Metric_fun
# from trainData import Dataset
# import ConstructHW
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import silhouette_score
# from sklearn.impute import SimpleImputer
# import warnings
# import copy
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve
#
# warnings.filterwarnings('ignore')
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# USE_GRID_SEARCH = True
# EARLY_STOPPING_PATIENCE = 10  # 早停机制：如果验证集性能在10个epoch内没有提升，则停止训练
#
#
# def find_best_dbscan_params(data, eps_min=0.1, eps_max=2.0, eps_steps=20, min_samples_min=3, min_samples_max=20):
#     best_eps, best_min_samples = None, None
#     best_score = -1
#     best_labels = None
#
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data)
#
#     eps_values = np.linspace(eps_min, eps_max, eps_steps)
#     min_samples_values = range(min_samples_min, min_samples_max + 1)
#
#     for eps in eps_values:
#         for min_samples in min_samples_values:
#             try:
#                 clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data_scaled)
#                 labels = clustering.labels_
#
#                 n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#
#                 if n_clusters <= 1:
#                     continue
#
#                 noise_ratio = list(labels).count(-1) / len(labels)
#
#                 if noise_ratio > 0.5:
#                     continue
#
#                 score = silhouette_score(data_scaled, labels)
#
#                 if score > best_score:
#                     best_score = score
#                     best_eps, best_min_samples = eps, min_samples
#                     best_labels = labels
#             except Exception as e:
#                 print(f"DBSCAN参数 {eps}, {min_samples} 出错: {e}")
#                 continue
#
#     if best_eps is None:
#         print("警告: 未能找到合适的DBSCAN参数，使用启发式方法...")
#         best_eps = find_optimal_eps(data_scaled)
#         best_min_samples = 10
#
#     print(f"最佳DBSCAN参数: eps={best_eps}, min_samples={best_min_samples}, 轮廓系数={best_score:.4f}")
#     return best_eps, best_min_samples
#
#
# def find_optimal_eps(data, min_samples=10):
#     neighbors = NearestNeighbors(n_neighbors=min_samples)
#     neighbors.fit(data)
#     distances, _ = neighbors.kneighbors(data)
#     distances = np.sort(distances[:, min_samples - 1], axis=0)
#
#     diff = np.diff(distances)
#     max_diff_idx = np.argmax(diff)
#     eps = distances[max_diff_idx]
#
#     print(f"估计的最佳eps: {eps:.4f}")
#     return eps
#
#
# def standardize_data(data):
#     scaler = StandardScaler()
#     return scaler.fit_transform(data)
#
#
# def train_epoch(model, train_data, optimizer, opt, scheduler=None, val_data=None):
#     model.train()
#     regression_crit = Myloss()
#
#     one_index = train_data[2][0].to(device).t().tolist()
#     zero_index = train_data[2][1].to(device).t().tolist()
#
#     dis_sim_integrate_tensor = train_data[0].to(device)
#     mi_sim_integrate_tensor = train_data[1].to(device)
#
#     concat_miRNA = np.hstack([train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])
#     concat_miRNA_std = standardize_data(concat_miRNA)
#     concat_mi_tensor = torch.FloatTensor(concat_miRNA_std).to(device)
#
#     if USE_GRID_SEARCH:
#         best_eps_mi, best_min_samples_mi = find_best_dbscan_params(
#             concat_mi_tensor.cpu().numpy()
#         )
#     else:
#         best_eps_mi = find_optimal_eps(concat_mi_tensor.cpu().numpy())
#         best_min_samples_mi = 10
#
#     G_mi_Kn = ConstructHW.constructHW_knn(concat_mi_tensor.cpu().numpy(), K_neigs=[13], is_probH=False)
#     G_mi_Km = ConstructHW.constructHW_dbscan(
#         concat_mi_tensor.cpu().numpy(), eps=best_eps_mi, min_samples=best_min_samples_mi, is_probH=False
#     )
#     G_mi_Kn, G_mi_Km = G_mi_Kn.to(device), G_mi_Km.to(device)
#
#     concat_dis = np.hstack([train_data[4].numpy().T, dis_sim_integrate_tensor.detach().cpu().numpy()])
#     concat_dis_std = standardize_data(concat_dis)
#     concat_dis_tensor = torch.FloatTensor(concat_dis_std).to(device)
#
#     if USE_GRID_SEARCH:
#         best_eps_dis, best_min_samples_dis = find_best_dbscan_params(
#             concat_dis_tensor.cpu().numpy()
#         )
#     else:
#         best_eps_dis = find_optimal_eps(concat_dis_tensor.cpu().numpy())
#         best_min_samples_dis = 10
#
#     G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.cpu().numpy(), K_neigs=[13], is_probH=False)
#     G_dis_Km = ConstructHW.constructHW_dbscan(
#         concat_dis_tensor.cpu().numpy(), eps=best_eps_dis, min_samples=best_min_samples_dis, is_probH=False
#     )
#     G_dis_Kn, G_dis_Km = G_dis_Kn.to(device), G_dis_Km.to(device)
#
#     best_loss = float('inf')
#     best_model = None
#     patience = EARLY_STOPPING_PATIENCE
#
#     for epoch in range(1, opt.epoch + 1):
#         model.train()
#
#         score, mi_cl_loss, dis_cl_loss = model(concat_mi_tensor, concat_dis_tensor,
#                                                G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
#
#         recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)
#         reg_loss = get_L2reg(model.parameters())
#
#         if torch.isnan(mi_cl_loss) or torch.isinf(mi_cl_loss):
#             mi_cl_loss = torch.tensor(0.0, device=device)
#
#         if torch.isnan(dis_cl_loss) or torch.isinf(dis_cl_loss):
#             dis_cl_loss = torch.tensor(0.0, device=device)
#
#         tol_loss = recover_loss + 0.1 * (mi_cl_loss + dis_cl_loss) + 0.00001 * reg_loss
#
#         optimizer.zero_grad()
#         tol_loss.backward()
#
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#
#         optimizer.step()
#
#         if scheduler:
#             if isinstance(scheduler, ReduceLROnPlateau):
#                 scheduler.step(tol_loss)
#             else:
#                 scheduler.step()
#
#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}/{opt.epoch}, 恢复损失: {recover_loss.item():.6f}, "
#                   f"对比学习损失: {mi_cl_loss.item():.6f} + {dis_cl_loss.item():.6f}, "
#                   f"正则化损失: {reg_loss.item():.6f}, 总损失: {tol_loss.item():.6f}")
#
#         if val_data is not None:
#             model.eval()
#             with torch.no_grad():
#                 val_score, _, _ = model(concat_mi_tensor, concat_dis_tensor,
#                                         G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
#                 val_loss = regression_crit(one_index, zero_index, train_data[4].to(device), val_score)
#
#             if val_loss < best_loss:
#                 best_loss = val_loss
#                 best_model = copy.deepcopy(model)
#                 patience = EARLY_STOPPING_PATIENCE
#             else:
#                 patience -= 1
#                 if patience <= 0:
#                     print(f"早停触发: 在第 {epoch} 轮停止训练")
#                     model = best_model
#                     break
#
#     model.eval()
#     with torch.no_grad():
#         score, _, _ = model(concat_mi_tensor, concat_dis_tensor,
#                             G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
#
#     true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(
#         model, train_data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km
#     )
#
#     return true_value_one, true_value_zero, pre_value_one, pre_value_zero, score.cpu().numpy()
#
#
# def test(model, data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km):
#     model.eval()
#     with torch.no_grad():
#         score, _, _ = model(concat_mi_tensor, concat_dis_tensor,
#                             G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
#
#     test_one_index = data[3][0].t().tolist()
#     test_zero_index = data[3][1].t().tolist()
#     true_one = data[5][test_one_index]
#     true_zero = data[5][test_zero_index]
#
#     pre_one = score[test_one_index]
#     pre_zero = score[test_zero_index]
#
#     if torch.isnan(pre_one).any() or torch.isinf(pre_one).any():
#         print("警告: pre_one包含NaN或inf，将其替换为0")
#         pre_one = torch.nan_to_num(pre_one, nan=0.0, posinf=0.0, neginf=0.0)
#
#     if torch.isnan(pre_zero).any() or torch.isinf(pre_zero).any():
#         print("警告: pre_zero包含NaN或inf，将其替换为0")
#         pre_zero = torch.nan_to_num(pre_zero, nan=0.0, posinf=0.0, neginf=0.0)
#
#     return true_one, true_zero, pre_one, pre_zero
#
#
# def evaluate(true_one, true_zero, pre_one, pre_zero):
#     Metric = Metric_fun()
#     metrics_tensor = np.zeros((1, 7))
#     auc_scores = []
#     aupr_scores = []
#
#     valid_runs = 0
#
#     for seed in range(10):
#         test_po_num = true_one.shape[0]
#
#         zero_indices = np.where(true_zero == 0)[0]
#         if len(zero_indices) < test_po_num:
#             print(f"警告: 种子 {seed}: true_zero中没有足够的0值，跳过此次评估")
#             continue
#
#         np.random.seed(seed)
#         np.random.shuffle(zero_indices)
#         test_ne_index = zero_indices[:test_po_num]
#
#         eval_true_zero = true_zero[test_ne_index]
#         eval_true_data = torch.cat([true_one, eval_true_zero])
#
#         eval_pre_zero = pre_zero[test_ne_index]
#         eval_pre_data = torch.cat([pre_one, eval_pre_zero])
#
#         if torch.isnan(eval_true_data).any() or torch.isinf(eval_true_data).any():
#             print(f"警告: 种子 {seed}: eval_true_data包含NaN或inf，跳过此次评估")
#             continue
#
#         if torch.isnan(eval_pre_data).any() or torch.isinf(eval_pre_data).any():
#             print(f"警告: 种子 {seed}: eval_pre_data包含NaN或inf，跳过此次评估")
#             continue
#
#         metrics = Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data)
#
#         try:
#             y_true = eval_true_data.cpu().numpy()
#             y_score = eval_pre_data.cpu().numpy()
#             auc_score = roc_auc_score(y_true, y_score)
#             precision, recall, _ = precision_recall_curve(y_true, y_score)
#             aupr_score = auc(recall, precision)
#
#             auc_scores.append(auc_score)
#             aupr_scores.append(aupr_score)
#
#             print(f"种子 {seed}: AUC={auc_score:.4f}, AUPR={aupr_score:.4f}")
#         except Exception as e:
#             print(f"计算AUC/AUPR出错: {e}")
#             auc_scores.append(0.0)
#             aupr_scores.append(0.0)
#
#         if np.isnan(metrics).any() or np.isinf(metrics).any():
#             print(f"警告: 种子 {seed}: 计算的指标包含NaN或inf，跳过此次评估")
#             continue
#
#         metrics_tensor += metrics
#         valid_runs += 1
#
#     if valid_runs > 0:
#         metrics_tensor_avg = metrics_tensor / valid_runs
#         avg_auc = np.mean(auc_scores)
#         avg_aupr = np.mean(aupr_scores)
#         print(f"平均AUC: {avg_auc:.4f}, 平均AUPR: {avg_aupr:.4f}")
#     else:
#         print("警告: 没有有效的评估运行，返回全0指标")
#         metrics_tensor_avg = np.zeros((1, 7))
#         avg_auc = 0.0
#         avg_aupr = 0.0
#
#     if np.isnan(metrics_tensor_avg).any():
#         print("警告: 最终指标包含NaN，将其替换为0")
#         metrics_tensor_avg = np.nan_to_num(metrics_tensor_avg)
#
#     return metrics_tensor_avg
#
#
# def train_random_forest(pre_one, pre_zero, labels):
#     pre_one_np = pre_one.detach().cpu().numpy()
#     pre_zero_np = pre_zero.detach().cpu().numpy()
#
#     print(f"pre_one shape: {pre_one_np.shape}")
#     print(f"pre_zero shape: {pre_zero_np.shape}")
#
#     if pre_one_np.size == 0 or pre_zero_np.size == 0:
#         print("警告: pre_one或pre_zero为空数组，跳过随机森林训练")
#         return None
#
#     pre_one_reshaped = pre_one_np.reshape(-1, 1)
#     pre_zero_reshaped = pre_zero_np.reshape(-1, 1)
#
#     X = np.concatenate([pre_one_reshaped, pre_zero_reshaped], axis=0)
#
#     print(f"合并后的X形状: {X.shape}")
#
#     if np.isnan(X).any():
#         print("警告: 数据包含NaN值，使用中位数填充")
#         imputer = SimpleImputer(strategy='median')
#         X = imputer.fit_transform(X)
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     if X_scaled.shape[1] == 0:
#         print("警告: 特征维度为0，跳过随机森林训练")
#         return None
#
#     clf = RandomForestClassifier(
#         n_estimators=500,
#         max_depth=10,
#         min_samples_split=5,
#         min_samples_leaf=2,
#         random_state=42,
#         n_jobs=-1
#     )
#
#     clf.fit(X_scaled, labels)
#     return clf
#
#
# def evaluate_random_forest(rf_model, pre_one, pre_zero, labels):
#     if rf_model is None:
#         return 0.0
#
#     pre_one_np = pre_one.detach().cpu().numpy()
#     pre_zero_np = pre_zero.detach().cpu().numpy()
#
#     if pre_one_np.size == 0 or pre_zero_np.size == 0:
#         print("警告: pre_one或pre_zero为空数组，返回0准确率")
#         return 0.0
#
#     pre_one_reshaped = pre_one_np.reshape(-1, 1)
#     pre_zero_reshaped = pre_zero_np.reshape(-1, 1)
#
#     X = np.concatenate([pre_one_reshaped, pre_zero_reshaped], axis=0)
#
#     if np.isnan(X).any():
#         print("警告: 数据包含NaN值，使用中位数填充")
#         imputer = SimpleImputer(strategy='median')
#         X = imputer.fit_transform(X)
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     if hasattr(rf_model, "predict_proba"):
#         probas = rf_model.predict_proba(X_scaled)[:, 1]
#         auc_score = roc_auc_score(labels, probas)
#         print(f"随机森林AUC: {auc_score:.4f}")
#         return auc_score
#     else:
#         preds = rf_model.predict(X_scaled)
#         return accuracy_score(labels, preds)
#
#
# def main(opt):
#     if not os.path.exists('saved_models'):
#         os.makedirs('saved_models')
#     if not os.path.exists('results'):
#         os.makedirs('results')
#
#     dataset = prepare_data(opt)
#     train_data = Dataset(opt, dataset)
#     metrics_cross = np.zeros((1, 7))
#
#     # 用于存储ROC曲线数据
#     plt.figure(figsize=(10, 8))
#     all_fpr = []
#     all_tpr = []
#     all_auc = []
#     colors = ['blue', 'green', 'red', 'cyan', 'magenta']
#     base_fpr = np.linspace(0, 1, 1000)  # 从100增加到1000个点
#
#     for i in range(opt.validation):
#         print(f"\n===== 验证轮次 {i + 1}/{opt.validation} =====")
#
#         hidden_list = [256, 256]
#         num_proj_hidden = 64
#
#         model = HGCLAMIR(args.mi_num, args.dis_num, hidden_list, num_proj_hidden, args)
#         model.to(device)
#
#         optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
#         scheduler = ReduceLROnPlateau(
#             optimizer,
#             mode='min',
#             factor=0.5,
#             patience=5,
#             verbose=True,
#             min_lr=1e-7
#         )
#
#         print("开始训练...")
#         true_score_one, true_score_zero, pre_score_one, pre_score_zero, reconstructed_matrix = train_epoch(
#             model, train_data[i], optimizer, opt, scheduler
#         )
#
#         print("训练完成，开始评估...")
#
#         # 绘制ROC曲线
#         y_true = torch.cat([true_score_one, true_score_zero]).cpu().numpy()
#         y_score = torch.cat([pre_score_one, pre_score_zero]).cpu().numpy()
#
#         fpr, tpr, _ = roc_curve(y_true, y_score)
#         roc_auc = auc(fpr, tpr)
#
#         # 使用样条插值使曲线更平滑
#         from scipy import interpolate
#         tck = interpolate.splrep(fpr, tpr, s=0.5)  # s是平滑因子，可以调整
#         tpr_smooth = interpolate.splev(base_fpr, tck, der=0)
#         tpr_smooth[0] = 0.0  # 确保从(0,0)开始
#         tpr_smooth[-1] = 1.0  # 确保结束于(1,1)
#
#         all_fpr.append(fpr)
#         all_tpr.append(tpr)
#         all_auc.append(roc_auc)
#
#         plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
#                  label='Fold %d (AUC = %0.3f)' % (i + 1, roc_auc))
#
#         # 训练随机森林模型
#         train_labels = np.hstack([
#             np.ones(true_score_one.shape[0]),
#             np.zeros(true_score_zero.shape[0])
#         ])
#         rf_model = train_random_forest(pre_score_one, pre_score_zero, train_labels)
#
#         if rf_model is not None:
#             rf_auc = evaluate_random_forest(rf_model, pre_score_one, pre_score_zero, train_labels)
#             print(f"Random Forest AUC: {rf_auc * 100:.3f}%")
#
#         metrics_value = evaluate(true_score_one, true_score_zero, pre_score_one, pre_score_zero)
#         print(f"此轮指标: {metrics_value}")
#
#         if np.isnan(metrics_value).all():
#             print("警告: 此轮指标全为NaN，跳过累加")
#             continue
#
#         metrics_cross += metrics_value
#
#         matrix_path = f'results/reconstructed_matrix_fold_{i + 1}.csv'
#         matrix_df = pd.DataFrame(reconstructed_matrix)
#         matrix_df.to_csv(matrix_path, index=False, header=False)
#         print(f"重构关联矩阵已保存至 {matrix_path}")
#
#         model_path = f'saved_models/model_fold_{i + 1}.pth'
#         torch.save(model.state_dict(), model_path)
#         print(f"模型已保存至 {model_path}")
#
#     # 绘制平均ROC曲线
#     mean_fpr = np.linspace(0, 1, 100)
#     mean_tpr = np.zeros_like(mean_fpr)
#     for i in range(opt.validation):
#         mean_tpr += np.interp(mean_fpr, all_fpr[i], all_tpr[i])
#     mean_tpr /= opt.validation
#     mean_auc = auc(mean_fpr, mean_tpr)
#
#     # 对平均曲线进行二次平滑
#     from scipy.ndimage import gaussian_filter1d
#     mean_tpr_smooth = gaussian_filter1d(mean_tpr, sigma=5)  # sigma控制平滑程度
#
#     plt.plot(mean_fpr, mean_tpr, color='black', linestyle='--',
#              label='Mean ROC (AUC = %0.3f)' % mean_auc, lw=3, alpha=.9)
#
#     plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate',fontsize=12)
#     plt.ylabel('True Positive Rate',fontsize=12)
#     plt.title('Smoothed ROC Curves', fontsize=14)
#     plt.legend(loc="lower right",fontsize=10)
#
#     # 设置网格线使图形更清晰
#     plt.grid(True, linestyle='--', alpha=0.5)
#
#     roc_curve_path = f'results/roc_curves_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png'
#     plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"ROC曲线图已保存至 {roc_curve_path}")
#
#     if np.sum(metrics_cross) != 0:
#         metrics_cross_avg = metrics_cross / opt.validation
#     else:
#         print("警告: 所有轮次的指标均无效，返回全0结果")
#         metrics_cross_avg = np.zeros((1, 7))
#
#     print('metrics_avg:', metrics_cross_avg)
#
#     metrics_path = f'results/metrics_results_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
#     metrics_df = pd.DataFrame(metrics_cross_avg, columns=[
#         'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'MCC', 'AUC'
#     ])
#     metrics_df.to_csv(metrics_path, index=False)
#     print(f"评估指标已保存至 {metrics_path}")
#
#
# if __name__ == '__main__':
#     args = parameter_parser()
#     main(args)







# # 这段代码实现了一个用于训练和评估图卷积网络（GCN）模型的框架，
# # 主要用于处理某种基因数据（如miRNA和dis）并进行分类任务。
# # 它包含数据准备、训练、测试、评估等多个环节。
#
#
# import torch
# from prepareData import prepare_data
# import numpy as np
# from torch import optim
# from param import parameter_parser
# from Module import HGCLAMIR
# from utils import get_L2reg, Myloss
# from Calculate_Metrics import Metric_fun
# from trainData import Dataset
# import ConstructHW
#
# import warnings
# warnings.filterwarnings('ignore')
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# import torch
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import silhouette_score
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 是否使用网格搜索
# USE_GRID_SEARCH = True
#
# def find_best_dbscan_params(data, eps_values=[0.3, 0.5, 0.7], min_samples_values=[5, 10, 15]):
#     """ 通过 Silhouette Score 选择最佳 DBSCAN 参数 """
#     best_eps, best_min_samples = None, None
#     best_score = -1
#
#     for eps in eps_values:
#         for min_samples in min_samples_values:
#             clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
#             labels = clustering.labels_
#
#             # 至少需要两个簇才能计算 Silhouette Score
#             if len(set(labels)) > 1:
#                 score = silhouette_score(data, labels)
#                 if score > best_score:
#                     best_score = score
#                     best_eps, best_min_samples = eps, min_samples
#
#     return best_eps, best_min_samples if best_eps is not None else (0.5, 10)  # 备用默认值
#
# def find_optimal_eps(data, min_samples=10, quantile=0.95):
#     """ 通过 K-Distance Graph 选择 `eps` """
#     neighbors = NearestNeighbors(n_neighbors=min_samples)
#     neighbors.fit(data)
#     distances, _ = neighbors.kneighbors(data)
#     distances = np.sort(distances[:, min_samples - 1], axis=0)
#     return distances[int(len(distances) * quantile)]  # 取 `quantile` 分位点作为 `eps`
#
# def train_epoch(model, train_data, optim, opt):
#     model.train()
#     regression_crit = Myloss()
#
#     one_index = train_data[2][0].to(device).t().tolist()
#     zero_index = train_data[2][1].to(device).t().tolist()
#
#     dis_sim_integrate_tensor = train_data[0].to(device)
#     mi_sim_integrate_tensor = train_data[1].to(device)
#
#     concat_miRNA = np.hstack([train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])
#     concat_mi_tensor = torch.FloatTensor(concat_miRNA).to(device)
#
#     # **自动搜索 DBSCAN 参数**
#     if USE_GRID_SEARCH:
#         best_eps_mi, best_min_samples_mi = find_best_dbscan_params(
#             concat_mi_tensor.cpu().numpy(),
#             eps_values=np.linspace(0.1, 1.0, 10),
#             min_samples_values=[5, 10, 15]
#         )
#     else:
#         best_eps_mi = find_optimal_eps(concat_mi_tensor.cpu().numpy(), min_samples=10, quantile=0.95)
#         best_min_samples_mi = 10
#
#     G_mi_Kn = ConstructHW.constructHW_knn(concat_mi_tensor.cpu().numpy(), K_neigs=[13], is_probH=False)
#     G_mi_Km = ConstructHW.constructHW_dbscan(
#         concat_mi_tensor.cpu().numpy(), eps=best_eps_mi, min_samples=best_min_samples_mi, is_probH=False
#     )
#
#     G_mi_Kn, G_mi_Km = G_mi_Kn.to(device), G_mi_Km.to(device)
#
#     concat_dis = np.hstack([train_data[4].numpy().T, dis_sim_integrate_tensor.detach().cpu().numpy()])
#     concat_dis_tensor = torch.FloatTensor(concat_dis).to(device)
#
#     # **自动搜索 DBSCAN 参数**
#     if USE_GRID_SEARCH:
#         best_eps_dis, best_min_samples_dis = find_best_dbscan_params(
#             concat_dis_tensor.cpu().numpy(),
#             eps_values=np.linspace(0.1, 1.0, 10),
#             min_samples_values=[5, 10, 15]
#         )
#     else:
#         best_eps_dis = find_optimal_eps(concat_dis_tensor.cpu().numpy(), min_samples=10, quantile=0.95)
#         best_min_samples_dis = 10
#
#     G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.cpu().numpy(), K_neigs=[13], is_probH=False)
#     G_dis_Km = ConstructHW.constructHW_dbscan(
#         concat_dis_tensor.cpu().numpy(), eps=best_eps_dis, min_samples=best_min_samples_dis, is_probH=False
#     )
#
#     G_dis_Kn, G_dis_Km = G_dis_Kn.to(device), G_dis_Km.to(device)
#
#     for epoch in range(1, opt.epoch + 1):
#         score, mi_cl_loss, dis_cl_loss = model(concat_mi_tensor, concat_dis_tensor,
#                                                G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
#
#         recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)
#         reg_loss = get_L2reg(model.parameters())
#
#         tol_loss = recover_loss + mi_cl_loss + dis_cl_loss + 0.00001 * reg_loss
#         optim.zero_grad()
#         tol_loss.backward()
#         optim.step()
#
#     true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(
#         model, train_data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km
#     )
#
#     return true_value_one, true_value_zero, pre_value_one, pre_value_zero
#
# # # 模型会根据输入数据计算损失并进行反向传播，从而更新网络权重。
# # def train_epoch(model, train_data, optim, opt):
# #
# #     model.train()
# #     regression_crit = Myloss()
# #
# #     one_index = train_data[2][0].to(device).t().tolist()
# #     zero_index = train_data[2][1].to(device).t().tolist()
# #
# #     dis_sim_integrate_tensor = train_data[0].to(device)
# #     mi_sim_integrate_tensor = train_data[1].to(device)
# #
# #
# #     concat_miRNA = np.hstack(
# #         [train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])
# #     concat_mi_tensor = torch.FloatTensor(concat_miRNA)
# #     concat_mi_tensor = concat_mi_tensor.to(device)
# #
# #
# #
# #     # 构建超图
# #     G_mi_Kn = ConstructHW.constructHW_knn(concat_mi_tensor.detach().cpu().numpy(), K_neigs=[13], is_probH=False)
# #     G_mi_Km = ConstructHW.constructHW_dbscan(
# #         concat_mi_tensor.detach().cpu().numpy(),
# #         eps=0.5,  # 这里应该是一个浮动的数值
# #         min_samples=10,  # 设置 min_samples 值（可以根据需要调整）
# #         is_probH=False  # 是否使用概率矩阵（根据需要设置）
# #     )
# #
# #     # G_mi_Km = ConstructHW.constructHW_kmean(concat_mi_tensor.detach().cpu().numpy(), clusters=[9])
# #
# #     G_mi_Kn = G_mi_Kn.to(device)
# #     G_mi_Km = G_mi_Km.to(device)
# #
# #
# #     concat_dis = np.hstack(
# #         [train_data[4].numpy().T, dis_sim_integrate_tensor.detach().cpu().numpy()])
# #     concat_dis_tensor = torch.FloatTensor(concat_dis)
# #     concat_dis_tensor = concat_dis_tensor.to(device)
# #
# #
# #     G_dis_Kn = ConstructHW.constructHW_knn(concat_dis_tensor.detach().cpu().numpy(), K_neigs=[13], is_probH=False)
# #     G_dis_Km = ConstructHW.constructHW_dbscan(
# #         concat_dis_tensor.detach().cpu().numpy(),
# #         eps=0.5,  # 设置适当的 eps 值（可以根据需要调整）
# #         min_samples=10,  # 设置 min_samples 值（可以根据需要调整）
# #         is_probH=False  # 是否使用概率矩阵（根据需要设置）
# #     )
# #
# #     #G_dis_Km = ConstructHW.constructHW_kmean(concat_dis_tensor.detach().cpu().numpy(), clusters=[9])
# #
# #     G_dis_Kn = G_dis_Kn.to(device)
# #     G_dis_Km = G_dis_Km.to(device)
# #
# #     for epoch in range(1, opt.epoch+1):
# #
# #         score, mi_cl_loss, dis_cl_loss = model(concat_mi_tensor, concat_dis_tensor,
# #                                                G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
# #
# #         recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)
# #         reg_loss = get_L2reg(model.parameters())
# #
# #         tol_loss = recover_loss + mi_cl_loss + dis_cl_loss + 0.00001 * reg_loss
# #         optim.zero_grad()
# #         tol_loss.backward()
# #         optim.step()
# #
# #     true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(model, train_data, concat_mi_tensor, concat_dis_tensor,
# #                                  G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
# #
# #     return true_value_one, true_value_zero, pre_value_one, pre_value_zero
#
#
# # 测试
# def test(model, data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km):
#     model.eval()
#     score,_,_ = model(concat_mi_tensor, concat_dis_tensor,
#                       G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
#     test_one_index = data[3][0].t().tolist()
#     test_zero_index = data[3][1].t().tolist()
#     true_one = data[5][test_one_index]
#     true_zero = data[5][test_zero_index]
#
#     pre_one = score[test_one_index]
#     pre_zero = score[test_zero_index]
#
#     return true_one, true_zero, pre_one, pre_zero
#
# # 评估
# def evaluate(true_one, true_zero, pre_one, pre_zero):
#
#     Metric = Metric_fun()
#     metrics_tensor = np.zeros((1, 7))
#
#     for seed in range(10):
#         test_po_num = true_one.shape[0]
#         test_index = np.array(np.where(true_zero == 0))
#         np.random.seed(seed)
#         np.random.shuffle(test_index.T)
#         test_ne_index = tuple(test_index[:, :test_po_num])
#
#         eval_true_zero = true_zero[test_ne_index]
#         eval_true_data = torch.cat([true_one,eval_true_zero])
#
#         eval_pre_zero = pre_zero[test_ne_index]
#         eval_pre_data = torch.cat([pre_one,eval_pre_zero])
#
#         metrics_tensor = metrics_tensor + Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data)
#
#     metrics_tensor_avg = metrics_tensor / 10
#
#     return metrics_tensor_avg
#
#
# def main(opt):
#     dataset = prepare_data(opt)
#     train_data = Dataset(opt, dataset)
#
#     metrics_cross = np.zeros((1, 7))
#
#     for i in range(opt.validation):
#
#         hidden_list = [256, 256]
#         num_proj_hidden = 64
#
#
#         model = HGCLAMIR(args.mi_num, args.dis_num, hidden_list, num_proj_hidden, args)
#         model.to(device)
#         optimizer = optim.Adam(model.parameters(), lr = 0.0001)
#         true_score_one, true_score_zero, pre_score_one, pre_score_zero = train_epoch(model, train_data[i], optimizer,
#                                                                                      opt)
#         metrics_value = evaluate(true_score_one, true_score_zero, pre_score_one, pre_score_zero)
#         metrics_cross = metrics_cross + metrics_value
#
#     metrics_cross_avg = metrics_cross / opt.validation
#     print('metrics_avg:',metrics_cross_avg)
#
#
# if __name__ == '__main__':
#
#     args = parameter_parser()
#     main(args)