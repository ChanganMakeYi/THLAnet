import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, \
    recall_score, matthews_corrcoef, confusion_matrix
import numpy as np
from tqdm import tqdm
import argparse
import logging
import os
import random
import MLPtrain as mlp
import process_encoder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 清空显存
torch.cuda.empty_cache()


class RatioSampler:
    """按指定比例抽取负样本和正样本的采样器，确保每批次至少一个正类。"""

    def __init__(self, dataset, neg_pos_ratio, batch_size):
        self.dataset = dataset
        self.neg_pos_ratio = neg_pos_ratio
        self.batch_size = batch_size
        self.labels = [int(dataset[i][1].item()) for i in range(len(dataset))]
        self.neg_indices = [i for i, label in enumerate(self.labels) if label == 0]
        self.pos_indices = [i for i, label in enumerate(self.labels) if label == 1]
        self.num_neg = len(self.neg_indices)
        self.num_pos = len(self.pos_indices)

        if self.num_neg == 0 or self.num_pos == 0:
            raise ValueError("数据集必须包含负样本和正样本")

        self.neg_per_batch = int(self.batch_size * self.neg_pos_ratio / (self.neg_pos_ratio + 1))
        self.pos_per_batch = max(1, self.batch_size - self.neg_per_batch)
        if self.neg_per_batch < 1:
            raise ValueError(f"批次大小 {self.batch_size} 不足以按比例 {self.neg_pos_ratio}:1 分配样本")

    def __iter__(self):
        neg_indices = self.neg_indices.copy()
        pos_indices = self.pos_indices.copy()
        random.shuffle(neg_indices)
        random.shuffle(pos_indices)

        batches = []
        neg_idx = pos_idx = 0
        while neg_idx < len(neg_indices) and pos_idx < len(pos_indices):
            batch = []
            neg_count = min(self.neg_per_batch, len(neg_indices) - neg_idx)
            batch.extend(neg_indices[neg_idx:neg_idx + neg_count])
            neg_idx += neg_count
            pos_count = min(self.pos_per_batch, len(pos_indices) - pos_idx)
            batch.extend(pos_indices[pos_idx:pos_idx + pos_count])
            pos_idx += pos_count
            random.shuffle(batch)
            batches.extend(batch)

        remaining = neg_indices[neg_idx:] + pos_indices[pos_idx:]
        random.shuffle(remaining)
        batches.extend(remaining)
        return iter(batches[:len(self.dataset)])

    def __len__(self):
        return len(self.dataset)


def load_data(args):
    """加载并预处理测试集的标签和 ESM 数据。"""
    try:
        labels = np.array(process_encoder.get_label(args.file_path_test))
        logger.info(f"原始标签唯一值: {np.unique(labels)}")
        labels = np.where(labels > 0, 1, 0).astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        logger.info(f"处理后标签唯一值: {torch.unique(labels)}")
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("标签必须为 0 或 1")

        num_neg = (labels == 0).sum().item()
        num_pos = (labels == 1).sum().item()
        logger.info(f"负样本数: {num_neg}, 正样本数: {num_pos}")

        esm_data = torch.load(args.pt_data, map_location='cpu')
        logger.info(f"ESM 数据形状: {esm_data.shape}")
        if esm_data.shape[1:] != (80, 1280):
            raise ValueError(f"预期 ESM 数据形状为 [n, 80, 21]，实际为 {esm_data.shape}")
        esm_data = esm_data.to(dtype=torch.float32)

        # 标准化输入
        esm_data = (esm_data - esm_data.mean(dim=0)) / (esm_data.std(dim=0) + 1e-8)

        # 检查样本数是否匹配
        if esm_data.shape[0] != labels.shape[0]:
            min_samples = min(esm_data.shape[0], labels.shape[0])
            logger.warning(
                f"样本数不匹配: esm_data 有 {esm_data.shape[0]} 个样本，labels 有 {labels.shape[0]} 个样本。截断到 {min_samples} 个样本。")
            esm_data = esm_data[:min_samples]
            labels = labels[:min_samples]

        return esm_data, labels, num_pos, num_neg
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise


def create_dataloader(esm_data, labels, batch_size, neg_pos_ratio):
    """创建测试集数据加载器。"""
    dataset = TensorDataset(esm_data, labels)
    sampler = RatioSampler(dataset, neg_pos_ratio, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    logger.info(f"测试集大小: {len(dataset)}")
    return dataloader


def evaluate(model, dataloader, device, classification_threshold=0.05):
    """在测试集上评估模型。"""
    model.eval()
    labels_all = []
    outputs_all = []
    predicted_all = []
    confusion_matrices = []
    skipped_batches = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="测试批次"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            outputs = model(inputs)  # 输出 logits
            outputs_prob = torch.sigmoid(outputs)  # 转换为概率
            predicted = (outputs_prob > classification_threshold).float()

            # 检查标签和预测的类别数
            unique_labels = torch.unique(labels).cpu().numpy()
            unique_predicted = torch.unique(predicted).cpu().numpy()
            logger.info(f"批次 {count + 1} - 真实标签唯一值: {unique_labels}, 预测标签唯一值: {unique_predicted}")

            # 强制生成 2x2 混淆矩阵
            cm = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1])
            if cm.shape != (2, 2):
                logger.warning(f"批次 {count + 1} 混淆矩阵形状异常: {cm.shape}")
                cm = np.zeros((2, 2))  # 填充为 2x2
                skipped_batches += 1
            confusion_matrices.append(cm)

            labels_all.extend(labels.cpu().numpy().flatten())
            outputs_all.extend(outputs_prob.cpu().numpy().flatten())
            predicted_all.extend(predicted.cpu().numpy().flatten())
            count += 1

            del inputs, labels, outputs, outputs_prob, predicted
            torch.cuda.empty_cache()

    if count == 0:
        logger.error("无有效批次可评估")
        return {"auc": 0.0, "aupr": 0.0, "accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "mcc": 0.0}, np.zeros((2, 2))

    logger.info(f"跳过批次数量: {skipped_batches}/{count}")

    labels_all = np.array(labels_all)
    outputs_all = np.array(outputs_all)
    predicted_all = np.array(predicted_all)

    # 计算指标
    metrics = {
        "auc": 0.0,
        "aupr": 0.0,
        "accuracy": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "mcc": 0.0
    }
    try:
        if len(np.unique(labels_all)) > 1:  # 确保至少有两类
            metrics["auc"] = roc_auc_score(labels_all, outputs_all)
            metrics["aupr"] = average_precision_score(labels_all, outputs_all)
        metrics["accuracy"] = accuracy_score(labels_all, predicted_all)
        metrics["f1"] = f1_score(labels_all, predicted_all, zero_division=0)
        metrics["precision"] = precision_score(labels_all, predicted_all, zero_division=0)
        metrics["recall"] = recall_score(labels_all, predicted_all, zero_division=0)
        metrics["mcc"] = matthews_corrcoef(labels_all, predicted_all)
    except ValueError as e:
        logger.warning(f"指标计算错误: {e}")

    # 累加混淆矩阵
    try:
        confusion = np.sum(confusion_matrices, axis=0)
        if confusion.shape != (2, 2):
            logger.error(f"累加混淆矩阵形状异常: {confusion.shape}")
            confusion = np.zeros((2, 2))
    except ValueError as e:
        logger.error(f"混淆矩阵累加失败: {e}")
        confusion = np.zeros((2, 2))

    # 记录输出概率统计
    logger.info(
        f"输出概率统计 - 均值: {outputs_all.mean():.4f}, 标准差: {outputs_all.std():.4f}, 最小值: {outputs_all.min():.4f}, 最大值: {outputs_all.max():.4f}")

    # 动态阈值优化
    best_f1, best_threshold = metrics["f1"], classification_threshold
    for threshold in [0.01, 0.05, 0.1, 0.15]:
        predicted_temp = (outputs_all > threshold).astype(float)
        f1_temp = f1_score(labels_all, predicted_temp, zero_division=0)
        if f1_temp > best_f1:
            best_f1, best_threshold = f1_temp, threshold
    logger.info(f"最佳阈值: {best_threshold:.2f}, F1: {best_f1:.4f}")

    return metrics, confusion


def save_results(metrics, confusion):
    """保存测试结果到文件。"""
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/test_output.txt", "w", encoding="utf-8") as file:
            file.write(f"测试集 - "
                       f"AUC: {metrics['auc']:.4f}, "
                       f"AUPR: {metrics['aupr']:.4f}, "
                       f"准确率: {metrics['accuracy']:.4f}, "
                       f"F1: {metrics['f1']:.4f}, "
                       f"精确率: {metrics['precision']:.4f}, "
                       f"召回率: {metrics['recall']:.4f}, "
                       f"MCC: {metrics['mcc']:.4f}\n")
            file.write(f"测试集混淆矩阵:\n{confusion}\n")
        logger.info("结果已保存至 data/test_output.txt")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")


def main():
    """主测试函数。"""
    parser = argparse.ArgumentParser(description="测试 MLP 模型用于二分类")
    parser.add_argument('--file_path_test', type=str, required=True, help='测试数据集路径')
    parser.add_argument('--pt_data', type=str, required=True, help='ESM 数据路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--neg_pos_ratio', type=float, default=1.0, help='负样本:正样本比例')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载模型
    try:
        model = mlp.MLPtrain(hidden_dims=[1024, 512, 256], dropout_rate=0.3).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

    # 加载数据
    esm_data, labels, num_pos, num_neg = load_data(args)
    test_dataloader = create_dataloader(esm_data, labels, args.batch_size, args.neg_pos_ratio)

    # 评估模型
    metrics, confusion = evaluate(model, test_dataloader, device, classification_threshold=0.05)
    logger.info(f"测试集 - "
                f"AUC: {metrics['auc']:.4f}, "
                f"AUPR: {metrics['aupr']:.4f}, "
                f"准确率: {metrics['accuracy']:.4f}, "
                f"F1: {metrics['f1']:.4f}, "
                f"精确率: {metrics['precision']:.4f}, "
                f"召回率: {metrics['recall']:.4f}, "
                f"MCC: {metrics['mcc']:.4f}")
    logger.info(f"测试集混淆矩阵:\n{confusion}")

    # 保存结果
    save_results(metrics, confusion)


if __name__ == "__main__":
    main()