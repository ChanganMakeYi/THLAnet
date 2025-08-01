import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, \
    recall_score, matthews_corrcoef, confusion_matrix
import numpy as np
from tqdm import tqdm
import argparse
import logging
import os
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import MLPtrain as mlp
import process_encoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.accum_steps = args.accum_steps
        self.neg_pos_ratio = args.neg_pos_ratio
        self.weight_decay = args.weight_decay
        self.patience = args.patience
        self.epochs = args.epoch
        self.file_path_train = args.file_path_train
        self.data_transfomer_esm = args.data_transfomer_esm
        self.model_esm_path = args.model_esm_path
        self.train_split = 0.9
        self.val_split = 0.05
        self.classification_threshold = 0.2

class DynamicHLASampler:
    def __init__(self, dataset, hla_list, neg_pos_ratio, batch_size):
        self.dataset = dataset
        self.neg_pos_ratio = neg_pos_ratio
        self.batch_size = batch_size
        self.labels = [int(dataset[i][1].item()) for i in range(len(dataset))]
        self.hla_list = hla_list
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

        self.hla_freq = self.compute_hla_frequencies()
        self.sample_weights = self.compute_sample_weights()

    def compute_hla_frequencies(self):
        from collections import Counter
        hla_counter = Counter(self.hla_list)
        total = len(self.hla_list)
        freq = {hla: count / total for hla, count in hla_counter.items()}
        return freq

    def compute_sample_weights(self):
        weights = []
        for hla in self.hla_list:
            freq = self.hla_freq.get(hla, 1e-6)
            weight = 1.0 / freq if freq > 0 else 1.0
            weights.append(weight)
        return np.array(weights)

    def __iter__(self):
        neg_indices = np.array(self.neg_indices)
        pos_indices = np.array(self.pos_indices)
        neg_weights = self.sample_weights[neg_indices]
        pos_weights = self.sample_weights[pos_indices]

        neg_probs = neg_weights / neg_weights.sum()
        pos_probs = pos_weights / pos_weights.sum()

        batches = []
        neg_sampled = 0
        pos_sampled = 0
        total_samples = len(self.dataset)

        while neg_sampled < len(neg_indices) or pos_sampled < len(pos_indices):
            batch = []

            if neg_sampled < len(neg_indices):
                neg_count = min(self.neg_per_batch, len(neg_indices) - neg_sampled)
                neg_batch = np.random.choice(neg_indices, size=neg_count, replace=False, p=neg_probs)
                batch.extend(neg_batch.tolist())
                neg_sampled += neg_count

            if pos_sampled < len(pos_indices):
                pos_count = min(self.pos_per_batch, len(pos_indices) - pos_sampled)
                pos_batch = np.random.choice(pos_indices, size=pos_count, replace=False, p=pos_probs)
                batch.extend(pos_batch.tolist())
                pos_sampled += pos_count

            random.shuffle(batch)
            batches.extend(batch)

            if len(batches) >= total_samples:
                break

        random.shuffle(batches)
        return iter(batches[:total_samples])

    def __len__(self):
        return len(self.dataset)

def load_data(config):
    try:
        labels = np.array(process_encoder.get_label(config.file_path_train))
        logger.info(f"原始标签唯一值: {np.unique(labels)}")
        labels = np.where(labels > 0, 1, 0).astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        logger.info(f"处理后标签唯一值: {torch.unique(labels)}")
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("标签必须为 0 或 1")

        num_neg = (labels == 0).sum().item()
        num_pos = (labels == 1).sum().item()
        logger.info(f"负样本数: {num_neg}, 正样本数: {num_pos}")

        esm_data = torch.load(config.data_transfomer_esm, map_location='cpu')
        logger.info(f"数据形状: {esm_data.shape}")
        if esm_data.shape[1:] != (80, 1000):
            raise ValueError(f"预期 ESM 数据形状为 [n, 80, 1000]，实际为 {esm_data.shape}")
        esm_data = esm_data.to(dtype=torch.float32)

        df = pd.read_csv(config.file_path_train)
        hla_list = df['HLA'].tolist()

        if esm_data.shape[0] != labels.shape[0] or esm_data.shape[0] != len(hla_list):
            min_samples = min(esm_data.shape[0], labels.shape[0], len(hla_list))
            logger.warning(
                f"样本数不匹配: esm_data 有 {esm_data.shape[0]} 个样本，labels 有 {labels.shape[0]} 个样本，HLA 有 {len(hla_list)} 个。截断到 {min_samples} 个样本。")
            esm_data = esm_data[:min_samples]
            labels = labels[:min_samples]
            hla_list = hla_list[:min_samples]

        return esm_data, labels, num_pos, num_neg, hla_list
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise

def create_dataloaders(esm_data, labels, hla_list, config):
    dataset = TensorDataset(esm_data, labels)
    total_size = len(dataset)
    train_size = int(config.train_split * total_size)
    val_size = int(config.val_split * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_hla = [hla_list[i] for i in train_dataset.indices]
    val_hla = [hla_list[i] for i in val_dataset.indices]
    test_hla = [hla_list[i] for i in test_dataset.indices]

    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}")

    train_sampler = DynamicHLASampler(train_dataset, train_hla, config.neg_pos_ratio, config.batch_size)
    val_sampler = DynamicHLASampler(val_dataset, val_hla, config.neg_pos_ratio, config.batch_size)
    test_sampler = DynamicHLASampler(test_dataset, test_hla, config.neg_pos_ratio, config.batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, sampler=test_sampler)
    return train_dataloader, val_dataloader, test_dataloader

def initialize_model(config):
    try:
        model = mlp.MLPtrain(hidden_dims=[1024, 512, 256, 128], dropout_rate=0.5).to(config.device)
        dummy_input = torch.randn(1, 80, 1000).to(config.device)
        output = model(dummy_input)
        if output.shape != torch.Size([1, 1]):
            raise ValueError(f"预期输出形状为 [batch_size, 1]，实际为 {output.shape}")
        logger.info(f"模型初始化成功，输出形状: {output.shape}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数量: {total_params}")
        return model
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        raise

def train_epoch(model, dataloader, optimizer, criterion, config):
    model.train()
    epoch_loss = 0
    for i, (batch_x, batch_y) in enumerate(tqdm(dataloader, desc="训练批次")):
        batch_x = batch_x.to(config.device, non_blocking=True)
        batch_y = batch_y.to(config.device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y) / config.accum_steps
        loss.backward()
        epoch_loss += loss.item() * config.accum_steps

        if (i + 1) % config.accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.empty_cache()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, config):
    model.eval()
    labels_all = []
    outputs_all = []
    predicted_all = []
    confusion_matrices = []
    count = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True).unsqueeze(1)
            outputs = model(inputs)
            outputs_prob = torch.sigmoid(outputs)
            predicted = (outputs_prob > config.classification_threshold).float()

            labels_all.extend(labels.cpu().numpy().flatten())
            outputs_all.extend(outputs_prob.cpu().numpy().flatten())
            predicted_all.extend(predicted.cpu().numpy().flatten())
            cm = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1])
            confusion_matrices.append(cm)
            count += 1

            torch.cuda.empty_cache()

    labels_all = np.array(labels_all)
    outputs_all = np.array(outputs_all)
    predicted_all = np.array(predicted_all)

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
        if len(np.unique(labels_all)) > 1:
            metrics["auc"] = roc_auc_score(labels_all, outputs_all)
            metrics["aupr"] = average_precision_score(labels_all, outputs_all)
        metrics["accuracy"] = accuracy_score(labels_all, predicted_all)
        metrics["f1"] = f1_score(labels_all, predicted_all, zero_division=0)
        metrics["precision"] = precision_score(labels_all, predicted_all, zero_division=0)
        metrics["recall"] = recall_score(labels_all, predicted_all, zero_division=0)
        metrics["mcc"] = matthews_corrcoef(labels_all, predicted_all)
    except ValueError as e:
        logger.warning(f"指标计算错误: {e}")

    confusion = np.sum(confusion_matrices, axis=0)
    logger.info(
        f"输出概率统计 - 均值: {outputs_all.mean():.4f}, 标准差: {outputs_all.std():.4f}, 最小值: {outputs_all.min():.4f}, 最大值: {outputs_all.max():.4f}")
    return metrics, confusion

def save_results(val_metrics, test_metrics, test_confusion, config):
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/output.txt", "a", encoding="utf-8") as file:
            for epoch in range(len(val_metrics["auc"])):
                file.write(f"Epoch {epoch} 验证集 - "
                           f"AUC: {val_metrics['auc'][epoch]:.4f}, "
                           f"AUPR: {val_metrics['aupr'][epoch]:.4f}, "
                           f"准确率: {val_metrics['accuracy'][epoch]:.4f}, "
                           f"F1: {val_metrics['f1'][epoch]:.4f}, "
                           f"精确率: {val_metrics['precision'][epoch]:.4f}, "
                           f"召回率: {val_metrics['recall'][epoch]:.4f}, "
                           f"MCC: {val_metrics['mcc'][epoch]:.4f}\n")
            file.write(f"测试集 - "
                       f"AUC: {test_metrics['auc']:.4f}, "
                       f"AUPR: {test_metrics['aupr']:.4f}, "
                       f"准确率: {test_metrics['accuracy']:.4f}, "
                       f"F1: {test_metrics['f1']:.4f}, "
                       f"精确率: {test_metrics['precision']:.4f}, "
                       f"召回率: {test_metrics['recall']:.4f}, "
                       f"MCC: {test_metrics['mcc']:.4f}\n")
            file.write(f"测试集混淆矩阵:\n{test_confusion}\n")
        logger.info("结果已保存至 data/output.txt")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="训练 MLP 模型用于二分类，带比例采样")
    parser.add_argument('--file_path_train', type=str, required=True, help='训练数据集路径')
    parser.add_argument('--epoch', type=int, required=True, help='训练轮数')
    parser.add_argument('--data_transfomer_esm', type=str, required=True, help='ESM 转换器数据路径')
    parser.add_argument('--model_esm_path', type=str, required=True, help='模型保存路径')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--accum_steps', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--neg_pos_ratio', type=float, default=2.0, help='负样本:正样本比例')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
    args = parser.parse_args()

    config = Config(args)
    logger.info(f"使用设备: {config.device}")

    esm_data, labels, num_pos, num_neg, hla_list = load_data(config)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(esm_data, labels, hla_list, config)
    model = initialize_model(config)

    pos_weight = torch.tensor([num_neg / num_pos * 0.5]).to(config.device) if num_pos > 0 else torch.tensor([1.0]).to(
        config.device)
    logger.info(f"正类权重: {pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    val_metrics = {
        "auc": [], "aupr": [], "accuracy": [],
        "f1": [], "precision": [], "recall": [], "mcc": []
    }
    best_val_auc = 0
    patience_counter = 0
    best_model_path = config.model_esm_path + ".best"

    for epoch in tqdm(range(config.epochs), desc="训练 Epoch"):
        loss = train_epoch(model, train_dataloader, optimizer, criterion, config)
        logger.info(f"Epoch {epoch}, 损失: {loss:.4f}")

        metrics, _ = evaluate(model, val_dataloader, config)
        for key in val_metrics:
            val_metrics[key].append(metrics[key])
        logger.info(f"Epoch {epoch} 验证集 - "
                    f"AUC: {metrics['auc']:.4f}, "
                    f"AUPR: {metrics['aupr']:.4f}, "
                    f"准确率: {metrics['accuracy']:.4f}, "
                    f"F1: {metrics['f1']:.4f}, "
                    f"精确率: {metrics['precision']:.4f}, "
                    f"召回率: {metrics['recall']:.4f}, "
                    f"MCC: {metrics['mcc']:.4f}")

        if metrics["auc"] > best_val_auc:
            best_val_auc = metrics["auc"]
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"保存最佳模型至 {best_model_path}, AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"早停触发于 epoch {epoch}")
                break

        scheduler.step(metrics["auc"])

    try:
        torch.save(model.state_dict(), config.model_esm_path)
        logger.info(f"最终模型已保存至 {config.model_esm_path}")
    except Exception as e:
        logger.error(f"保存模型失败: {e}")

    test_metrics, test_confusion = evaluate(model, test_dataloader, config)
    logger.info(f"测试集 - "
                f"AUC: {test_metrics['auc']:.4f}, "
                f"AUPR: {test_metrics['aupr']:.4f}, "
                f"准确率: {test_metrics['accuracy']:.4f}, "
                f"F1: {test_metrics['f1']:.4f}, "
                f"精确率: {test_metrics['precision']:.4f}, "
                f"召回率: {test_metrics['recall']:.4f}, "
                f"MCC: {test_metrics['mcc']:.4f}")
    logger.info(f"测试集混淆矩阵:\n{test_confusion}")

    save_results(val_metrics, test_metrics, test_confusion, config)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()