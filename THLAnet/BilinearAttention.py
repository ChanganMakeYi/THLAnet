import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
import os
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BilinearMultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in1, dim_in2, dim_out, num_heads, seq_len1=80, seq_len2=79, dropout=0.3):
        super(BilinearMultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads
        self.dropout = nn.Dropout(dropout)
        self.seq_len1 = seq_len1
        self.seq_len2 = seq_len2

        if dim_out % num_heads != 0:
            raise ValueError("dim_out 必须能被 num_heads 整除")

        self.linear_q = nn.Linear(dim_in1, dim_out, bias=False)
        self.linear_k = nn.Linear(dim_in2, dim_out, bias=False)
        self.linear_v = nn.Linear(dim_in1, dim_out, bias=False)
        self.fc_out = nn.Linear(dim_out, dim_out)
        self.layer_norm = nn.LayerNorm(dim_out)

        self.align_seq = nn.Linear(seq_len2, seq_len1)

    def forward(self, x1_chunk, x2_chunk):
        x1_chunk, x2_chunk = x1_chunk.float(), x2_chunk.float()
        x2_chunk = self.align_seq(x2_chunk.transpose(1, 2)).transpose(1, 2)
        Q = self.linear_q(x1_chunk)
        K = self.linear_k(x2_chunk)
        V = self.linear_v(x1_chunk)
        batch_size = Q.size(0)
        Q = Q.view(batch_size, self.seq_len1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, self.seq_len1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, self.seq_len1, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, self.seq_len1, self.dim_out)
        output = self.fc_out(output)
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output

def validate_args(args):
    if not isinstance(args.esm_data, str) or not args.esm_data:
        raise ValueError("esm_data 必须为非空字符串路径")
    if not isinstance(args.transformer_data, str) or not args.transformer_data:
        raise ValueError("transformer_data 必须为非空字符串路径")
    if not os.path.exists(args.esm_data):
        raise FileNotFoundError(f"ESM 数据文件 {args.esm_data} 不存在")
    if not os.path.exists(args.transformer_data):
        raise FileNotFoundError(f"Transformer 数据文件 {args.transformer_data} 不存在")
    if args.num_heads < 1:
        raise ValueError("num_heads 必须至少为 1")
    if args.dim_out % args.num_heads != 0:
        raise ValueError("dim_out 必须能被 num_heads 整除")
    if args.chunk_size < 1:
        raise ValueError("chunk_size 必须至少为 1")
    if args.dropout < 0 or args.dropout > 1:
        raise ValueError("dropout 必须在 0 到 1 之间")
    if args.seq_len1 < 1 or args.seq_len2 < 1:
        raise ValueError("seq_len1 和 seq_len2 必须为正整数")

def load_data(file_path, data_name):
    logger.info(f"加载 {data_name}...")
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pt':
            data = torch.load(file_path, weights_only=False)
        elif ext == '.npy':
            data = np.load(file_path)
        else:
            logger.warning(f"未知文件扩展名 {ext}，尝试使用 torch.load")
            data = torch.load(file_path, weights_only=False)
    except Exception as e:
        raise ValueError(f"加载 {data_name} 失败: {e}. 请确保文件是有效的 .pt 或 .npy 格式，路径正确")

    return convert_to_tensor(data, data_name)

def convert_to_tensor(data, data_name):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        logger.info(f"将 {data_name} 从 NumPy 数组转换为 torch.Tensor")
        return torch.from_numpy(data)
    else:
        try:
            logger.warning(f"尝试将 {data_name} 从未知类型转换为 torch.Tensor")
            return torch.tensor(data)
        except Exception as e:
            raise ValueError(f"无法将 {data_name} 转换为 torch.Tensor: {e}")

def process_data_in_chunks(model, x1, x2, chunk_size, device):
    logger.info("开始分块处理数据...")
    model.eval()
    output_chunks = []
    for i in tqdm(range(0, len(x1), chunk_size), desc="处理分块"):
        x1_chunk = x1[i:i + chunk_size].to(device)
        x2_chunk = x2[i:i + chunk_size].to(device)
        with torch.no_grad():
            output_chunk = model(x1_chunk, x2_chunk)
            output_chunks.append(output_chunk.cpu())
        del x1_chunk, x2_chunk, output_chunk
        torch.cuda.empty_cache()
    return torch.cat(output_chunks, dim=0)

def validate_output(output, expected_shape):
    if output.shape != expected_shape:
        logger.warning(f"输出形状 {output.shape} 与预期 {expected_shape} 不匹配")
    if torch.isnan(output).any() or torch.isinf(output).any():
        logger.warning("输出中包含 NaN 或 Inf 值，可能影响后续训练")
    return output

def main():
    parser = argparse.ArgumentParser(description="双线性多头自注意力机制处理蛋白质序列数据")
    parser.add_argument('--esm_data', type=str, required=True, help='ESM 数据文件路径')
    parser.add_argument('--transformer_data', type=str, required=True, help='Transformer 数据文件路径')
    parser.add_argument('--output_path', type=str, default='bilinear_attention_output.pt',
                        help='输出文件路径')
    parser.add_argument('--dim_in1', type=int, default=1280, help='ESM 数据输入维度')
    parser.add_argument('--dim_in2', type=int, default=21, help='Transformer 数据输入维度')
    parser.add_argument('--dim_out', type=int, default=1000, help='输出维度')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--seq_len1', type=int, default=80, help='ESM 数据序列长度')
    parser.add_argument('--seq_len2', type=int, default=80, help='Transformer 数据序列长度')
    parser.add_argument('--dropout', type=float, default=0.3, help='丢弃率')
    parser.add_argument('--chunk_size', type=int, default=2000, help='分块大小')

    args = parser.parse_args()

    try:
        validate_args(args)
    except Exception as e:
        logger.error(f"参数验证失败: {e}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    try:
        x1 = load_data(args.esm_data, "ESM 数据")
        x2 = load_data(args.transformer_data, "Transformer 数据")
        logger.info(f"ESM 数据形状: {x1.shape}")
        logger.info(f"Transformer 数据形状: {x2.shape}")
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return

    if x2.size(1) == 21 and x2.size(2) == 80:
        logger.info("检测到 Transformer 数据维度顺序为 [n, 21, 79]，正在转置为 [n, 79, 21]")
        x2 = x2.transpose(1, 2)

    if x1.size(0) != x2.size(0):
        logger.error("ESM 和 Transformer 数据的样本数不匹配")
        return
    if x1.size(1) != args.seq_len1 or x1.size(2) != args.dim_in1:
        logger.error(f"ESM 数据形状应为 [n, {args.seq_len1}, {args.dim_in1}]")
        return
    if x2.size(1) != args.seq_len2 or x2.size(2) != args.dim_in2:
        logger.error(f"Transformer 数据形状应为 [n, {args.seq_len2}, {args.dim_in2}]")
        return

    try:
        logger.info("初始化双线性注意力模型...")
        model = BilinearMultiHeadSelfAttention(
            dim_in1=args.dim_in1,
            dim_in2=args.dim_in2,
            dim_out=args.dim_out,
            num_heads=args.num_heads,
            seq_len1=args.seq_len1,
            seq_len2=args.seq_len2,
            dropout=args.dropout
        ).to(device)
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        return

    try:
        output = process_data_in_chunks(model, x1, x2, args.chunk_size, device)
        expected_shape = (x1.size(0), args.seq_len1, args.dim_out)
        output = validate_output(output, expected_shape)
        logger.info(f"输出数据形状: {output.shape}")
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        return

    try:
        torch.save(output, args.output_path)
        logger.info(f"输出已保存至 {args.output_path}")
    except Exception as e:
        logger.error(f"保存输出失败: {e}")

if __name__ == "__main__":
    main()