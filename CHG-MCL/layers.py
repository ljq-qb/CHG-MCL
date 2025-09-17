
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 该Attention模块实现了缩放点积注意力，用于计算输入query(q)、key(k)和value(v)之间的注意力分数，并根据这些分数加权v，最终输出一个上下文感知的向量。
class Attention(nn.Module):
    # 初始化Attention类
    def __init__(self, temperature, attn_dropout=0.1):

        super().__init__()
        self.temperature = temperature    # temperature：表示注意力分数的缩放因子。
        self.dropout = nn.Dropout(attn_dropout)    # 在softmax之后施加Dropout防止过拟合。

    # 前向传播forward
    def forward(self, q, k, v, mask=None):
        # 计算 q-k 相关性（未归一化的注意力分数）
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # 应用 Mask 机制
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # softmax+dropout
        attn = attn / abs(attn.min())
        attn = self.dropout(F.softmax(F.normalize(attn, dim=-1), dim=-1))
        # 计算 v 的加权求和
        output = torch.matmul(attn, v)

        return output, attn, v


# 该层实现了前馈神经网络，包含两层线性变换和激活函数GELU。用于处理每个位置的非线性变换。它接受输入并返回经过非线性变换后的输出。
class FeedForwardLayer(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


# 该层专门处理多模态输入数据。对于每一种输入模态（例如不同类型的特征），
# 通过各自独立的线性层（w_q, w_k, w_v）生成对应的查询、键、值向量。
class VariLengthInputLayer(nn.Module):
    def __init__(self, input_data_dims, d_k, d_v, n_head, dropout):
        super(VariLengthInputLayer, self).__init__()
        self.n_head = n_head
        self.dims = input_data_dims
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = []
        self.w_ks = []
        self.w_vs = []
        for i, dim in enumerate(self.dims):
            self.w_q = nn.Linear(dim, n_head * d_k, bias=False)
            self.w_k = nn.Linear(dim, n_head * d_k, bias=False)
            self.w_v = nn.Linear(dim, n_head * d_v, bias=False)

            self.w_qs.append(self.w_q)
            self.w_ks.append(self.w_k)
            self.w_vs.append(self.w_v)
            self.add_module('linear_q_%d_%d' % (dim, i), self.w_q)
            self.add_module('linear_k_%d_%d' % (dim, i), self.w_k)
            self.add_module('linear_v_%d_%d' % (dim, i), self.w_v)

        self.attention = Attention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.fc = nn.Linear(n_head * d_v, n_head * d_v)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_head * d_v, eps=1e-6)

    def forward(self, input_data, mask=None):

        temp_dim = 0
        bs = input_data.size(0)
        modal_num = len(self.dims)
        q = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(device)
        k = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(device)
        v = torch.zeros(bs, modal_num, self.n_head * self.d_v).to(device)

        for i in range(modal_num):
            w_q = self.w_qs[i]
            w_k = self.w_ks[i]
            w_v = self.w_vs[i]

            data = input_data[:, temp_dim: temp_dim + self.dims[i]]
            temp_dim += self.dims[i]
            q[:, i, :] = w_q(data)
            k[:, i, :] = w_k(data)
            v[:, i, :] = w_v(data)

        q = q.view(bs, modal_num, self.n_head, self.d_k)
        k = k.view(bs, modal_num, self.n_head, self.d_k)
        v = v.view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn, residual = self.attention(q, k, v)
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        residual = residual.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        # 通过 Dropout 和 LayerNorm 对输出进行处理，以生成最终的多模态融合特征。
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn



# 将不同模态的数据经过多头自注意力（Multi-Head Attention）进行编码和融合，得到最终的统一表示。
class EncodeLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, dropout):
        super(EncodeLayer, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = Attention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, modal_num, mask=None):
        bs = q.size(0)
        residual = q
        q = self.w_q(q).view(bs, modal_num, self.n_head, self.d_k)
        k = self.w_k(k).view(bs, modal_num, self.n_head, self.d_k)
        v = self.w_v(v).view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn, _ = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


# 总体流程图解
# 1.输入数据处理：每种模态通过独立的线性变换层转换为对应的 q, k, v。
# 2.多模态注意力计算：对不同模态的 q, k, v 通过 Attention 模块进行加权和计算，生成融合后的特征。
# 3.多头注意力与前馈网络：通过 多头自注意力（EncodeLayer）进一步对多模态数据进行编码，提取高阶特征。使用 FeedForwardLayer 进行非线性变换，输出最终的表示。