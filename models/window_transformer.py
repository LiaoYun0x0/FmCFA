import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(LinearAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)
        self.fc_o = nn.Linear(dim, dim)

        self.scale = 0.5

    def forward(self, x,source):
        b, n, _ = x.shape
        q = self.fc_q(x)
        k = self.fc_k(source)
        v = self.fc_v(source)
        
        q = q.view(b, n, self.num_heads, self.head_dim)
        k = k.view(b, n, self.num_heads, self.head_dim)
        v = v.view(b, n, self.num_heads, self.head_dim)
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        
        attn_scores = torch.matmul(q, k) / (self.head_dim ** self.scale)
        attn_scores = F.softmax(attn_scores, dim=-1)
        
        attention = torch.matmul(attn_scores, v)
        attention = attention.permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        output = self.fc_o(attention)
        return output

class WindowAttention(nn.Module):
    '''
        WindowAttention 模块接受输入 x
        然后将图像划分为窗口并在每个窗口内应用自定义的 LinearAttention 层，
        最终返回窗口内的 top-k 注意力信息。
    '''
    def __init__(self, dim, num_heads, window_size, top_k):
        super(WindowAttention, self).__init__()
        self.window_size = window_size  # 窗口大小
        self.top_k = top_k  # 保留的top-k注意力值数量
        self.linear_attention = LinearAttention(dim, num_heads)  # 使用自定义的LinearAttention层
        self.dim = dim

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入张量的形状，b是批次大小，c是通道数，h和w是高度和宽度
        x = x.view(b, c, h // self.window_size, self.window_size, w // self.window_size, self.window_size)
        # 将输入x重塑为多个小窗口，以便将注意力机制应用于窗口内部
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, c, self.window_size, self.window_size)
        # 将维度重新排列以适应注意力计算
        
        # 使用自定义的LinearAttention计算注意力
        attn_output = self.linear_attention(x)

        # 将输出重新排列为适当的形状，以便处理top-k信息
        attn_output = attn_output.view(-1, self.dim, self.top_k, self.window_size, self.window_size)

        # 再次调整输出形状
        attn_output = attn_output.permute(0, 2, 1, 3, 4).contiguous().view(-1, self.dim, self.top_k * self.window_size * self.window_size)
        
        
        return attn_output

class TopKAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, top_k):
        super(TopKAttention, self).__init__()
        self.window_attention = WindowAttention(dim, num_heads, window_size, top_k)
        self.linear_attention = LinearAttention(dim, num_heads)

    def forward(self, x):
        top_k_indices = self.window_attention(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h // self.window_size, self.window_size, w // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, c, self.window_size, self.window_size)
        
        top_k_values = torch.gather(x, 3, top_k_indices.unsqueeze(3).expand(-1, -1, -1, self.window_size))
        top_k_values = top_k_values.view(-1, c, self.top_k, self.window_size, self.window_size)
        top_k_values = top_k_values.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, self.top_k * self.window_size * self.window_size)
        
        # 使用LinearAttention计算注意力
        attn_output = self.linear_attention(top_k_values)
        
        return attn_output

class WindowTransformer(nn.Module):
    def __init__(self, dim, num_heads, window_size, top_k):
        super(WindowTransformer, self).__init__()
        self.window_attention = WindowAttention(dim, num_heads, window_size, top_k)
        self.top_k_attention = TopKAttention(dim, num_heads, window_size, top_k)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        window_attention_map = self.window_attention(x)
        top_k_attention_map = self.top_k_attention(x)
        combined_attention_map = window_attention_map + top_k_attention_map
        
        # 计算融合后的注意力权重
        attention_weights = F.softmax(combined_attention_map, dim=-1)
        
        # 将注意力权重应用于原始输入
        weighted_input = torch.matmul(attention_weights.unsqueeze(2), x.unsqueeze(1))
        weighted_input = weighted_input.squeeze(2)
        
        # 通过全连接层融合原始输入和注意力加权输入
        fused_output = self.fc  (weighted_input)
        
        return fused_output