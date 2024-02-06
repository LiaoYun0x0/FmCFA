import torch
import torch.nn as nn
from typing import Callable, Optional, Union, Tuple, List
import torch.nn.functional as F
def qk_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.qk_map = qk_map
        self.eps = eps

    def forward(self, q, k, v, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, V]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, V)
        """
        Q = self.qk_map(q)
        K = self.qk_map(k)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            v = v * kv_mask[:, :, None, None]

        v_length = v.size(1)
        v = v / v_length  # prevent fp16 overflow

        # KV = torch.matmul(K.permute(0,2,3,1), V.permute(0,2,1,3))
        KV = torch.einsum("nshd,nshv->nhdv", K, v)  # (S,D)' @ S,V

        # compuet  similarity between each query and the mean of keys  
        # Z = 1 / ( (Q * (K.sum(dim=1)+self.eps)).sum(-1) ) 
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)


        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()

class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class FullAttention(nn.Module):
    def __init__(self):
        super(FullAttention, self).__init__()

    def forward(self, query, key, value):
        #         # Compute the unnormalized attention and apply the masks
        # QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        # if kv_mask is not None:
        #     QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # # Compute the attention and the weighted average
        # softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        # A = torch.softmax(softmax_temp * QK, dim=2)
        # if self.use_dropout:
        #     A = self.dropout(A)

        # queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        # return queried_values.contiguous()
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        
        # 归一化注意力分数
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 使用注意力权重加权平均值来计算输出
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.attention = FullAttention()
        self.merge = nn.Linear(embed_dim, embed_dim, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim*2, bias=False),
            nn.ReLU(True),
            nn.Linear(embed_dim*2, embed_dim, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x,source): #x:[b*num_window,c,window_size,window_size]
        b,c,h,w = x.shape #b: batch_size * nums_window
        x = x.view(b,c,h*w).permute(0,2,1)
        source = source.view(b,c,h*w).permute(0,2,1)
        b, n, embed_dim = x.size()
        # 将输入分割成多个头
        query = self.query(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(source).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(source).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用LinearAttention计算自注意力
        attention_output, _ = self.attention(query, key, value)

        attention_output = self.merge(attention_output.view(b, -1, self.num_heads*self.head_dim))  # [N, L, C]
        attention_output = self.norm1(attention_output)

        # feed-forward network
        attention_output = self.mlp(torch.cat([x, attention_output], dim=2))
        attention_output = self.norm2(attention_output) 
        
        # 将多个头的输出连接起来
        # attention_output = attention_output.transpose(1, 2).contiguous().view(b, n, embed_dim)
        
        return x + attention_output


def window_partition_nchw(x, window_size: List[int]):
    B, C, H, W = x.shape
    assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    assert(W % window_size[1] == 0, '')
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size[0], window_size[1])
    return windows

def window_reverse_nchw(windows, window_size: List[int], img_size: List[int]):#[b*num_window,window_size*window_size,c]
    H, W = img_size
    C = windows.shape[-1]
    windows = windows.permute(0,2,1).view(-1,C,window_size[0],window_size[1])
    x = windows.view(-1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1])
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x

class WindowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super(WindowAttention, self).__init__()
        self.multihead_attention = MultiheadSelfAttention(embed_dim, num_heads)
        self.window_size = window_size
        self.dim = embed_dim

    def forward(self, x,source,mask0=None,mask1=None):
        x0 = x
        b,embed_dim,h,w = x.shape
        n = h*w
        nums_window = (h//self.window_size) * (w//self.window_size)
        #b, n, embed_dim = x.size()
        # 将输入划分成窗口
        x = window_partition_nchw(x,[self.window_size,self.window_size]) #x:[b*num_window,c,window_size,window_size]
        source = window_partition_nchw(source,[self.window_size,self.window_size])
        x_win_attn = self.multihead_attention(x,x)
        s_win_attn = self.multihead_attention(source,source)

        return x_win_attn,s_win_attn,nums_window,h,w #[b*num_window,h*w,c]

class WinTopKAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, topk):
        super(WinTopKAttention, self).__init__()
        self.window_size = window_size
        self.topk = topk
        self.linear_attention = EncoderLayer(embed_dim,num_heads)
        self.dim = embed_dim

    def forward(self,x_win_attn,s_win_attn,attn_name,batch_size,nums_window,h,w ,mask0=None,mask1=None): #x: [b*num_window,h*w,c]
        # x_win_attn 和 s_win_attn是一样的
        b = batch_size
        embed_dim = self.dim
        
        if attn_name == 'self':
            '''平均池化计算相似性'''
            average_pooled = torch.mean(x_win_attn, dim=1, keepdim=True)
            cosine_similarities = F.cosine_similarity(average_pooled, x_win_attn, dim=2)
            topk_values, indices = torch.topk(cosine_similarities, k=self.topk, dim=1)
  
            window_attention_topk = torch.gather(x_win_attn, 1, indices.unsqueeze(2).expand(-1, -1, 256))
            window_attention_topk = window_attention_topk.view(b,nums_window*self.topk,embed_dim)
            
            
            window_attention_output = self.linear_attention(window_attention_topk, window_attention_topk) #[b*num_window,topk,c]
            window_attention_output = window_attention_output.view(b*nums_window,self.topk,embed_dim)
            indices = indices.view(b * nums_window, self.topk, 1)
            
            combined_attention = x_win_attn.clone()
            combined_attention.scatter_add_(dim=1, index=indices.expand(-1, -1, embed_dim), src=window_attention_output)

            win_output = window_reverse_nchw(combined_attention,[self.window_size,self.window_size],[h,w])
        elif attn_name == 'cross':
            '''平均池化计算相似性'''
            average_pooled = torch.mean(x_win_attn, dim=1, keepdim=True)
            cosine_similarities = F.cosine_similarity(average_pooled, x_win_attn, dim=2)
            topk_values1, indices1 = torch.topk(cosine_similarities, k=self.topk, dim=1)

            window_attention_topk1 = torch.gather(x_win_attn, 1, indices1.unsqueeze(2).expand(-1, -1, 256))
            window_attention_topk1 = window_attention_topk1.view(b,nums_window*self.topk,embed_dim)
            
            '''平均池化计算相似性'''
            average_pooled = torch.mean(s_win_attn, dim=1, keepdim=True)
            cosine_similarities = F.cosine_similarity(average_pooled, s_win_attn, dim=2)
            topk_values2, indices2 = torch.topk(cosine_similarities, k=self.topk, dim=1)

            window_attention_topk2 = torch.gather(s_win_attn, 1, indices2.unsqueeze(2).expand(-1, -1, 256))
            window_attention_topk2 = window_attention_topk2.view(b,nums_window*self.topk,embed_dim)

            # 使用LinearAttention计算窗口之间的注意力
            window_attention_output = self.linear_attention(window_attention_topk1, window_attention_topk2) #[b*num_window,topk,c]
            window_attention_output = window_attention_output.view(b*nums_window,self.topk,embed_dim)
            indices1 = indices1.view(b * nums_window, self.topk, 1)

            # 使用 scatter_add 将 window_attention_output 添加回 window_attention
            combined_attention = x_win_attn.clone()
            combined_attention.scatter_add_(dim=1, index=indices1.expand(-1, -1, embed_dim), src=window_attention_output)

            win_output = window_reverse_nchw(combined_attention,[self.window_size,self.window_size],[h,w])

        return win_output #[b,c,h,w]

class WindowTopKAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, topk):
        super(WindowTopKAttention, self).__init__()
        self.multihead_attention = MultiheadSelfAttention(embed_dim, num_heads)
        self.window_size = window_size
        self.topk = topk
        self.linear_attention = EncoderLayer(embed_dim,num_heads)
        self.dim = embed_dim

    def forward(self,x,source,mask0=None,mask1=None): #x: [b,c,h,w]
        x0 = x
        b,embed_dim,h,w = x.shape
        n = h*w
        nums_window = (h//self.window_size) * (w//self.window_size)
        #b, n, embed_dim = x.size()
        # 将输入划分成窗口
        x = window_partition_nchw(x,[self.window_size,self.window_size]) #x:[b*num_window,c,window_size,window_size]
        source = window_partition_nchw(source,[self.window_size,self.window_size])
        if attn_name == 'self':
            # 计算每个窗口的注意力
            window_attention = self.multihead_attention(x,source) #[b*num_window,h*w,c]
            
            '''计算向量模长'''
            #vector_lengths = torch.norm(window_attention, dim=2)
            '''计算向量方差'''
            # vector_lengths = torch.var(window_attention,dim=2)
            # indices = torch.topk(vector_lengths, k=self.topk, dim=1)[1]
            # window_attention_topk = torch.gather(window_attention, dim=1, index=indices.unsqueeze(2).expand(-1, -1, 256))

            '''平均池化计算相似性'''
            average_pooled = torch.mean(window_attention, dim=1, keepdim=True)
            cosine_similarities = F.cosine_similarity(average_pooled, window_attention, dim=2)
            topk_values, indices = torch.topk(cosine_similarities, k=self.topk, dim=1)

            # 从window_attention中提取前K个张量
            window_attention_topk = torch.gather(window_attention, 1, indices.unsqueeze(2).expand(-1, -1, 256))
            # 选择每个窗口的topk个patches
            #window_attention_topk, indices = window_attention.topk(self.topk, dim=-2) # [b*num_window,topk,c]
            
            window_attention_topk = window_attention_topk.view(b,nums_window*self.topk,embed_dim)
            
            # 使用LinearAttention计算窗口之间的注意力
            window_attention_output = self.linear_attention(window_attention_topk, window_attention_topk) #[b*num_window,topk,c]
            window_attention_output = window_attention_output.view(b*nums_window,self.topk,embed_dim)
            indices = indices.view(b * nums_window, self.topk, 1)

            # 使用 scatter_add 将 window_attention_output 添加回 window_attention
            combined_attention = window_attention.clone()
            combined_attention.scatter_add_(dim=1, index=indices.expand(-1, -1, embed_dim), src=window_attention_output)

            win_output = window_reverse_nchw(combined_attention,[self.window_size,self.window_size],[h,w])
        elif attn_name == 'cross':
            # 计算每个窗口的注意力
            window_attention1 = self.multihead_attention(x,x) #[b*num_window,h*w,c]
            window_attention2 = self.multihead_attention(source,source)
            
            '''计算向量模长'''
            #vector_lengths = torch.norm(window_attention, dim=2)
            '''计算向量方差'''
            # vector_lengths = torch.var(window_attention,dim=2)
            # indices = torch.topk(vector_lengths, k=self.topk, dim=1)[1]
            # window_attention_topk = torch.gather(window_attention, dim=1, index=indices.unsqueeze(2).expand(-1, -1, 256))

            '''平均池化计算相似性'''
            average_pooled = torch.mean(window_attention1, dim=1, keepdim=True)
            cosine_similarities = F.cosine_similarity(average_pooled, window_attention1, dim=2)
            topk_values1, indices1 = torch.topk(cosine_similarities, k=self.topk, dim=1)

            # 从window_attention中提取前K个张量
            window_attention_topk1 = torch.gather(window_attention1, 1, indices.unsqueeze(2).expand(-1, -1, 256))
            window_attention_topk1 = window_attention_topk1.view(b,nums_window*self.topk,embed_dim)
            
            '''平均池化计算相似性'''
            average_pooled = torch.mean(window_attention2, dim=1, keepdim=True)
            cosine_similarities = F.cosine_similarity(average_pooled, window_attention2, dim=2)
            topk_values2, indices2 = torch.topk(cosine_similarities, k=self.topk, dim=1)

            # 从window_attention中提取前K个张量
            window_attention_topk2 = torch.gather(window_attention2, 1, indices.unsqueeze(2).expand(-1, -1, 256))
            window_attention_topk2 = window_attention_topk2.view(b,nums_window*self.topk,embed_dim)

            # 使用LinearAttention计算窗口之间的注意力
            window_attention_output = self.linear_attention(window_attention_topk1, window_attention_topk2) #[b*num_window,topk,c]
            window_attention_output = window_attention_output.view(b*nums_window,self.topk,embed_dim)
            indices = indices.view(b * nums_window, self.topk, 1)

            # 使用 scatter_add 将 window_attention_output 添加回 window_attention
            combined_attention = window_attention1.clone()
            combined_attention.scatter_add_(dim=1, index=indices.expand(-1, -1, embed_dim), src=window_attention_output)

            win_output = window_reverse_nchw(combined_attention,[self.window_size,self.window_size],[h,w])
        return win_output #[b,c,h,w]

    def reduce_dimension(self, x):
        return nn.Linear(x.size(-1), self.dim)(x)


