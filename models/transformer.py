import copy
import torch
import torch.nn as nn
import math
from models.channel_transformer import ChannelTransformer,ChannelBlock
#from models.windowTopkTransformer import WindowTopKAttention
from models.windowTopkTransformer import WindowTopKAttention,WindowAttention,WinTopKAttention
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
    
class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = torch.nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

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

class _LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(_LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        win_attn = WindowAttention(config['d_model'], config['nhead'], 8)
        win_topk_attn = WinTopKAttention(config['d_model'], config['nhead'], 8,50)
        self.win_layers = nn.ModuleList([copy.deepcopy(win_attn) for _ in range(len(self.layer_names))])
        self.wintopk_layers = nn.ModuleList([copy.deepcopy(win_topk_attn) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


# class _LocalFeatureTransformer(nn.Module):
#     """A Local Feature Transformer (LoFTR) module."""

#     def __init__(self, config):
#         super(_LocalFeatureTransformer, self).__init__()

#         self.config = config
#         self.d_model = config['d_model']
#         self.nhead = config['nhead']
#         self.layer_names = config['layer_names']
#         encoder_layer = EncoderLayer(config['d_model'], config['nhead'], config['attention'])
#         #wtopk_layer = WindowTopKAttention(config['d_model'], config['nhead'],config["window_size"],config["topk"])
#         wtopk_layer = WindowTopKAttention(config['d_model'], config['nhead'],8,50) 
#         module_list = nn.ModuleList()
#         for i in range(len(self.layer_names)):
#             if i % 2 == 0:
#                 module_list.append(copy.deepcopy(wtopk_layer))
#             else:
#                 module_list.append(copy.deepcopy(encoder_layer))

#         self.layers = module_list
#         # self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
#         self._reset_parameters()

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)


class _LocalFineFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(_LocalFineFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = EncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class GlobalFeatureTransformer(_LocalFeatureTransformer):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(GlobalFeatureTransformer, self).__init__(config)
    
    def forward(self, feat, mask=None):
        """
        Args:
            feat (torch.Tensor): [N, L, C]
            mask (torch.Tensor): [N, L] (optional)
        """
        assert self.d_model == feat.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
                feat = layer(feat, feat, mask, mask)
        return feat
    
class LocalFeatureTransformer(_LocalFeatureTransformer):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__(config)
    
    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        #assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
        b,c,h0,w0 = feat0.shape
        _,_,h1,w1 = feat1.shape



        for win_layer,wintopk_layer, name in zip(self.win_layers,self.wintopk_layers, self.layer_names):
            feat0_win ,feat1_win,nums_window,h,w  = win_layer(feat0,feat1)
            feat0 = wintopk_layer(feat0_win,feat1_win,name,b,nums_window,h,w )
            feat1 = wintopk_layer(feat0_win,feat1_win,name,b,nums_window,h,w )
            # if name == 'self':
            #     if(feat0.ndim == 3):#[b,n,c] -> [b,c,h,w]
            #         feat0 = feat0.transpose(1,2).view(b,c,h0,w0)
            #         feat1 = feat1.transpose(1,2).view(b,c,h1,w1)
            #     feat0 = layer(feat0, feat0, mask0, mask0)
            #     feat1 = layer(feat1, feat1, mask1, mask1)
            # elif name == 'cross':
            #     feat0 = feat0.view(b,c,h0*w0).transpose(1,2)
            #     feat1 = feat1.view(b,c,h1*w1).transpose(1,2)
            #     feat0 = layer(feat0, feat1, mask0, mask1)
            #     feat1 = layer(feat1, feat0, mask1, mask0)
            # else:
            #     raise KeyError

        feat0 = feat0.view(b,c,h0*w0).transpose(1,2)
        feat1 = feat1.view(b,c,h1*w1).transpose(1,2)

        return feat0, feat1
# class LocalFeatureTransformer(_LocalFeatureTransformer):
#     """A Local Feature Transformer (LoFTR) module."""

#     def __init__(self, config):
#         super(LocalFeatureTransformer, self).__init__(config)
    
#     def forward(self, feat0, feat1, mask0=None, mask1=None):
#         """
#         Args:
#             feat0 (torch.Tensor): [N, L, C]
#             feat1 (torch.Tensor): [N, S, C]
#             mask0 (torch.Tensor): [N, L] (optional)
#             mask1 (torch.Tensor): [N, S] (optional)
#         """

#         #assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
#         b,c,h0,w0 = feat0.shape
#         _,_,h1,w1 = feat1.shape



#         for layer, name in zip(self.layers, self.layer_names):
#             if name == 'self':
#                 if(feat0.ndim == 3):#[b,n,c] -> [b,c,h,w]
#                     feat0 = feat0.transpose(1,2).view(b,c,h0,w0)
#                     feat1 = feat1.transpose(1,2).view(b,c,h1,w1)
#                 feat0 = layer(feat0, feat0, mask0, mask0)
#                 feat1 = layer(feat1, feat1, mask1, mask1)
#             elif name == 'cross':
#                 feat0 = feat0.view(b,c,h0*w0).transpose(1,2)
#                 feat1 = feat1.view(b,c,h1*w1).transpose(1,2)
#                 feat0 = layer(feat0, feat1, mask0, mask1)
#                 feat1 = layer(feat1, feat0, mask1, mask0)
#             else:
#                 raise KeyError

#         return feat0, feat1


class LocalFineFeatureTransformer(_LocalFineFeatureTransformer):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFineFeatureTransformer, self).__init__(config)
    
    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1

class LocalFeatureTransformerAndChannel(_LocalFeatureTransformer):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformerAndChannel, self).__init__(config)
        self.stages = 8
        self.dim = 256
        self.num_heads = 8
        self.channel_blocks0 = nn.ModuleList([
            ChannelBlock(dim=self.dim,
                         num_heads=self.num_heads,
                         drop_path=0.)
                         for _ in range(self.stages)
        ])

        self.channel_blocks1 = nn.ModuleList([
            ChannelBlock(dim=self.dim,
                         num_heads=self.num_heads,
                         drop_path=0.)
                         for _ in range(self.stages)
        ])

    def forward(self, feat0, feat1, size,mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        idx = 0
        feat0_size = size
        feat1_size = size

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0) #1600 256
                feat1 = layer(feat1, feat1, mask1, mask1)
                feat0,feat0_size = self.channel_blocks0[idx](feat0,feat0_size)
                feat1,feat1_size = self.channel_blocks1[idx](feat1,feat1_size)
                idx+=1
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
                feat0,feat0_size = self.channel_blocks0[idx](feat0,feat0_size)
                feat1,feat1_size = self.channel_blocks1[idx](feat1,feat1_size)
                idx+=1
            else:
                raise KeyError

        return feat0, feat1



class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]