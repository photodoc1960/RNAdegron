import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rinalmo.model.model import RiNALMo
from rinalmo.data.alphabet import Alphabet
from rinalmo.data.constants import RNA_TOKENS, MASK_TKN
import pickle


#mish activation
# not actually used in the code, but probably worth trying at some point
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))


from torch.nn.parameter import Parameter

# he defined a really cool pooling function somewhere between average and max pooling
# and again, DIDN'T USE IT!!

def gem(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.gamma=nn.Parameter(torch.tensor(100.0))

    def forward(self, q, k, v, mask=None, attn_mask=None):

        attn = torch.matmul(q, k.transpose(2, 3))/ self.temperature
        to_plot=attn[0,0].detach().cpu().numpy()

        if mask is not None:
            #attn = attn.masked_fill(mask == 0, -1e9)
            #attn = attn#*self.gamma
            attn = attn+mask*self.gamma
        if attn_mask is not None:
            attn=attn.float().masked_fill(attn_mask == 0, float('-1e-9'))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        #self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None,src_mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask  # For head axis broadcasting

        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(-1).float()
            attn_mask = torch.matmul(src_mask, src_mask.transpose(1, 2))  # [batch, seq_len, seq_len]

            # Explicit authoritative resizing to match attention dimensions
            if attn_mask.shape[-2:] != (q.shape[2], q.shape[2]):
                attn_mask = F.interpolate(attn_mask.unsqueeze(1), size=(q.shape[2], q.shape[2]),
                                          mode='nearest').squeeze(1)

            attn_mask = attn_mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            q, attn = self.attention(q, k, v, mask=mask, attn_mask=attn_mask)
            assert attn_mask.shape[-2:] == (q.shape[2], q.shape[2]), \
                f"attn_mask shape explicitly must match attention dimensions {(q.shape[2], q.shape[2])}, got {attn_mask.shape[-2:]}"
        else:
            q, attn = self.attention(q, k, v, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        return q, attn


class EnhancedMaskProjection(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=32, out_channels=8, kernel_size=3):
        super().__init__()
        self.spatial_context = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.spatial_context(x)


class ConvTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, k=3):
        super(ConvTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model // nhead, d_model // nhead, dropout=dropout)

        # Calculate precise padding for both even and odd kernel sizes
        if k % 2 == 0:  # Even kernel size
            self.padding = k // 2
            # For even kernels, we need special handling for output padding
            self.output_padding = (0, 0)
        else:  # Odd kernel size
            self.padding = (k - 1) // 2
            self.output_padding = (0, 0)

        # Store kernel size for forward pass reference
        self.kernel_size = k

        # Spatial dimension preservation with generalized padding parameters
        self.mask_conv1 = nn.Sequential(
            nn.Conv2d(4, nhead // 4, kernel_size=k, padding=self.padding),
            nn.BatchNorm2d(nhead // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nhead // 4, nhead, kernel_size=1)
        )

        # Transposed convolution with matching padding
        self.mask_deconv = nn.Sequential(
            nn.ConvTranspose2d(nhead, nhead // 4, kernel_size=k, padding=self.padding),
            nn.BatchNorm2d(nhead // 4),
            nn.Sigmoid()
        )

        # Remaining initialization unchanged
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.dense_in = nn.Linear(d_model, d_model)
        self.dense_out = nn.Linear(d_model, d_model)

        self.dim_reducer = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, stride=2)
        self.dim_reducer.weight = nn.Parameter(self.dim_reducer.weight.half())  # Cast weights to half-precision
        self.dim_reducer.bias = nn.Parameter(self.dim_reducer.bias.half())  # cast bias if it's also causing issues.

    def forward(self, src, mask, src_mask=None):
        res = src
        src = self.norm3(self.dense_in(src))
        seq_len = src.size(1)

        # Initial dimension verification and normalization
        if mask.shape[-2:] != (seq_len, seq_len):
            mask = F.interpolate(mask, size=(seq_len, seq_len), mode='bilinear', align_corners=False)

        # Store original mask for residual connection
        mask_res = mask.clone()

        # Apply convolution sequence with proper padding
        mask = self.mask_conv1(mask)

        # Precise dimension verification (critical) - must match sequence length exactly
        if mask.shape[-2:] != (seq_len, seq_len):
            # Deterministic correction for even kernel sizes which may produce off-by-one dimensions
            mask = F.interpolate(mask, size=(seq_len, seq_len), mode='bilinear', align_corners=False)

        assert mask.shape[-2:] == (seq_len, seq_len), \
            f"Mask dimensions {mask.shape[-2:]} must match sequence dimensions {(seq_len, seq_len)}"

        # Self-attention mechanism
        src2, attention_weights = self.self_attn(src, src, src, mask=mask, src_mask=src_mask)

        # Transformer encoder processing
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = res + self.dropout3(self.dense_out(src))
        src = self.norm4(src)

        # Apply transposed convolution
        deconv_mask = self.mask_deconv(mask)

        # Ensure precision compatibility before dim_reducer operation
        deconv_mask = deconv_mask.half()  # Cast to half-precision to match parameters
        deconv_mask = self.dim_reducer(deconv_mask)

        # Resize tensor to match spatial dimensions for residual connection
        deconv_mask = F.interpolate(
            deconv_mask.float(),
            size=(mask_res.shape[2], mask_res.shape[3]),
            mode='nearest'  # Significantly faster than bilinear
        )
        # print(deconv_mask.shape)
        # print(mask_res.shape)
        mask = deconv_mask + mask_res

        return src, attention_weights, mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class RNADegformer(nn.Module):
    def __init__(self, ntoken, nclass, ninp, nhead, nhid, nlayers, kmer_aggregation, kmers,
                 stride=1, dropout=0.5, pretrain=False, return_aw=False, rinalmo_weights_path=None):
        super(RNADegformer, self).__init__()
        self.use_cnn = True
        self.model_type = 'Transformer'
        self.src_mask = None
        self.kmers = kmers
        self.ninp = ninp
        self.pretrain = pretrain
        self.return_aw = return_aw

        # --- RiNALMo pretrained model ---
        self.alphabet = Alphabet(standard_tkns=RNA_TOKENS)
        self.mask_token_id = self.alphabet.tkn_to_idx[MASK_TKN]
        from rinalmo.config import model_config

        config = model_config('micro')  # Explicit config
        self.rinalmo = RiNALMo(config)

        state_dict = torch.load(rinalmo_weights_path, map_location='cpu')
        self.rinalmo.load_state_dict(state_dict)
        self.rinalmo.eval()

        for param in self.rinalmo.parameters():
            param.requires_grad = False

        # --- 5-mer CNN Embedding Stack ---
        self.kmer_embedding = nn.Embedding(num_embeddings=21, embedding_dim=ninp // 4)  # 128D
        self.kmer_cnn = nn.Conv1d(
            in_channels=ninp // 4,
            out_channels=ninp // 4,
            kernel_size=5,
            stride=1,
            padding=2  # maintain sequence length
        )

        # --- Fusion Projection ---
        self.fusion_layer = nn.Sequential(
            nn.Linear(480 + (ninp // 4), ninp),  # 480 + 128 --> 512
            nn.LayerNorm(ninp)
        )

        # --- Transformer Encoder Stack ---
        self.transformer_encoder = nn.ModuleList([
            ConvTransformerEncoderLayer(ninp, nhead, nhid, dropout, k=max(1, kmers[0] - i))
            for i in range(nlayers)
        ])

        # --- Final Prediction Heads ---
        self.decoder = nn.Linear(ninp, nclass)  # For supervised phase

        self.mask_dense = nn.Conv2d(4, nhead // 4, 1)  # BPP + distance conditioning

        self.pretrain_decoders = nn.ModuleDict({
            'nucleotide': nn.Linear(ninp, 4),
            'structure': nn.Linear(ninp, 10),
            'loop': nn.Linear(ninp, 7),
            'bpp': nn.Linear(ninp, 1)
        })

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, rinalmo_emb, bpp, src_mask):
        if self.use_cnn:
            tokens = src[:, :, 0].long()
            x = self.kmer_embedding(tokens)
            x = x.transpose(1, 2)
            x = self.kmer_cnn(x)
            x = x.transpose(1, 2)
        else:
            x = 0

        combined = torch.cat([x, rinalmo_emb], dim=-1)
        combined = self.fusion_layer(combined)

        src = combined
        attention_maps = []

        mask = bpp  # pass raw BPP input directly

        for i, layer in enumerate(self.transformer_encoder):
            src, attention_weights, mask = layer(src, mask, src_mask[:, i])
            attention_maps.append(attention_weights)

        encoder_output = src

        if self.pretrain:
            output = {head_name: head(encoder_output) for head_name, head in self.pretrain_decoders.items()}
        else:
            output = self.decoder(encoder_output)

        return output, attention_maps
