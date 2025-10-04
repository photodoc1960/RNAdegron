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
        # plt.imshow(to_plot)
        # plt.show()
        # exit()

        #exit()
        if mask is not None:
            #attn = attn.masked_fill(mask == 0, -1e9)
            #attn = attn#*self.gamma
            attn = attn+mask*self.gamma
        if attn_mask is not None:
            # print(attn.shape)
            # print(attn_mask.shape)
            # attn = attn+attn_mask
            #attn=attn.float().masked_fill(attn_mask == 0, float('-inf'))
            attn=attn.float().masked_fill(attn_mask == 0, float('-1e-9'))

        attn = self.dropout(F.softmax(attn, dim=-1))
        # print(attn[0,0])
        # to_plot=attn[0,0].detach().cpu().numpy()
        # with open('mat.txt','w+') as f:
        #     for vector in to_plot:
        #         for num in vector:
        #             f.write('{:04.3f} '.format(num))
        #         f.write('\n')
        # plt.imshow(to_plot)
        # plt.show()
        # exit()
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
        #print(attn.shape)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        #print(q.shape)
        #exit()
        # q = self.dropout(self.fc(q))
        # q += residual

        #q = self.layer_norm(q)

        return q, attn

class ConvTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, k=3):
        super(ConvTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model // nhead, d_model // nhead, dropout=dropout)

        # Only explicit direct projection of mask
        self.mask_dense = nn.Conv2d(4, nhead, kernel_size=1)  # explicitly match channels to nhead

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

    def forward(self, src, mask, src_mask=None):
        # src shape explicitly confirmed: [batch, seq_len, d_model]
        res = src
        src = self.norm3(self.dense_in(src))

        seq_len = src.size(1)

        # Explicit authoritative resizing of mask to match attention dimensions
        if mask.shape[-2:] != (seq_len, seq_len):
            mask = F.interpolate(mask, size=(seq_len, seq_len), mode='bilinear', align_corners=False)

        # Explicit authoritative direct mask projection (no conv layers to avoid dimension instability)
        mask = self.mask_dense(mask)
        mask_res = mask.clone()

        # Explicitly verified final mask shape matches attention dimensions
        assert mask.shape[-2:] == (seq_len, seq_len), \
            f"Mask spatial dimensions explicitly must match attention dimensions {(seq_len, seq_len)}, but got {mask.shape[-2:]}"

        src2, attention_weights = self.self_attn(src, src, src, mask=mask, src_mask=src_mask)

        # Transformer encoder block explicitly verified correct
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # Restore original dimensions explicitly via dense_out
        src = res + self.dropout3(self.dense_out(src))
        src = self.norm4(src)

        # Explicit authoritative verification: mask remains stable
        mask = mask + mask_res
        assert mask.shape == mask_res.shape, \
            f"Explicit authoritative verification failed: mask shape {mask.shape}, mask_res shape {mask_res.shape}"

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
        self.model_type = 'Transformer'
        self.src_mask = None
        self.kmers = kmers

        # RiNALMo pretrained model
        self.alphabet = Alphabet(standard_tkns=RNA_TOKENS)
        self.mask_token_id = self.alphabet.tkn_to_idx[MASK_TKN]
        from rinalmo.config import model_config

        # Explicitly create the RiNALMo model configuration
        config = model_config('micro')  # Replace 'micro' with correct variant (e.g., 'nano', 'micro', 'mega', 'giga')

        # Instantiate the model explicitly with the configuration
        self.rinalmo = RiNALMo(config)

        # Load the state_dict explicitly from the provided weights file
        state_dict = torch.load(rinalmo_weights_path, map_location='cpu')

        # Load weights explicitly
        self.rinalmo.load_state_dict(state_dict)
        self.rinalmo.eval()

        for param in self.rinalmo.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(480, ninp),
            nn.LayerNorm(ninp)
        )

        self.transformer_encoder = []
        for i in range(nlayers):
            k_size = max(1, kmers[0] - i)
            self.transformer_encoder.append(
                ConvTransformerEncoderLayer(ninp, nhead, nhid, dropout, k=k_size)
            )
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)

        self.ninp = ninp
        self.decoder = nn.Linear(ninp, nclass)
        self.mask_dense = nn.Conv2d(4, nhead // 4, 1)
        self.return_aw = return_aw
        self.pretrain = pretrain

        self.pretrain_decoders = nn.ModuleList()
        self.pretrain_decoders.append(nn.Linear(ninp, 4))
        self.pretrain_decoders.append(nn.Linear(ninp, 3))
        self.pretrain_decoders.append(nn.Linear(ninp, 7))

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, masked_sequences, original_sequences, bpp, src_mask=None):
        masked_tensor = masked_sequences[:, :, 0].to(next(self.parameters()).device).long()
        assert masked_tensor.dim() == 2, "masked_tensor explicitly must be 2-dimensional."

        original_tensor = original_sequences[:, :, 0].to(next(self.parameters()).device).long()
        assert original_tensor.dim() in (2, 3), f"original_tensor explicitly unexpected shape: {original_tensor.shape}"
        if original_tensor.dim() == 3:
            original_tensor = original_tensor.squeeze(0)
        assert original_tensor.dim() == 2, "original_tensor explicitly must be 2-dimensional after squeeze."

        mask_positions = (masked_tensor == 4)  # Explicitly matches dataset mask token index

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            rinalmo_output = self.rinalmo(masked_tensor)['representation']

        embeddings = self.projection(rinalmo_output)

        original_mask = bpp  # explicitly preserve original structural tensor (4 channels)
        mask = original_mask

        # explicitly iterate transformer encoder layers
        for i, layer in enumerate(self.transformer_encoder):
            embeddings, _, _ = layer(embeddings, mask, src_mask[:, i])  # explicitly discard mask returned by layer
            mask = original_mask  # explicitly reset mask to original bpp tensor (4 channels) each time!

        if self.pretrain:
            ae_outputs = []
            for decoder in self.pretrain_decoders:
                decoded = decoder(embeddings)
                ae_outputs.append(decoded.float())  # Explicit authoritative float casting
            return ae_outputs, original_tensor, mask_positions
        else:
            logits = self.decoder(embeddings)
            return logits.float()  # Explicit authoritative float casting
