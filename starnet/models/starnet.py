import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numbers
from einops import rearrange
from thop import profile
import time
import psutil
import os

# ============================================================= BottleneckLSTM ====================================================================
class BottleneckLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, norm='layer'):
        """
        From: https://github.com/vikrant7/mobile-vod-bottleneck-lstm/blob/master/network/mvod_bottleneck_lstm1.py
        Creates a bottleneck LSTM cell
        @param input_channels: number of input channels
        @param hidden_channels: number of hidden channels
        @param kernel_size: size of the kernel for convolutions (gates)
        @param norm: normalisation to use on output gates (default: LayerNorm) - Other normalisation not implemented yet
        """
        super(BottleneckLSTMCell, self).__init__()
        assert hidden_channels % 2 == 0
        
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.norm = norm

        # Depth-wise convolution for input processing
        self.W = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=kernel_size,
                           groups=self.input_channels, stride=1, padding=1)
        # 1x1 convolution for bottleneck gate
        self.Wy = nn.Conv2d(int(self.input_channels+self.hidden_channels), self.hidden_channels, kernel_size=1)
        # Depth-wise 3x3 convolution for bottleneck processing
        self.Wi = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1, stride=1,
                            groups=self.hidden_channels, bias=False)
        # 1x1 convolutions for gates
        self.Wbi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.Wbf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.Wbc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.Wbo = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        # Layer normalization for each gate
        if norm is not None:
            if norm == 'layer':
                self.norm_wbi = nn.GroupNorm(1, self.hidden_channels)
                self.norm_wbf = nn.GroupNorm(1, self.hidden_channels)
                self.norm_wbc = nn.GroupNorm(1, self.hidden_channels)
                self.norm_wbo = nn.GroupNorm(1, self.hidden_channels)

        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initialize bias of the cell (default to 1)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.fill_(1)

    def forward(self, x, h, c):
        """
        Forward pass Bottleneck LSTM cell
        @param x: input tensor with shape (B, C, H, W)
        @param h: hidden states
        @param c: cell states
        @return: new hidden states and new cell states
        """
        x = self.W(x)
        # Concatenate gate: concatenate input and hidden layers
        y = torch.cat((x, h), 1) 
        # Bottleneck gate: reduce to hidden layer size
        i = self.Wy(y) 
        b = self.Wi(i)	# depth wise 3*3
        
        # Input gate
        if self.norm is not None:
            ci = self.sigmoid(self.norm_wbi(self.Wbi(b)))
        else:
            ci = self.sigmoid(self.Wbi(b))

        # Forget gate
        if self.norm is not None:
            cf = self.sigmoid(self.norm_wbf(self.Wbf(b)))
        else:
            cf = self.sigmoid(self.Wbf(b))

        # Multiply forget gate with cell state + add output of
        # input gate multiplied by output of the conv after bottleneck gate
        if self.norm is not None:
            cc = cf * c + ci * self.relu(self.norm_wbc(self.Wbc(b)))
        else:
            cc = cf * c + ci * self.relu(self.Wbc(b))

        # Output gate
        if self.norm is not None:
            co = self.sigmoid(self.norm_wbo(self.Wbo(b)))
        else:
            co = self.sigmoid(self.Wbo(b))

        ch = co * self.relu(cc)
        return ch, cc

    @staticmethod
    def init_hidden(batch_size, hidden, shape):
        # Mandatory to specify cuda here as Pytorch Lightning doesn't do it automatically for new tensors
        if torch.cuda.is_available():
            h_init = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda()
            c_init = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda()
        else:
            h_init = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]))
            c_init = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]))
        return h_init, c_init


class BottleneckLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, norm='layer'):
        """
        Single layer Bottleneck LSTM cell
        @param input_channels: number of input channels of the cell
        @param hidden_channels: number of hidden channels of the cell
        @param norm: normalisation to use (default: LayerNorm) - Other normalisation are not implemented yet.
        """
        super(BottleneckLSTM, self).__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)

        self.cell = BottleneckLSTMCell(self.input_channels, self.hidden_channels, norm=norm)

    def forward(self, inputs, h, c):
        """
        Forward pass Bottleneck LSTM layer
        If stateful LSTM h and c must be None. Else they must be Tensor.
        @param inputs: input tensor
        @param h: hidden states (if None, they are automatically initialised)
        @param c: cell states (if None, they are automatically initialised)
        @return: new hidden and cell states
        """
        if h is None and c is None:
            h, c = self.cell.init_hidden(batch_size=inputs.shape[0], hidden=self.hidden_channels,
                                         shape=(inputs.shape[-2], inputs.shape[-1]))
        new_h, new_c = self.cell(inputs, h, c)
        return new_h, new_c

# ============================================================= ASPP ====================================================================
class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch=256, input_size=(500, 512)):
        super().__init__()
        self.input_size = input_size
        # Use adaptive global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        # Get input spatial dimensions
        _, _, h, w = x.shape
        
        # Global average pooling + 1x1 conv + upsample to original size
        x1 = self.global_avg_pool(x)
        x1 = self.conv1_1x1(x1)
        x1 = F.interpolate(x1, size=(h, w), mode='bilinear', align_corners=False)
        
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat


class DoubleConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        x = self.block(x)
        return x
    

class ConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        x = self.block(x)
        return x


class EncodingBranch(nn.Module):
    """
    Encoding branch for radar RD view

    PARAMETERS
    ----------
    n_frames: int
        Number of input frames
    """

    def __init__(self, n_frames):
        super().__init__()
        self.n_frames = n_frames
        # First conv block with double convolutions
        self.double_conv_block1 = DoubleConvBlock(in_ch=self.n_frames, out_ch=128, k_size=5,
                                                    pad=2, dil=1)
        # Special max pooling for range dimension (Doppler needs special handling)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=5,
                                                  pad=2, dil=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1,
                                                pad=0, dil=1)

    def forward(self, x):
        x1 = self.double_conv_block1(x)

        # range dimension needs special processing
        x1_pad = F.pad(x1, (0, 1, 0, 0), "constant", 0)
        x1_down = self.doppler_max_pool(x1_pad)
        x2 = self.double_conv_block2(x1_down)
        x2_pad = F.pad(x2, (0, 1, 0, 0), "constant", 0)
        x2_down = self.doppler_max_pool(x2_pad)

        x3 = self.single_conv_block1_1x1(x2_down)
        return x2_down, x3


class DecodingBranch(nn.Module):
    """
    Decoding branch for radar RD view

    PARAMETERS
    ----------  
    input_channels: int
        Number of input channels
    n_classes: int
        Number of output channels
    """

    def __init__(self, input_channels=896, n_classes=3):
        super().__init__()
        # First transpose conv for upsampling (special stride for Doppler dimension)
        self.rd_upconv1 = nn.ConvTranspose2d(input_channels, 128, (2, 1), stride=(2, 1))
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        # Second transpose conv for final upsampling
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, (2, 1), stride=(2, 1))
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        # Final 1x1 conv for classification
        self.rd_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)
    def forward(self, x):
        # range - upsample
        x1_rd = self.rd_upconv1(x)
        
        x2_rd = self.rd_double_conv_block1(x1_rd)

        x3_rd = self.rd_upconv2(x2_rd)

        x4_rd = self.rd_double_conv_block2(x3_rd)

        # Final classifier
        x5_rd = self.rd_final(x4_rd)
        return x5_rd

# ============================================================= ConvAttention ====================================================================
def to_3d(x):
    """Convert 4D tensor to 3D for processing"""
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """Convert 3D tensor back to 4D"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # Learnable temperature parameter for attention scaling
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # QKV projection
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # Depth-wise convolution for QKV processing
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias) 
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv_dw = self.qkv(x) # C x H x W => 3C x H x W
        
        qkv = self.qkv_dwconv(qkv_dw)  # 3C x H x W => 3C x H x W         by 1x1 dwconv
        q, k, v = qkv.chunk(3, dim=1)  # 3C x H x W => 3 x C x H x W 

        # Reshape for multi-head attention
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # n x C/n x H x W
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # n x C/n x H x W
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # n x C/n x H x W

        # Normalize queries and keys
        q = torch.nn.functional.normalize(q, dim=-1) # normalize
        k = torch.nn.functional.normalize(k, dim=-1) # normalize

        # Compute attention scores: QK^T / sqrt(d)
        attn = (q @ k.transpose(-2, -1)) * self.temperature # Aten = QK^T / d
        attn = attn.softmax(dim=-1)                         # aten = softmax(QK^T) 

        # Apply attention to values
        out = (attn @ v)                                    # output = softmax(QK^T)V

        # Reshape back to original dimensions
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) # n x C/n x H x W => # C x H x W

        # Final projection
        out = self.project_out(out) # 1 x 1 conv
        return out


class ConvAtten(nn.Module):
    """
    Channel-wise Self-Attention Module
    Based on the Transformer architecture from the provided code
    
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        ffn_expansion_factor (float): Expansion factor for feedforward network
        bias (bool): Whether to use bias in convolutions
        LayerNorm_type (str): Type of layer normalization ('BiasFree' or 'WithBias')
    """
    
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(ConvAtten, self).__init__()

        # First layer normalization
        self.norm1 = LayerNorm(dim, LayerNorm_type) 
        self.attn = Attention(dim, num_heads, bias)
        # Second layer normalization
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        """
        Forward pass of ConvAtten module
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor with same shape (B, C, H, W) as input
        """
        # Self-attention with residual connection
        temp_y = x + self.attn(self.norm1(x)) # x + Atten
        
        # Feed-forward with residual connection
        y = temp_y + self.ffn(self.norm2(temp_y)) # x + FFN

        return x + y


class ConvAttenStack(nn.Module):
    """
    Stack of ConvAtten modules
    Args:
        dim (int): Number of input channels
        num_blocks (int): Number of ConvAtten blocks to stack
        num_heads (int): Number of attention heads
        ffn_expansion_factor (float): Expansion factor for feedforward network
        bias (bool): Whether to use bias in convolutions
        LayerNorm_type (str): Type of layer normalization
    """
    
    def __init__(self, dim, num_blocks=4, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(ConvAttenStack, self).__init__()
        
        self.blocks = nn.Sequential(*[
            ConvAtten(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, 
                 bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        return self.blocks(x)

class FusionAttention(nn.Module):
    """
    Feature Fusion Attention Module    
    """
    def __init__(self, in_ch, num_blocks=2, num_heads = 8, ffn_expansion_factor=2.66, bias=False, layerNorm_type='WithBias'):
        super().__init__()
        self.ConvAtten = ConvAttenStack(dim=in_ch, num_blocks=num_blocks, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, 
                           bias=bias, LayerNorm_type=layerNorm_type)

    def forward(self, x):
        x = self.ConvAtten(x)
        return x

# ============================================================= Iterative Feature Branch ====================================================================
class IterFeatureBranch(nn.Module):
    """
    Multi-step ConvBottleneckLSTM module.

    Args:
        input_channels:   Number of channels of the input tensor.
        hidden_channels:  Number of channels of the hidden state.
        kernel_size:      Convolutional kernel size.
        num_layers:       Number of stacked ConvBottleneckLSTM layers.
        bias:             Whether to include bias in convolutions.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers=3):
        super(IterFeatureBranch, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        # Initial convolution to match hidden channels
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        # Stack of LSTM cells
        self.cells = nn.ModuleList([
            BottleneckLSTMCell(
                input_channels if i == 0 else hidden_channels,
                hidden_channels,
                kernel_size,
                'layer'
            ) for i in range(num_layers)
        ])

    def forward(self, x, h=None, c=None):
        """
        Args:
            x: Tensor of shape (seq_len, batch, channels, H, W).
            h: Optional list of initial hidden states. If None, zeros.
            c: Optional list of initial cell states. If None, zeros.

        Returns:
            output_seq: Tensor of shape (seq_len, batch, hidden_channels, H, W).
            (h_list, c_list): Lists of final hidden and cell states.
        """
        batch, _, H, W = x.size()
        # Initialize states if not provided
        if h is None:
            h = [torch.zeros(batch, self.hidden_channels, H, W, device=x.device)
                 for _ in range(self.num_layers)]
        if c is None:
            c = [torch.zeros(batch, self.hidden_channels, H, W, device=x.device)
                 for _ in range(self.num_layers)]
        
        input_x = self.conv(x)
        for i, cell in enumerate(self.cells):
            if i == 0 :
                h, c = cell(x, h, c)
            else:
                h, c = cell(input_x, h, c)
        return h, c

# ============================================================= STARNet Main Model ====================================================================
class STARNet(nn.Module):
    """
    Spatio-Temporal Attention Refinement Network : Baseline + ASPP + IterBranch
    Modified for input size: N x F x 500 x 512
    Multi-view and multi-frame implementation
    
    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for semantic segmentation task
    n_frames: int
        Total number of frames used as a sequence
    hidden_channels: int
        Number of hidden channels in LSTM modules
    num_layers: int
        Number of LSTM layers in iterative branch
    """

    def __init__(self, n_classes, n_frames, hidden_channels=128, num_layers=3):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # Backbone (encoding)
        self.rd_encoding_branch = EncodingBranch(self.n_frames)
        
        # Attention module for feature refinement
        self.Atten = FusionAttention(in_ch=128, num_blocks=3)
        
        # ASPP Blocks - Modified input size parameters
        self.rd_aspp_block = ASPPBlock(in_ch=128, out_ch=128, input_size=(125, 128))
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)
        
        # Latent space processing
        self.rd_single_conv_block2_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)
        
        # LSTM branch for temporal feature processing
        self.rd_LSTMBranch = IterFeatureBranch(input_channels=640, hidden_channels=hidden_channels, kernel_size=3, num_layers=num_layers)

        # Decoding branch for final output
        self.rd_decoding_branch = DecodingBranch(input_channels=128, n_classes=n_classes)


    def forward(self, x_rd):
        # Input: (N, F, 500, 512)
        # Backbone - Encoder
        rd_features, rd_latent = self.rd_encoding_branch(x_rd) # [4, 128, 125, 512], [4, 128, 125, 512]
        
        # Attention
        rd_features_att = self.Atten(rd_features)              # [4, 128, 125, 512]
        # Latent Space Attention
        rd_latent_att = self.Atten(rd_latent)                  # [4, 128, 125, 512]

        # ASPP blocks
        x1_rd = self.rd_aspp_block(rd_features_att)            # [4, 640, 125, 512]
        x2_rd = self.rd_single_conv_block1_1x1(rd_features_att) # [4, 128, 125, 512]

        # latent
        x3_rd = self.rd_single_conv_block2_1x1(rd_latent_att)  # [4, 128, 125, 512]

        # feature fusion
        x4_rd, _ = self.rd_LSTMBranch(x1_rd, x2_rd, x3_rd)     # [4, 128, 125, 512], [4, 128, 125, 512]

        # decoding - Decoder
        x5_rd = self.rd_decoding_branch(x4_rd)                 # [4, 3, 500, 512]

        return x5_rd


def profile_model(model, input_tensor):
    """Count parameters and FLOPs"""
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    print(f"\n🔹 Parameters (Params): {params / 1e6:.3f} M")
    print(f"🔹 FLOPs: {flops / 1e9:.3f} GFLOPs")

def test_starnet_module_speed(model, x_rd, ):
    import time
    print("="*60)
    print("STARNet Module Operation Time Test")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test device: {device}")

    # Model
    model.eval()

    print(f"Input shape: {x_rd.shape}")
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_rd)
    if device.type == 'cuda':
        torch.cuda.synchronize()


    # Module step-by-step timing
    timings = {}
    with torch.no_grad():
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_total = time.time()

        # Encoder
        t0 = time.time()
        rd_features, rd_latent = model.rd_encoding_branch(x_rd)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        timings['Encoder'] = time.time() - t0

        # Attention
        t0 = time.time()
        rd_features_att = model.Atten(rd_features)
        rd_latent_att = model.Atten(rd_latent)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        timings['Attention'] = time.time() - t0

        # ASPP + Conv
        t0 = time.time()
        x1_rd = model.rd_aspp_block(rd_features_att)
        x2_rd = model.rd_single_conv_block1_1x1(rd_features_att)
        x3_rd = model.rd_single_conv_block2_1x1(rd_latent_att)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        timings['ASPP+Conv'] = time.time() - t0

        # LSTM
        t0 = time.time()
        x4_rd, _ = model.rd_LSTMBranch(x1_rd, x2_rd, x3_rd)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        timings['LSTM'] = time.time() - t0

        # Decoder
        t0 = time.time()
        x5_rd = model.rd_decoding_branch(x4_rd)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        timings['Decoder'] = time.time() - t0

        end_total = time.time()
        timings['Total'] = end_total - start_total

    # Output results
    for k, v in timings.items():
        print(f"{k: <15}: {v:.4f} s")

    print("✅ STARNet module operation time test completed!")


# =========================
# Main function
# =========================
if __name__ == "__main__":
    # ====== Initialize model ======
    from starnet import STARNet  # ⚠️ Change to your own file path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STARNet(n_classes=3, n_frames=5).to(device)

    # ====== Construct input ======
    x = torch.randn(1, 5, 500, 512).to(device)

    # ====== Parameters and FLOPs ======
    profile_model(model, x)

    # ====== Inference time test ======
    test_starnet_module_speed(model, x)
