import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from attrdict import AttrDict

from dataset import START, PAD

from networks.TUBE import TUBEPosBias
from networks.stn import FlexibleSTN
from networks.CSTR_Module import CBAM, SAM, SADM_A
from torchvision.transforms.functional import rotate

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BottleneckBlock(nn.Module): # 1x1로 channel 작게 -> 3x3으로 원하는 채널 생성 + 시야각 늘리기
    """
    Dense Bottleneck Block

    It contains two convolutional layers, a 1x1 and a 3x3.
    """

    def __init__(self, input_size, growth_rate, dropout_rate=0.2, num_bn=3):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added. That is the ouput
                size of the last convolutional layer.
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(BottleneckBlock, self).__init__()
        inter_size = num_bn * growth_rate
        self.norm1 = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            input_size, inter_size, kernel_size=1, stride=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(inter_size)
        self.conv2 = nn.Conv2d(
            inter_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    """
    Transition Block

    A transition layer reduces the number of feature maps in-between two bottleneck
    blocks.
    """

    def __init__(self, input_size, output_size):
        """
        Args:
            input_size (int): Number of channels of the input
            output_size (int): Number of channels of the output
        """
        super(TransitionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            input_size, output_size, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        return self.pool(out)


class DenseBlock(nn.Module): 
    """
    Dense block

    A dense block stacks several bottleneck blocks.
    """

    def __init__(self, input_size, growth_rate, depth, dropout_rate=0.2):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added per bottleneck block
            depth (int): Number of bottleneck blocks
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(DenseBlock, self).__init__()
        layers = [
            BottleneckBlock(
                input_size + i * growth_rate, growth_rate, dropout_rate=dropout_rate
            )
            for i in range(depth)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DeepCNN300(nn.Module):
    """
    This is specialized to the math formula recognition task
    - three convolutional layers to reduce the visual feature map size and to capture low-level visual features
    (128, 256) -> (8, 32) -> total 256 features
    - transformer layers cannot change the channel size, so it requires a wide feature dimension
    ***** this might be a point to be improved !!
    """

    def __init__(
        self, input_channel, num_in_features, output_channel=256, dropout_rate=0.2, depth=16, growth_rate=24
    ):
        super(DeepCNN300, self).__init__()
        self.conv0 = nn.Conv2d( # 무조건 1/2
            input_channel,  # 3
            num_in_features,  # 48
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm0 = nn.BatchNorm2d(num_in_features)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # 1/4 (128, 128) -> (32, 32)
        num_features = num_in_features

        self.block1 = DenseBlock(
            num_features,  # 48
            growth_rate=growth_rate,  # 48 + growth_rate(24)*depth(16) -> 432
            depth=depth,  # 16?
            dropout_rate=0.2,
        )
        num_features = num_features + depth * growth_rate
        self.trans1 = TransitionBlock(num_features, num_features // 2)  # 16 x 16
        num_features = num_features // 2
        self.block2 = DenseBlock(
            num_features,  # 128
            growth_rate=growth_rate,  # 16
            depth=depth,  # 8
            dropout_rate=0.2,
        )
        num_features = num_features + depth * growth_rate
        self.trans2_norm = nn.BatchNorm2d(num_features)
        self.trans2_relu = nn.ReLU(inplace=True)
        self.trans2_conv = nn.Conv2d(
            num_features, num_features // 2, kernel_size=1, stride=1, bias=False  # 128
        )

    def forward(self, input):
        # NF = num_features
        # D = depth
        # GR = growth_rate
        # input [B, C, H, W] = [B, 1, 128, 128]
        
        out = self.conv0(input)  # [B, NF, (H올림)//2, (W올림)//2] = [B, 48, 64, 64]
        
        out = self.relu(self.norm0(out))
        # [B, NF, ((=내림)]
        out = self.max_pool(out) # [B, NF, H//4, W//4] = [B, 48, 32, 32]
        
        out = self.block1(out) # [B, NF+D*GR, H//4, W//4] = [B, 432, 32, 32]
        out = self.trans1(out) # [B, (NF+D*GR)//2, H//8, W//8] = [B, 216, 16, 16]
        # [B, NF, ((=내림))]
        
        out = self.block2(out) # [B, (NF+D*GR)//2+D*GR, H//8, W//8] = [B, 600, 16, 16]
        out_before_trans2 = self.trans2_relu(self.trans2_norm(out))
        out_A = self.trans2_conv(out_before_trans2)  
        # [B, ((NF+D*GR)//2+D*GR)//2, H//8, W//8] = [B, 300, 16, 16]
        
        return out_A  # 128 x (16x16)


class CustomDeepCNN300(nn.Module):
    def __init__(
        self, input_channel, num_in_features, output_channel=256, dropout_rate=0.2, depth=16, growth_rate=24
    ):
        super(CustomDeepCNN300, self).__init__()
        self.conv0 = nn.Conv2d( # 무조건 1/2
            input_channel,  # 3
            num_in_features,  # 48
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm0 = nn.BatchNorm2d(num_in_features)
        self.relu = nn.ReLU(inplace=True)

        self.cbam0 = CBAM(num_in_features, num_in_features//2)
        self.sam0 = SAM()

        self.pooler0 = SADM_A(num_in_features, num_in_features)
        num_features = num_in_features

        self.block1 = DenseBlock(
            num_features,  # 48
            growth_rate=growth_rate,  # 48 + growth_rate(24)*depth(16) -> 432
            depth=depth,  # 16?
            dropout_rate=0.2,
        )
        num_features1 = num_features + depth * growth_rate

        self.cbam1 = CBAM(num_features1, num_features1//2)
        self.sam1 = SAM()
        
        self.pooler1 = SADM_A(num_features1, num_features1 // 2)
        # self.trans1 = TransitionBlock(num_features, num_features // 2)  # 16 x 16
        num_features2 = num_features1 // 2
        self.block2 = DenseBlock(
            num_features2,  # 128
            growth_rate=growth_rate,  # 16
            depth=depth,  # 8
            dropout_rate=0.2,
        )
        num_features = num_features2 + depth * growth_rate
        self.trans2_norm = nn.BatchNorm2d(num_features)
        self.trans2_relu = nn.ReLU(inplace=True)
        self.trans2_conv = nn.Conv2d(
            num_features, num_features // 2, kernel_size=1, stride=1, bias=False  # 128
        )
        num_features = num_features // 2
        self.cbam2 = CBAM(num_features, num_features//2)
        self.sam2 = SAM()


    def forward(self, input):
        # NF = num_features
        # D = depth
        # GR = growth_rate
        # input [B, C, H, W] = [B, 1, 128, 128]
        
        out = self.conv0(input)  # [B, NF, (H올림)//2, (W올림)//2] = [B, 48, 64, 64]
        out = self.relu(self.norm0(out))
        tmp = self.cbam0(out)
        tmp = self.sam0(tmp)

        out = out + tmp
        # [B, NF, ((=내림)]

        out = self.pooler0(out)
        
        out = self.block1(out) # [B, NF+D*GR, H//4, W//4] = [B, 432, 32, 32]
        tmp = self.cbam1(out)
        tmp = self.sam1(tmp)
        out = out + tmp
        out = self.pooler1(out) # [B, (NF+D*GR)//2, H//8, W//8] = [B, 216, 16, 16]
        # [B, NF, ((=내림))]
        
        out = self.block2(out) # [B, (NF+D*GR)//2+D*GR, H//8, W//8] = [B, 600, 16, 16]
        out_before_trans2 = self.trans2_relu(self.trans2_norm(out))

        out_A = self.trans2_conv(out_before_trans2)  
        # [B, ((NF+D*GR)//2+D*GR)//2, H//8, W//8] = [B, 300, 16, 16]
        tmp = self.cbam2(out_A)
        tmp = self.sam2(tmp)
        out_A = out_A + tmp
        
        return out_A  # 128 x (16x16)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None, attn_bias=None):
        # B, HEAD, Q_LEN, K_LEN
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        if attn_bias is not None:
            attn += attn_bias

        if mask is not None:
            attn = attn.masked_fill(mask=mask, value=float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, q_channels, k_channels, head_num=8, dropout=0.1, use_tube=False):
        super(MultiHeadAttention, self).__init__()

        self.q_channels = q_channels
        self.k_channels = k_channels
        self.head_dim = q_channels // head_num
        self.head_num = head_num

        self.q_linear = nn.Linear(q_channels, self.head_num * self.head_dim)
        self.k_linear = nn.Linear(k_channels, self.head_num * self.head_dim)
        self.v_linear = nn.Linear(k_channels, self.head_num * self.head_dim)

        if use_tube:
            temperature = 2 * (self.head_num * self.head_dim) ** 0.5
        else:
            temperature = (self.head_num * self.head_dim) ** 0.5

        self.attention = ScaledDotProductAttention(
            temperature=temperature, dropout=dropout
        )
        self.out_linear = nn.Linear(self.head_num * self.head_dim, q_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None, attn_bias=None):
        b, q_len, k_len, v_len = q.size(0), q.size(1), k.size(1), v.size(1)
        q = (
            self.q_linear(q)
            .view(b, q_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(k)
            .view(b, k_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(v)
            .view(b, v_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )

        if mask is not None:
            mask = mask.unsqueeze(1)

        out, attn = self.attention(q, k, v, mask=mask, attn_bias=attn_bias)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(b, q_len, self.head_num * self.head_dim)
        )
        out = self.out_linear(out)
        out = self.dropout(out)

        return out, attn


class Feedforward(nn.Module):
    def __init__(self, filter_size=2048, hidden_dim=512, dropout=0.1):
        super(Feedforward, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, filter_size, True),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(filter_size, hidden_dim, True),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        )

    def forward(self, input, h=0, w=0):
        return self.layers(input)


class LocalityAwareFeedforward(nn.Module):
    def __init__(self, filter_size=2048, hidden_dim=512, dropout=0.1):
        super(LocalityAwareFeedforward, self).__init__()

        self.layers = nn.Sequential( # [b, hidden_dim, h, w]
            nn.Conv2d(hidden_dim, filter_size, 1, bias=False), # [b, filter, h, w]
            nn.BatchNorm2d(filter_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Conv2d(filter_size, filter_size, 3, stride=1, padding=1, bias=False), # [b, filter, h, w]
            nn.BatchNorm2d(filter_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Conv2d(filter_size, hidden_dim, 1, bias=False), # [b, hidden_dim, h, w]
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        )

    def forward(self, input, h=0, w=0):
        # [b, h * w, c]
        b, _, c = input.shape
        input = input.transpose(-1, -2).reshape(b, c, h, w) # [b, c, h, w]

        out = self.layers(input) # [b, c, h, w]

        out = out.reshape(b, c, h*w).transpose(-1, -2)
        return out # [b, h * w, c]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_size, filter_size, head_num, dropout_rate=0.2,
        locality_aware_feedforward=False, use_tube=False):
        super(TransformerEncoderLayer, self).__init__()

        self.attention_layer = MultiHeadAttention(
            q_channels=input_size,
            k_channels=input_size,
            head_num=head_num,
            dropout=dropout_rate,
            use_tube=use_tube,
        )
        self.attention_norm = nn.LayerNorm(normalized_shape=input_size)

        self.locality_aware_feedforward = locality_aware_feedforward
        if self.locality_aware_feedforward:
            self.feedforward_layer = LocalityAwareFeedforward(
                filter_size=filter_size, hidden_dim=input_size
            )
        else:
            self.feedforward_layer = Feedforward(
                filter_size=filter_size, hidden_dim=input_size
            )
        self.feedforward_norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, input, h, w, attn_bias=None):
        out, _ = self.attention_layer(input, input, input,
                attn_bias=attn_bias)
        out = self.attention_norm(out + input)

        ff = self.feedforward_layer(out, h, w)
        out = self.feedforward_norm(ff + out)
        return out


class PositionalEncoding2D(nn.Module):
    def __init__(self, in_channels, device, max_h=512, max_w=512, dropout=0.1,
        use_adaptive_2d_encoding=False):
        super(PositionalEncoding2D, self).__init__()

        self.h_position_encoder = self.generate_encoder(in_channels // 2, max_h)
        self.w_position_encoder = self.generate_encoder(in_channels // 2, max_w)

        if use_adaptive_2d_encoding:
            self.alpha_layer = nn.Sequential(
                nn.Linear(in_channels, in_channels//4),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(in_channels//4, in_channels),
                nn.Sigmoid(),
            )
        else:
            self.h_linear = nn.Linear(in_channels // 2, in_channels // 2)
            self.w_linear = nn.Linear(in_channels // 2, in_channels // 2)

        self.D = in_channels // 2
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.use_adaptive_2d_encoding = use_adaptive_2d_encoding

    def generate_encoder(self, in_channels, max_len):
        pos = torch.arange(max_len).float().unsqueeze(1)
        i = torch.arange(in_channels).float().unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_channels)
        position_encoder = pos * angle_rates
        position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
        position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])
        return position_encoder  # (Max_len, In_channel)

    def forward(self, input, method='add'):
        # method: add(input+encode), plain(encode)

        ### Require DEBUG
        b, c, h, w = input.size()
        h_pos_encoding = (
            self.h_position_encoder[:h, :].unsqueeze(1).to(self.device)
        ) # H 1 D

        w_pos_encoding = (
            self.w_position_encoder[:w, :].unsqueeze(0).to(self.device)
        )

        if self.use_adaptive_2d_encoding:
            ge = torch.mean(input, dim=(-1, -2)) # [B, 2*D]

            alpha_together = self.alpha_layer(ge) # [B, 2*D]
            alpha = alpha_together[:, :self.D] # [B, D]
            beta = alpha_together[:, self.D:] # [B, D]

            alpha = alpha.view(b, 1, 1, -1) # [B, 1, 1, D]
            beta = beta.view(b, 1, 1, -1) # [B, 1, 1, D]
            h_pos_encoding = alpha * h_pos_encoding.unsqueeze(0)  # [B, H, 1, D]
            w_pos_encoding = beta * w_pos_encoding.unsqueeze(0)  # [B, 1, W, D]

            h_pos_encoding = h_pos_encoding.expand(-1, -1, w, -1)   # [B, H, W, D]
            w_pos_encoding = w_pos_encoding.expand(-1, h, -1, -1)   # [B, H, W, D]

            pos_encoding = torch.cat([h_pos_encoding, w_pos_encoding], dim=-1)  # [B, H, W, 2*D]

            pos_encoding = pos_encoding.permute(0, 3, 1, 2)  # [B, 2*D, H, W]
        else:
            h_pos_encoding = self.h_linear(h_pos_encoding)  # [H, 1, D]
            w_pos_encoding = self.w_linear(w_pos_encoding)  # [1, W, D]

            h_pos_encoding = h_pos_encoding.expand(-1, w, -1)   # h, w, c/2
            w_pos_encoding = w_pos_encoding.expand(h, -1, -1)   # h, w, c/2

            pos_encoding = torch.cat([h_pos_encoding, w_pos_encoding], dim=2)  # [H, W, 2*D]

            pos_encoding = pos_encoding.permute(2, 0, 1)  # [2*D, H, W]
            pos_encoding = pos_encoding.unsqueeze(0) # [1, 2*D, H, W]
            pos_encoding = pos_encoding.expand(b, -1, -1, -1)

        if method == 'add':
            out = input + pos_encoding # [B, 2*D, H, W]
        elif method == 'plain':
            out = pos_encoding
        else:
            raise NotImplementedError(f'method {method}')

        out = self.dropout(out)

        return out


class TransformerEncoderFor2DFeatures(nn.Module):
    """
    Transformer Encoder for Image
    1) ShallowCNN : low-level visual feature identification and dimension reduction
    2) Positional Encoding : adding positional information to the visual features
    3) Transformer Encoders : self-attention layers for the 2D feature maps
    """

    def __init__(
        self,
        input_size,
        hidden_dim,
        filter_size,
        head_num,
        layer_num,
        device,
        dropout_rate=0.1,
        checkpoint=None,
        use_adaptive_2d_encoding=False,
        locality_aware_feedforward=False,
        use_tube=False,
        use_cstr_module=False,
    ):
        super(TransformerEncoderFor2DFeatures, self).__init__()

        self.use_tube = use_tube

        if use_cstr_module:
            self.shallow_cnn = CustomDeepCNN300(
                input_size,
                num_in_features=48,
                output_channel=hidden_dim,
                dropout_rate=dropout_rate,
            )
        else:
            self.shallow_cnn = DeepCNN300(
                input_size,
                num_in_features=48,
                output_channel=hidden_dim,
                dropout_rate=dropout_rate,
            )
        self.positional_encoding = PositionalEncoding2D(hidden_dim, device=device,
            use_adaptive_2d_encoding=use_adaptive_2d_encoding)

        self.attention_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_dim, filter_size, head_num, dropout_rate,
                    locality_aware_feedforward, self.use_tube)
                for _ in range(layer_num)
            ]
        )
        if self.use_tube:
            self.pos_bias = TUBEPosBias(hidden_dim, hidden_dim, head_num, dropout_rate)

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def forward(self, input):
        # input [B, C, H, W] = [B, 1, 128, 128]
        out = self.shallow_cnn(input)  
        b, c, h, w = out.size()
        # [B, ((NF+D*GR)//2+D*GR)//2, H//8, W//8] = [B, 300, 16, 16]
        # NF//4 + 3*D*GR//4 = SATRN.encoder.hidden_dim

        if self.use_tube:
            enc = self.positional_encoding(out, method='plain')  # [b, c, h, w]
            enc = enc.view(b, c, h * w).transpose(1, 2)
            attn_bias = self.pos_bias(enc, enc)
        else:
            out = self.positional_encoding(out)  # [b, c, h, w]
            attn_bias = None

            # flatten
        out = out.view(b, c, h * w).transpose(1, 2)  # [b, h x w, c]

        for layer in self.attention_layers:
            out = layer(out, h, w, attn_bias=attn_bias)

        result = AttrDict(
            out=out,
            h=h,
            w=w,
        )
        
        return result

class RotationApplier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_dim,
        device,
        dropout_rate=0.1,
        checkpoint=None,
    ):
        super().__init__()

        self.shallow_cnn = DeepCNN300(
            input_size,
            num_in_features=48,
            output_channel=hidden_dim,
            dropout_rate=dropout_rate,
        )

        self.middle = nn.Sequential(
            nn.Conv2d(hidden_dim, 16, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.AdaptiveAvgPool2d((4, 4)), # B, 16, 4, 4
            nn.Flatten(), # B, 16 * 4 * 4 = 256
            nn.Linear(256, 8),
        )

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def forward(self, input):
        # input [B, C, H, W] = [B, 1, 128, 128]
        out = self.shallow_cnn(input)
        b, c, h, w = out.shape
        out = self.middle(out) # [B, 4]
        
        which = torch.argmax(out, dim=-1) # [B]

        # theta = theta.view(-1, 2, 3)

        if which % 4 == 1:
            new_input = input.transpose(-1, -2).flip(-1)
        elif which % 4 == 2:
            new_input = input.flip(dims=(-1, -2))
        elif which % 4 == 3:
            new_input = input.transpose(-1, -2).flip(-2)
        else:
            new_input = input
            
        if which // 4 == 1:
            new_input = new_input.flip(-1)
        
        return new_input, which


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_size, src_size, filter_size, head_num, 
        dropout_rate=0.2, use_tube=False):
        super(TransformerDecoderLayer, self).__init__()

        self.use_tube = use_tube
        self.self_attention_layer = MultiHeadAttention(
            q_channels=input_size,
            k_channels=input_size,
            head_num=head_num,
            dropout=dropout_rate,
            use_tube=self.use_tube,
        )
        self.self_attention_norm = nn.LayerNorm(normalized_shape=input_size)

        self.attention_layer = MultiHeadAttention(
            q_channels=input_size,
            k_channels=src_size,
            head_num=head_num,
            dropout=dropout_rate,
            use_tube=self.use_tube,
        )
        self.attention_norm = nn.LayerNorm(normalized_shape=input_size)

        self.feedforward_layer = Feedforward(
            filter_size=filter_size, hidden_dim=input_size
        )
        self.feedforward_norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, tgt, tgt_prev, src, tgt_mask,
         get_attn=False, attn_bias=None, attn_2d_bias=None):
        if tgt_prev is None:  # Train
            att, attn_1 = self.self_attention_layer(tgt, tgt, tgt, tgt_mask, attn_bias=attn_bias)

            out = self.self_attention_norm(att + tgt)

            att, attn_2 = self.attention_layer(tgt, src, src, attn_bias=attn_2d_bias)

            out = self.attention_norm(att + out)

            ff = self.feedforward_layer(out)
            out = self.feedforward_norm(ff + out)

            result = AttrDict(
                out=out,
            )

            if get_attn:
                result['attn_1'] = attn_1
                result['attn_2'] = attn_2
 
            return result
        else:
            
            # tgt, tgt_prev, src, tgt_mask
            # tgt [B, 1, D]
            # tgt_prev [B, t, D]
            # src [B, H*W, C]
            # tgt_mask [1, t+1] (t 현재 생성 index)

            tgt_prev = torch.cat([tgt_prev, tgt], 1) # [B, t+1, D]
            att, attn_1 = self.self_attention_layer(tgt, tgt_prev, tgt_prev, tgt_mask, 
                attn_bias=attn_bias) # [B, 1, D]

            out = self.self_attention_norm(att + tgt)

            att, attn_2 = self.attention_layer(tgt, src, src, attn_bias=attn_2d_bias) # [B, 1, D]

            out = self.attention_norm(att + out)

            ff = self.feedforward_layer(out)
            out = self.feedforward_norm(ff + out)

            result = AttrDict(
                out=out,
            )

            if get_attn:
                result['attn_1'] = attn_1
                result['attn_2'] = attn_2
 
            return result


class PositionEncoder1D(nn.Module):
    def __init__(self, in_channels, device, max_len=500, dropout=0.1):
        super(PositionEncoder1D, self).__init__()

        self.position_encoder = self.generate_encoder(in_channels, max_len)
        self.position_encoder = self.position_encoder.unsqueeze(0)
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

    def generate_encoder(self, in_channels, max_len):
        pos = torch.arange(max_len).float().unsqueeze(1)

        i = torch.arange(in_channels).float().unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_channels)

        position_encoder = pos * angle_rates
        position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
        position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])

        return position_encoder

    def forward(self, x, point=-1, method='add'):
    # method: add(input+encode), plain(encode)):
        if point == -1:
            emb = self.position_encoder[:, : x.size(1), :].to(self.device)
        else:
            emb = self.position_encoder[:, point, :].unsqueeze(1).to(self.device)
        
        emb = emb.expand(x.size(0), -1, -1)
        if method == 'add':
            out = x + emb 
        elif method == 'plain':
            out = emb
        else:
            raise NotImplementedError(f'method {method}')

        out = self.dropout(out)

        return out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_classes,
        src_dim,
        hidden_dim,
        filter_dim,
        head_num,
        dropout_rate,
        pad_id,
        st_id,
        device,
        layer_num=1,
        checkpoint=None,
        use_tube=False
    ):
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(num_classes + 1, hidden_dim)
        self.hidden_dim = hidden_dim
        self.filter_dim = filter_dim
        self.num_classes = num_classes
        self.layer_num = layer_num
        self.use_tube = use_tube
        self.head_num = head_num

        self.pos_encoder = PositionEncoder1D(
            in_channels=hidden_dim, dropout=dropout_rate, device=device
        )

        self.attention_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    hidden_dim, src_dim, filter_dim, head_num, dropout_rate, use_tube=self.use_tube
                )
                for _ in range(layer_num)
            ]
        )
        self.generator = nn.Linear(hidden_dim, num_classes)

        self.pad_id = pad_id
        self.st_id = st_id
        self.device = device

        if self.use_tube:
            self.pos_attn_layer = TUBEPosBias(hidden_dim, hidden_dim, head_num=head_num, dropout=dropout_rate)
            self.pos_2d_attn_layer = TUBEPosBias(hidden_dim, src_dim, head_num=head_num, dropout=dropout_rate)

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def pad_mask(self, text):
        pad_mask = text == self.pad_id
        pad_mask[:, 0] = False
        pad_mask = pad_mask.unsqueeze(1)

        return pad_mask

    def order_mask(self, length):
        order_mask = torch.triu(torch.ones(length, length), diagonal=1).bool()
        order_mask = order_mask.unsqueeze(0).to(self.device)
        return order_mask

    def text_embedding(self, texts):
        tgt = self.embedding(texts)
        tgt *= math.sqrt(tgt.size(2))

        return tgt

    def forward(
        self, src, text, is_train=True, batch_max_length=50, teacher_forcing_ratio=1.0,
        return_attn=False, enc_2d=None,
    ): 
        # src [B, H*W, C]
        # enc_2d [B, H*W, C]

        # text [B, S] (S = not including [EOS])

        if is_train and random.random() < teacher_forcing_ratio:
            tgt = self.text_embedding(text) # [B, S, D]

            if self.use_tube:
                pos_enc = self.pos_encoder(tgt, method='plain') # [B, S, D]
                attn_bias = self.pos_attn_layer(pos_enc, pos_enc) # [B, HEAD_NUM, S, S]
                attn_2d_bias = self.pos_2d_attn_layer(pos_enc, enc_2d) # [B, HEAD_NUM, S, H*W]
            else:
                tgt = self.pos_encoder(tgt, method='add') # [B, S, D]
                attn_bias = None
                attn_2d_bias = None
            
            # [B, 1, S] | [1, S, S] = [B, S, S]
            tgt_mask = self.pad_mask(text) | self.order_mask(text.size(1))

            for layer in self.attention_layers:
                layer_output_dict = layer(tgt, None, src, tgt_mask, return_attn,
                    attn_bias=attn_bias, attn_2d_bias=attn_2d_bias)
                tgt = layer_output_dict['out'] # [B, S, D]
                if return_attn:
                    attns_1 = layer_output_dict['attn_1']
                    attns_2 = layer_output_dict['attn_2']
            out = self.generator(tgt)
        else:
            out = []
            num_steps = batch_max_length - 1

            # target [B]
            b = src.size(0)
            target = torch.LongTensor(b).view(-1).fill_(self.st_id).to(self.device) # [START] token
            features = [None] * self.layer_num

            if return_attn:
                attns_1 = []
                attns_2 = []

            if self.use_tube:
                pos_enc_cache = None
                # attn_cache = torch.zeros(b, self.head_num, num_steps, num_steps).to(self.device)
                # attn_2d_cache = torch.zeros(b, self.head_num, num_steps, src.size(1)).to(self.device)
            for t in range(num_steps):
                target = target.unsqueeze(1) # [B, 1]
                tgt = self.text_embedding(target) # [B, 1, D]

                if self.use_tube:
                    pos_enc = self.pos_encoder(tgt, method='plain') # [B, 1, D]
                    pos_enc_cache = ( 
                        pos_enc if pos_enc_cache is None else torch.cat([pos_enc_cache, pos_enc], 1)
                    )
                    
                    attn_bias = self.pos_attn_layer(pos_enc, pos_enc_cache[:, :t+1]) # [B, HEAD_NUM, 1, t+1]
                    # attn_cache[:, :, t, :t+1] = attn_bias.squeeze(2)
                    # cur_attn_bias = attn_cache[:, :, :t+1, :t+1]
                
                    attn_2d_bias = self.pos_2d_attn_layer(pos_enc, enc_2d)
                    # attn_2d_cache[:, :, t] = .squeeze(2) # [B, HEAD_NUM, 1, H*W]
                    # cur_attn_2d_bias = attn_2d_cache[:, :, :t+1]
                else:
                    tgt = self.pos_encoder(tgt, method='add') # [B, S, D]
                    attn_bias = None
                    attn_2d_bias = None

                tgt_mask = self.order_mask(t + 1) # [1, t+1, t+1]
                tgt_mask = tgt_mask[:, -1].unsqueeze(1)  # [1, t+1]

                for l, layer in enumerate(self.attention_layers):
                    layer_output_dict = layer(tgt, features[l], src, tgt_mask, return_attn,
                        attn_bias=attn_bias, attn_2d_bias=attn_2d_bias)
                    tgt = layer_output_dict['out'] # [B, 1, D]
                    if return_attn:
                        attn_1 = layer_output_dict['attn_1']
                        attn_2 = layer_output_dict['attn_2']

                    # features[l] [B, t+1, D]
                    features[l] = ( 
                        tgt if features[l] is None else torch.cat([features[l], tgt], 1)
                    )

                    if return_attn:
                        attns_1.append(attn_1.cpu().data.numpy())
                        attns_2.append(attn_2.cpu().data.numpy())

                _out = self.generator(tgt)  # [b, 1, c]
                target = torch.argmax(_out[:, -1:, :], dim=-1)  # [b, 1]
                target = target.squeeze(-1)   # [b]
                out.append(_out)
            
            out = torch.stack(out, dim=1).to(self.device)    # [b, max length, 1, class length]
            out = out.squeeze(2)    # [b, max length, class length]

        result = AttrDict(
            out=out
        )
        if return_attn:
            result['attns_1'] = attns_1
            result['attns_2'] = attns_2
        return result

class SATRN(nn.Module):
    def __init__(self, FLAGS, train_dataset, device, checkpoint=None):
        super(SATRN, self).__init__()

        if not hasattr(FLAGS.SATRN, 'use_adaptive_2d_encoding'):
            use_adaptive_2d_encoding = False
        else:
            use_adaptive_2d_encoding = FLAGS.SATRN.use_adaptive_2d_encoding

        if not hasattr(FLAGS.SATRN, 'locality_aware_feedforward'):
            locality_aware_feedforward = False
        else:
            locality_aware_feedforward = FLAGS.SATRN.locality_aware_feedforward

        if not hasattr(FLAGS.SATRN, 'solve_extra_pb'):
            self.solve_extra_pb = False
        else:
            self.solve_extra_pb = FLAGS.SATRN.solve_extra_pb

        if not hasattr(FLAGS.SATRN, 'use_tube'):
            self.use_tube = False
        else:
            self.use_tube = FLAGS.SATRN.use_tube

        if not hasattr(FLAGS.SATRN, 'use_cstr_module'):
            self.use_cstr_module = False
        else:
            self.use_cstr_module = FLAGS.SATRN.use_cstr_module

        self.use_flexible_stn = FLAGS.SATRN.flexible_stn.use
        if FLAGS.SATRN.flexible_stn.use and \
            not FLAGS.SATRN.flexible_stn.train_stn_only:
            self.stn = RotationApplier(
                input_size=FLAGS.data.rgb,
                hidden_dim=FLAGS.SATRN.encoder.hidden_dim,
                dropout_rate=FLAGS.dropout_rate,
                device=device,
            )

        self.encoder = TransformerEncoderFor2DFeatures(
            input_size=FLAGS.data.rgb,
            hidden_dim=FLAGS.SATRN.encoder.hidden_dim,
            filter_size=FLAGS.SATRN.encoder.filter_dim,
            head_num=FLAGS.SATRN.encoder.head_num,
            layer_num=FLAGS.SATRN.encoder.layer_num,
            dropout_rate=FLAGS.dropout_rate,
            device=device,
            use_adaptive_2d_encoding=use_adaptive_2d_encoding,
            locality_aware_feedforward=locality_aware_feedforward,
            use_tube=self.use_tube,
            use_cstr_module=self.use_cstr_module,
        )

        self.decoder = TransformerDecoder(
            num_classes=len(train_dataset.id_to_token),
            src_dim=FLAGS.SATRN.decoder.src_dim,
            hidden_dim=FLAGS.SATRN.decoder.hidden_dim,
            filter_dim=FLAGS.SATRN.decoder.filter_dim,
            head_num=FLAGS.SATRN.decoder.head_num,
            dropout_rate=FLAGS.dropout_rate,
            pad_id=train_dataset.token_to_id[PAD],
            st_id=train_dataset.token_to_id[START],
            layer_num=FLAGS.SATRN.decoder.layer_num,
            device=device,
            use_tube=self.use_tube,
        )

        if self.solve_extra_pb:
            self.level_classifer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(FLAGS.SATRN.encoder.hidden_dim, 5)
            )

            self.source_classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(FLAGS.SATRN.encoder.hidden_dim, 2)
            )

            self.criterion = (
                nn.CrossEntropyLoss(ignore_index=train_dataset.token_to_id[PAD]),
                nn.CrossEntropyLoss(ignore_index=train_dataset.token_to_id[PAD]),
                nn.CrossEntropyLoss(ignore_index=train_dataset.token_to_id[PAD]),
            ) 
        else:
            self.criterion = (
                nn.CrossEntropyLoss(ignore_index=train_dataset.token_to_id[PAD])
            )  # without ignore_index=train_dataset.token_to_id[PAD]

        if checkpoint:
            self.load_state_dict(checkpoint)

        if FLAGS.SATRN.flexible_stn.use and \
            FLAGS.SATRN.flexible_stn.train_stn_only:
            self.stn = RotationApplier(
                input_size=FLAGS.data.rgb,
                hidden_dim=FLAGS.SATRN.encoder.hidden_dim,
                dropout_rate=FLAGS.dropout_rate,
                device=device,
            )

    def forward(self, input, expected, is_train, teacher_forcing_ratio,
             return_attn=False, return_stn=False):
        # input [B, C, H, W] = [B, 1, 128, 128]
        result = AttrDict()

        if self.use_flexible_stn:
            input, which = self.stn(input)
            if return_stn:
                result['stn'] = input.cpu().numpy()
                result['which'] = which

            
        enc_result_dict = self.encoder(input) # [B, H*W, C] = [B, 16*16, 300]
        enc_result = enc_result_dict['out']
        h = enc_result_dict['h']
        w = enc_result_dict['w']
        b = input.size(0)
        if self.use_tube:
            pos_2d = enc_result.reshape(b, h, w, -1).permute(0, 3, 1, 2) # [B, C, H, W]
            enc_2d = self.encoder.positional_encoding(pos_2d, method='plain')  # [B, C, H, W]

            enc_result = pos_2d.permute(0, 2, 3, 1).reshape(b, h*w, -1) # [B, H*W, C]
            enc_2d = enc_2d.permute(0, 2, 3, 1).reshape(b, h*w, -1) # [B, H*W, C]
        else:
            enc_2d = None

        dec_result_dict = self.decoder(
            enc_result,
            expected[:, :-1],
            is_train,
            expected.size(1),
            teacher_forcing_ratio,
            return_attn=return_attn,
            enc_2d=enc_2d,
        )

        dec_result = dec_result_dict['out']
        if return_attn:
            attns_1 = dec_result_dict['attns_1']
            attns_2 = dec_result_dict['attns_2']

        result['out'] =dec_result

        if self.solve_extra_pb:
            enc_mean = torch.mean(enc_result, dim=1) # [B, 300]
            level_result = self.level_classifer(enc_mean) # [B, 5]
            source_result = self.source_classifier(enc_mean) # [B, 5]

            result['level_out'] = level_result
            result['source_out'] = source_result

        if return_attn:
            result['enc_result'] = enc_result.reshape(b, h, w, -1).cpu().data.numpy()
            result['attns_1'] = attns_1
            result['attns_2'] = attns_2
        
        return result