
import random
import math
import torch
import torch.nn as nn
from attrdict import AttrDict
from networks.SATRN import (
    DenseBlock,
    Feedforward,
    TransitionBlock,
    TransformerEncoderLayer,
    PositionalEncoding2D,
    PositionEncoder1D,
    MultiHeadAttention,
    PAD, 
    START,
)


class SmallerDeepCNN300(nn.Module):
    """"
    8X8로 변환
    """

    def __init__(
        self, input_channel, num_in_features, output_channel=256, dropout_rate=0.2, depth=16, growth_rate=24,
            use_256_input=True,
    ):
        super(SmallerDeepCNN300, self).__init__()
        self.conv0 = nn.Conv2d( # 무조건 1/2
            input_channel,  # 3
            num_in_features,  # 48
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.use_256_input = use_256_input
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

        if self.use_256_input:
            print('use_256_input:', self.use_256_input)
            self.trans2_norm = nn.BatchNorm2d(num_features)
            self.trans2_relu = nn.ReLU(inplace=True)
            self.trans2_conv = nn.Conv2d(
                num_features, num_features // 2, kernel_size=1, stride=1, bias=False  # 128
            )
        else:
            self.trans2 = TransitionBlock(num_features, num_features // 2)  # 16 x 16
            num_features = num_features // 2
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

        if self.use_256_input:
            out_before_trans2 = self.trans2_relu(self.trans2_norm(out))
            out_A = self.trans2_conv(out_before_trans2)  
        else:
            out = self.trans2(out) # [B, 300, 8, 8]

            out_A = self.trans2_relu(self.trans2_norm(out))
        # [B, ((NF+D*GR)//2+D*GR)//2, H//16, W//16] = [B, 300, 8, 8]
        
        return out_A  # 128 x (16x16)

class SmallerTransformerEncoderFor2DFeatures(nn.Module):
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
        locality_aware_feedforward=False,
        use_256_input=True,
    ):
        super(SmallerTransformerEncoderFor2DFeatures, self).__init__()


        self.shallow_cnn = SmallerDeepCNN300(
            input_size,
            num_in_features=48,
            output_channel=hidden_dim,
            dropout_rate=dropout_rate,
            use_256_input=use_256_input,
        )
        self.positional_encoding = PositionalEncoding2D(hidden_dim, device=device,
            use_adaptive_2d_encoding=False)

        self.attention_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_dim, filter_size, head_num, dropout_rate,
                    locality_aware_feedforward, False)
                for _ in range(layer_num)
            ]
        )
        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def forward(self, input):
        # input [B, C, H, W] = [B, 1, 128, 128]
        out = self.shallow_cnn(input)  
        b, c, h, w = out.size()
        # [B, ((NF+D*GR)//2+D*GR)//2, H//8, W//8] = [B, 300, 8, 8]
        # NF//4 + 3*D*GR//4 = SATRN.encoder.hidden_dim


        out = self.positional_encoding(out)  # [b, c, h, w]
        attn_bias = None

            # flatten
        out = out.view(b, c, h * w).transpose(1, 2)  # [b, h x w, c]

        for layer in self.attention_layers:
            out = layer(out, h, w, attn_bias=attn_bias)

        result = AttrDict(
            out=out, # [b, h x w, c]
            h=h,
            w=w,
        )
        
        return result 

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, input_size, filter_size, head_num, 
        dropout_rate=0.2):
        super(CustomTransformerDecoderLayer, self).__init__()

        # self.self_attention_layer = MultiHeadAttention(
        #     q_channels=input_size,
        #     k_channels=input_size,
        #     head_num=head_num,
        #     dropout=dropout_rate,
        #     use_tube=False,
        # )
        # self.self_attention_norm = nn.LayerNorm(normalized_shape=input_size)

        self.attention_layer = MultiHeadAttention(
            q_channels=input_size,
            k_channels=input_size,
            head_num=head_num,
            dropout=dropout_rate,
            use_tube=False,
        )
        self.attention_norm = nn.LayerNorm(normalized_shape=input_size)

        self.feedforward_layer = Feedforward(
            filter_size=filter_size, hidden_dim=input_size
        )
        self.feedforward_norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, tgt, tgt_prev, tgt_mask,
         return_attn=False):
        if tgt_prev is None:  # Train
            # tgt [B, H*W+S, D]
            # tgt_mask [B, H*W+S, H*W+S]

            out_dict = self.attention_layer(tgt, tgt, tgt, mask=tgt_mask, get_attn=return_attn)
            att = out_dict['out']
            if return_attn:
                attn = out_dict['attn']

            out = self.attention_norm(att + tgt)

            ff = self.feedforward_layer(out)
            out = self.feedforward_norm(ff + out)

            result = AttrDict(
                out=out,
            )

            if return_attn:
                result['attn'] = attn
 
            return result
        else:
            # TODO: out_dict attn

            # tgt, tgt_prev, src, tgt_mask
            # tgt [B, 1, D]
            # tgt_prev [B, h*w+t, D]
            # src [B, H*W, C]
            # tgt_mask [1, h*w+t+1] (t 현재 생성 index)

            tgt_prev = torch.cat([tgt_prev, tgt], 1) # [B, h*w+t+1, D]
            out_dict = self.attention_layer(tgt, tgt_prev, tgt_prev, tgt_mask, get_attn=return_attn) # [B, 1, D]

            att = out_dict['out']
            if return_attn:
                attn = out_dict['attn']

            out = self.attention_norm(att + tgt)

            ff = self.feedforward_layer(out)
            out = self.feedforward_norm(ff + out)

            result = AttrDict(
                out=out,
            )

            if return_attn:
                result['attn'] = attn

            return result

class CustomTransformerDecoder(nn.Module):
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
    ):
        super(CustomTransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(num_classes + 1, hidden_dim)
        self.hidden_dim = hidden_dim
        self.filter_dim = filter_dim
        self.num_classes = num_classes
        self.layer_num = layer_num
        self.head_num = head_num

        self.pos_encoder = PositionEncoder1D(
            in_channels=hidden_dim, dropout=dropout_rate, device=device
        )

        self.attention_layers = nn.ModuleList(
            [
                CustomTransformerDecoderLayer(
                    hidden_dim, filter_dim, head_num, dropout_rate
                )
                for _ in range(layer_num)
            ]
        )
        self.generator = nn.Linear(hidden_dim, num_classes)

        self.pad_id = pad_id
        self.st_id = st_id
        self.device = device

        self.channel_to_dim = nn.Linear(src_dim, hidden_dim)

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
        return_attn=False,
    ): 
        # src [B, H*W, C]

        # text [B, S] (S = not including [EOS])

        src = self.channel_to_dim(src) # [B, H*W, D]
        hw = src.size(1)

        if is_train and random.random() < teacher_forcing_ratio:
            tgt = self.text_embedding(text) # [B, S, D]
            tgt = self.pos_encoder(tgt, method='add') # [B, S, D]
            tgt = torch.cat((src, tgt), dim=1)
            
            # [B, 1, S] | [1, S, S] = [B, S, S]

            zero_tmp = torch.zeros_like(src[:, :, 0], dtype=torch.bool).unsqueeze(1) # [B, 1, H*W]
            pad_mask = torch.cat([zero_tmp, self.pad_mask(text)], dim=-1) # [B, 1, H*W + S]
            order_mask = self.order_mask(hw + text.size(1)) # [1, H*W + S, H*W + S]
            order_mask[:, :, :hw] = False
            tgt_mask = pad_mask | order_mask

            for layer in self.attention_layers:
                layer_output_dict = layer(tgt, None, tgt_mask, return_attn=return_attn)
                tgt = layer_output_dict['out'] # [B, S, D]
                if return_attn:
                    attns = layer_output_dict['attn']
            out = self.generator(tgt)
            out = out[:, hw:]
        else:
            # return attn 일단 안함
            # src [B, H*W, C]
            # text [B, S] (S = not including [EOS])

            out = []
            num_steps = batch_max_length - 1

            # target [B]
            b = src.size(0)
            target = torch.LongTensor(b).view(-1).fill_(self.st_id).to(self.device) # [START] token
            # features = [None] * self.layer_num

            if return_attn:
                attns = []

            features = []
            for layer in self.attention_layers:
                layer_output_dict = layer(src, None, None, return_attn)
                src = layer_output_dict['out'] # [B, h*w, D]
                features.append(src)

            for t in range(num_steps):
                target = target.unsqueeze(1) # [B, 1]
                tgt = self.text_embedding(target) # [B, 1, D]

                tgt = self.pos_encoder(tgt, method='add') # [B, 1, D]

                tgt_mask = self.order_mask(hw + t + 1) # [1, h*w+t+1, h*w+t+1]
                tgt_mask = tgt_mask[:, -1].unsqueeze(1)  # [1, h*w+t+1]

                for l, layer in enumerate(self.attention_layers):
                    layer_output_dict = layer(tgt, features[l], tgt_mask, return_attn)
                    tgt = layer_output_dict['out'] # [B, 1, D]
                    if return_attn:
                        attn = layer_output_dict['attn']

                    # features[l] [B, t+1, D]
                    features[l] = ( 
                        torch.cat([features[l], tgt], 1)
                    )

                    if return_attn:
                        attns.append(attn.cpu().data.numpy())

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
            result['attns'] = attns
        return result

class DecoderOnly(nn.Module):
    def __init__(self, FLAGS, train_dataset, device, checkpoint=None):
        super(DecoderOnly, self).__init__()

        if not hasattr(FLAGS.DecoderOnly, 'locality_aware_feedforward'):
            locality_aware_feedforward = False
        else:
            locality_aware_feedforward = FLAGS.DecoderOnly.locality_aware_feedforward

        rgb = FLAGS.data.rgb
        if FLAGS.data.use_flip_channel:
            rgb *= 2
        print('model use_256_input:', FLAGS.DecoderOnly.encoder.use_256_input)
        self.encoder = SmallerTransformerEncoderFor2DFeatures(
            input_size=rgb,
            hidden_dim=FLAGS.DecoderOnly.encoder.hidden_dim,
            filter_size=FLAGS.DecoderOnly.encoder.filter_dim,
            head_num=FLAGS.DecoderOnly.encoder.head_num,
            layer_num=FLAGS.DecoderOnly.encoder.layer_num,
            dropout_rate=FLAGS.dropout_rate,
            device=device,
            locality_aware_feedforward=locality_aware_feedforward,
            use_256_input=FLAGS.DecoderOnly.encoder.use_256_input,
        )

        self.decoder = CustomTransformerDecoder(
            num_classes=len(train_dataset.id_to_token),
            src_dim=FLAGS.DecoderOnly.decoder.src_dim,
            hidden_dim=FLAGS.DecoderOnly.decoder.hidden_dim,
            filter_dim=FLAGS.DecoderOnly.decoder.filter_dim,
            head_num=FLAGS.DecoderOnly.decoder.head_num,
            dropout_rate=FLAGS.dropout_rate,
            pad_id=train_dataset.token_to_id[PAD],
            st_id=train_dataset.token_to_id[START],
            layer_num=FLAGS.DecoderOnly.decoder.layer_num,
            device=device,
        )

        self.criterion = (
            nn.CrossEntropyLoss(ignore_index=train_dataset.token_to_id[PAD])
        )  # without ignore_index=train_dataset.token_to_id[PAD]

        if checkpoint:
            self.load_state_dict(checkpoint)

    def forward(self, input, expected, is_train, teacher_forcing_ratio,
             return_attn=False):
        # input [B, C, H, W] = [B, 1, 128, 128]
        result = AttrDict()

        enc_result_dict = self.encoder(input) # [B, H*W, C] = [B, 16*16, 300]
        enc_result = enc_result_dict['out']
        hw = enc_result.size(1)
        h = enc_result_dict['h']
        w = enc_result_dict['w']
        b = input.size(0)
   
        dec_result_dict = self.decoder(
            enc_result,
            expected[:, :-1],
            is_train,
            expected.size(1),
            teacher_forcing_ratio,
            return_attn=return_attn,
        )

        dec_result = dec_result_dict['out']
        if return_attn:
            attns = dec_result_dict['attns']

        result['out'] = dec_result
        result['hw'] = hw

        if return_attn:
            result['enc_result'] = enc_result.reshape(b, h, w, -1).cpu().data.numpy()
            result['attns'] = attns
        
        return result