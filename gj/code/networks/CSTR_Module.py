import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, input_channel, filter_channel):
        super().__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.shared_mlp = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(input_channel, filter_channel),
            nn.ReLU(),
            # nn.BatchNorm2d(filter_channel),
            nn.Linear(filter_channel ,input_channel),
            # nn.ReLU(),
            # nn.BatchNorm2d(input_channel),
        )


    def forward(self, x):
        # x [B, C, H, W]

        mp = self.max_pool(x) # [B, C, 1, 1]
        ap = self.avg_pool(x) # [B, C, 1, 1]

        mp = self.shared_mlp(mp)
        ap = self.shared_mlp(ap)

        score = mp + ap
        attn = torch.sigmoid(score)
        attn = attn[:, :, None, None]

        x = x * attn
        return x

class SAM(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, 1)

    def forward(self, x):
        # x [B, C, H, W]

        mp = torch.topk(x, 1, dim=1)[0] # [B, 1, H, W]
        ap = torch.mean(x, dim=1).unsqueeze(1) # [B, 1, H, W]

        concat = torch.cat((mp, ap), dim=1) # [B, 2, H, W]
        score = self.conv(concat)
        attn = torch.sigmoid(score) # [B, 1, H, W]

        x = attn * x

        return x

class CBAM_Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 1)
        self.cbam = CBAM(input_channel, filter_channel)
        self.sam = SAM()

    def forward(self, x):
        tmp = self.conv(x)
        tmp = self.cbam(tmp)
        tmp = self.sam(tmp)

        x = tmp + x
        return x

class SADM_A(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()

        self.max_pool = nn.MaxPool2d(2)
        self.one_conv = nn.Conv2d(input_channel, output_channel, 1)


    def forward(self, x):
        # x [B, C, H, W]
        b, c, h, w = x.shape

        q = x.reshape(b, -1, h*w) # [B, C, H*W]
        k = x.reshape(b, -1, h*w) # [B, C, H*W]

        score = torch.bmm(q.transpose(-1, -2), k) # [B, H*W, H*W]
        attn = torch.softmax(score, -1) # [B, H*W, H*W]

        v = self.one_conv(x) # [B, out_channel, H, W]
        v = v.reshape(b, -1, h*w) # [B, out_channel, H*W]
        v = v.transpose(-1, -2) # [B, H*W, out_channel]
        x = torch.bmm(attn, v) # [B, H*W, out_channel]
        x = x.transpose(-1, -1).reshape(b, -1, h, w)
        x = self.max_pool(x) # [B, out_channel, h//2, w//2]

        return x
        
        
