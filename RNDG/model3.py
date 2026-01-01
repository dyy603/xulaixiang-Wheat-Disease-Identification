from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops
from timm.layers import trunc_normal_  # 注意使用更新后的timm导入方式

def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

def _mcfg(**kwargs):
    cfg = dict(se_ratio=0., bottle_ratio=1., stem_width=32)
    cfg.update(**kwargs)
    return cfg

model_cfgs = {
    "regnety_400mf": _mcfg(w0=48, wa=27.89, wm=2.09, group_w=8, depth=16, se_ratio=0.25)
}

def generate_width_depth(wa, w0, wm, depth, q=8):
    assert wa > 0 and w0 > 0 and wm > 1 and w0 % q == 0
    widths_cont = np.arange(depth) * wa + w0
    width_exps = np.round(np.log(widths_cont / w0) / np.log(wm))
    widths_j = w0 * np.power(wm, width_exps)
    widths_j = np.round(np.divide(widths_j, q)) * q
    num_stages, max_stage = len(np.unique(widths_j)), width_exps.max() + 1
    assert num_stages == int(max_stage)
    assert num_stages == 4
    widths = widths_j.astype(int).tolist()
    return widths, num_stages

def adjust_width_groups_comp(widths: list, groups: list):
    groups = [min(g, w_bot) for g, w_bot in zip(groups, widths)]
    # Adjust w to an integral multiple of g
    widths = [int(round(w / g) * g) for w, g in zip(widths, groups)]
    return widths, groups

class ConvBNAct(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 kernel_s: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 act: Optional[nn.Module] = nn.ReLU(inplace=True)):
        super(ConvBNAct, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=kernel_s,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = nn.BatchNorm2d(out_c)
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RegHead(nn.Module):
    def __init__(self,
                 in_unit: int = 1768,
                 out_unit: int = 1000,
                 output_size: tuple = (2, 2),
                 drop_ratio: float = 0.5):  # 增加到 0.5
        super(RegHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

        # Dropout 层
        self.dropout = nn.Dropout(p=drop_ratio) if drop_ratio > 0 else nn.Identity()

        self.fc = nn.Linear(in_features=in_unit, out_features=out_unit)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)  # 正确位置：展平后，全连接前
        x = self.fc(x)
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, expand_c: int, se_ratio: float = 0.25):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class Bottleneck(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 stride: int = 1,
                 group_width: int = 1,
                 se_ratio: float = 0.,
                 drop_ratio: float = 0.):
        super(Bottleneck, self).__init__()

        self.conv1 = ConvBNAct(in_c=in_c, out_c=out_c, kernel_s=1)
        self.conv2 = ConvBNAct(in_c=out_c,
                               out_c=out_c,
                               kernel_s=3,
                               stride=stride,
                               padding=1,
                               groups=out_c // group_width)

        if se_ratio > 0:
            self.se = SqueezeExcitation(in_c, out_c, se_ratio)
        else:
            self.se = nn.Identity()

        self.conv3 = ConvBNAct(in_c=out_c, out_c=out_c, kernel_s=1, act=None)
        self.ac3 = nn.ReLU(inplace=True)

        if drop_ratio > 0:
            self.dropout = nn.Dropout(p=drop_ratio)
        else:
            self.dropout = nn.Identity()

        if (in_c != out_c) or (stride != 1):
            self.downsample = ConvBNAct(in_c=in_c, out_c=out_c, kernel_s=1, stride=stride, act=None)
        else:
            self.downsample = nn.Identity()

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.se(x)
        x = self.conv3(x)

        x = self.dropout(x)

        shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.ac3(x)
        return x

class RegStage(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 depth: int,
                 group_width: int,
                 se_ratio: float):
        super(RegStage, self).__init__()
        for i in range(depth):
            block_stride = 2 if i == 0 else 1
            block_in_c = in_c if i == 0 else out_c

            name = "b{}".format(i + 1)
            self.add_module(name,
                            Bottleneck(in_c=block_in_c,
                                       out_c=out_c,
                                       stride=block_stride,
                                       group_width=group_width,
                                       se_ratio=se_ratio))

    def forward(self, x: Tensor) -> Tensor:
        for block in self.children():
            x = block(x)
        return x
class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class DAttention(nn.Module):
    def __init__(
            self, channel, q_size, n_heads=8, n_groups=4,
            attn_drop=0.0, proj_drop=0.0, stride=1,
            offset_range_factor=4, use_pe=True, dwc_pe=True,
            no_off=False, fixed_pe=False, ksize=3, log_cpb=False,
    ):
        super().__init__()
        n_head_channels = channel // n_heads
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(
            channel, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            channel, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            channel, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, channel,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def forward(self, x):
       # print(f"Input shape to DAttention: {x.shape}")
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],
                mode='bilinear', align_corners=True)

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                        q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                               n_sample,
                                                                                               2).unsqueeze(1)).mul(
                    4.0)
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                        q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                               n_sample,
                                                                                               2).unsqueeze(1)).mul(
                    0.5)
                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
                                           g=self.n_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True)

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
       # print(f"Input shape to GAT: {x.shape}")
        # x shape: [B, N, in_features]
        B, N, _ = x.size()
        h = torch.matmul(x, self.W)  # [B, N, out_features]

        # 生成注意力分数
        h_i = h.unsqueeze(2)  # [B, N, 1, out_feat]
        h_j = h.unsqueeze(1)  # [B, 1, N, out_feat]
        a_input = torch.cat([h_i.expand(-1, -1, N, -1), h_j.expand(-1, N, -1, -1)], dim=-1)
        e = torch.matmul(a_input, self.a).squeeze(-1)  # [B, N, N]
        attention = self.leakyrelu(e)

        # 应用邻接矩阵掩码
        mask = adj.unsqueeze(0).expand(B, -1, -1)
        attention = attention.masked_fill(~mask, -1e4)  # 使用~mask并调整填充值
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 加权求和
        h_prime = torch.matmul(attention, h)  # [B, N, out_features]
        return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, n_final_out, dropout, alpha, nheads):
        super().__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
                          for _ in range(nheads)]
        for i, att in enumerate(self.attentions):
            self.add_module(f'att_{i}', att)

        self.out_att = GraphAttentionLayer(
            in_features=nhid * nheads,
            out_features=n_final_out,
            dropout=dropout,
            alpha=alpha,
            concat=False
        )

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x

    # 新增代码部分 --------------------------------------------------------


class GATWrapper(nn.Module):
    def __init__(self, in_c, nhid, nheads, dropout, alpha):
        super().__init__()
        self.gat = GAT(
            nfeat=in_c,
            nhid=nhid,
            n_final_out=in_c,  # 保持输出维度与输入一致
            dropout=dropout,
            alpha=alpha,
            nheads=nheads
        )

    def _build_adj(self, H, W, device):
        """创建3x3邻接矩阵（优化版）"""
        N = H * W
        adj = torch.zeros(N, N, dtype=torch.bool, device=device)  # 使用布尔类型

        idx = torch.arange(H * W, device=device).view(H, W)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                x = torch.clamp((idx // W) + dx, 0, H - 1)
                y = torch.clamp((idx % W) + dy, 0, W - 1)
                neighbors = x * W + y
                adj[idx.view(-1), neighbors.view(-1)] = True  # 设置为True表示相邻
        return adj  # [N, N]

    def forward(self, x):
        B, C, H, W = x.size()
        # 转换特征图形状为图节点格式 [B, N, C]
        x_nodes = x.view(B, C, H * W).permute(0, 2, 1)
        # 构建邻接矩阵
        adj = self._build_adj(H, W, x.device)
        # 应用GAT
        out = self.gat(x_nodes, adj)  # [B, N, C]
        # 恢复特征图形状
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

class RegNetWithAttention(nn.Module):
    def __init__(self, cfg: dict, in_c=3, num_classes=1000, zero_init_last_bn=True, dropout_prob=0.2,
                 gat_config=None, dattention_config=None):
        super().__init__()
        self.gat_config = gat_config or {}
        self.dattention_config = dattention_config or {}
        stem_c = cfg["stem_width"]
        self.stem = ConvBNAct(in_c, stem_c, kernel_s=3, stride=2, padding=1)
        self.dropout = nn.Dropout(dropout_prob)
        input_channels = stem_c
        stage_info = self._build_stage_info(cfg)
        self.final_dropout = nn.Dropout(0.2)
        # 跟踪特征图尺寸（假设输入224x224）
        current_h, current_w = 112, 112
        stage_output_sizes = []
        # 构建各阶段
        for i, stage_args in enumerate(stage_info):
            stage_name = f"s{i + 1}"
            self.add_module(stage_name, RegStage(input_channels, **stage_args))
            input_channels = stage_args["out_c"]
            stage_output_h, stage_output_w = current_h // 2, current_w // 2
            stage_output_sizes.append((stage_output_h, stage_output_w))
            current_h, current_w = stage_output_h, stage_output_w

            # 插入DAttention
            if self.dattention_config.get('use_dattention', False) and (i + 1) in self.dattention_config.get('stages',
                                                                                                             []):
                dattn = DAttention(
                    channel=input_channels,
                    q_size=(stage_output_h, stage_output_w),
                    n_heads=self.dattention_config.get('n_heads', 8),
                    n_groups=self.dattention_config.get('n_groups', 4),
                )
                self.add_module(f"dattn_{i + 1}", dattn)
            # 插入GAT
            if self.gat_config.get('use_gat', False) and (i + 1) in self.gat_config.get('stages', []):
                gat_layer = GATWrapper(
                    in_c=input_channels,
                    nhid=self.gat_config.get('nhid', 64),
                    nheads=self.gat_config.get('nheads', 4),
                    dropout=self.gat_config.get('dropout', 0.1),
                    alpha=self.gat_config.get('alpha', 0.2)
                )
                self.add_module(f"gat_s{i + 1}", gat_layer)

        # 添加缺失的head模块
        self.head = RegHead(
            in_unit=input_channels,
            out_unit=num_classes,
            output_size=(1, 1),
            drop_ratio=0.5  # 增强正则化
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, "zero_init_last_bn"):
                    m.zero_init_last_bn()
    def _build_stage_info(self, cfg: dict):
        """生成每个stage的参数字典列表（核心逻辑）"""
        wa, w0, wm, d = cfg["wa"], cfg["w0"], cfg["wm"], cfg["depth"]

        # 生成宽度列表和阶段数（与generate_width_depth逻辑匹配）
        widths, num_stages = generate_width_depth(wa, w0, wm, d)

        # 获取各阶段的宽度和重复次数（原逻辑）
        stage_widths, stage_depths = np.unique(widths, return_counts=True)
        stage_groups = [cfg['group_w'] for _ in range(num_stages)]  # 从配置获取分组数

        # 调整宽度和分组数的兼容性（保持原数据预处理逻辑）
        stage_widths, stage_groups = adjust_width_groups_comp(stage_widths, stage_groups)

        # 构建阶段配置字典列表
        stage_info = []
        for i in range(num_stages):
            stage_params = {
                "out_c": int(stage_widths[i]),  # 当前阶段输出通道数
                "depth": int(stage_depths[i]),  # 当前阶段重复块数
                "group_width": int(stage_groups[i]),  # 分组卷积的组宽
                "se_ratio": cfg["se_ratio"]  # SE模块的压缩比例
            }
            stage_info.append(stage_params)

        return stage_info

    def forward(self, x):
        x = self.stem(x)
        for module in self.children():
            if isinstance(module, (ConvBNAct, RegHead)):
                continue
            x = module(x)
        x = self.head(x)  # 这里已包含 Dropout
        return x

def create_regnet_with_attention(model_name, num_classes=1000, **kwargs):
    cfg = model_cfgs[model_name]
    return RegNetWithAttention(
        cfg,
        num_classes=num_classes,
        gat_config=kwargs.get('gat_config'),
        dattention_config=kwargs.get('dattention_config')
    )
# 示例用法
if __name__ == "__main__":
    model = create_regnet_with_attention(
        "regnety_400mf",
        num_classes=1000,
        dattention_config={'use_dattention': True, 'stages': [3]},
        gat_config={'use_gat': True, 'stages': [3]}
    )
    x = torch.randn(2, 3, 224, 224)
    print(model(x).shape)  # 应输出 torch.Size([2, 1000])
    # 在示例代码后添加
    print([name for name, _ in model.named_modules()])
    # 应该包含类似dattn_2、gat_s3的子模块