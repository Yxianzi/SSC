import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffGuidedFilter(nn.Module):
    """
    可微导向滤波 (Differentiable Guided Filter)
    """

    def __init__(self, r=1, eps=1e-8):
        super(DiffGuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        # 使用 AvgPool2d 实现 BoxFilter 逻辑
        self.boxfilter = nn.AvgPool2d(kernel_size=2 * r + 1, stride=1, padding=r, count_include_pad=False)

    def forward(self, guidance, src):
        N_x = self.boxfilter(torch.ones_like(guidance))

        mean_x = self.boxfilter(guidance) / N_x
        mean_y = self.boxfilter(src) / N_x
        mean_xx = self.boxfilter(guidance * guidance) / N_x
        mean_xy = self.boxfilter(guidance * src) / N_x

        var_x = mean_xx - mean_x * mean_x
        cov_xy = mean_xy - mean_x * mean_y

        a = cov_xy / (var_x + self.eps)
        b = mean_y - a * mean_x

        mean_a = self.boxfilter(a) / N_x
        mean_b = self.boxfilter(b) / N_x

        q = mean_a * guidance + mean_b
        return q


class EndToEndSSC(nn.Module):
    """
    端到端光谱风格校准 (Spectral Style Calibration)
    """

    def __init__(self, channels, r=1, eps=0.009):
        super(EndToEndSSC, self).__init__()
        # 统计量编码器：输入均值和方差(2*C)，输出仿射参数 alpha, beta (2*C)
        self.stats_encoder = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Linear(channels, channels * 2)
        )
        self.softplus = nn.Softplus()
        self.gf = DiffGuidedFilter(r=r, eps=eps)

    def forward(self, x_s, x_t):
        B, C, H, W = x_s.shape

        # 1. 提取目标域统计量
        t_mean = x_t.view(B, C, -1).mean(dim=2)
        t_var = x_t.view(B, C, -1).var(dim=2)
        t_stats = torch.cat([t_mean, t_var], dim=1)

        # 2. 预测仿射变换参数
        affine_params = self.stats_encoder(t_stats)
        alpha = self.softplus(affine_params[:, :C]).view(B, C, 1, 1)  # Softplus保证单调缩放 (>0)
        beta = affine_params[:, C:].view(B, C, 1, 1)

        # 3. 得到 Reference (x')
        x_calibrated = alpha * x_s + beta

        # 4. 可微导向滤波 (以 x_calibrated 为引导，x_s 为原图)
        x_gf = self.gf(guidance=x_calibrated, src=x_s)

        return x_gf, x_calibrated


def spectral_angle_loss(x, x_calibrated):
    """
    物理一致性正则：光谱角约束 (SAM Loss)
    """
    x_flat = x.view(x.size(0), x.size(1), -1)
    xc_flat = x_calibrated.view(x_calibrated.size(0), x_calibrated.size(1), -1)

    dot_product = torch.sum(x_flat * xc_flat, dim=1)
    norm_x = torch.norm(x_flat, dim=1)
    norm_xc = torch.norm(xc_flat, dim=1)

    cos_theta = dot_product / (norm_x * norm_xc + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-8, 1.0 - 1e-8)
    sam = torch.acos(cos_theta)
    return torch.mean(sam)