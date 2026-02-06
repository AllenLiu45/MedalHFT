import torch
import torch.nn as nn
import torch.nn.functional as F

max_punish = 1e12


def modulate(x, shift, scale):

    return x * (1 + scale) + shift



import numpy as np

def generate_freq_ranges_with_bottleneck(
    look_back_window,
    B=3,
    r_min=8,
    r_max=64,
    p=2.0
):
    """
    Frequency band partition with bottleneck allocation.
    Args:
        look_back_window (int): temporal window size
        B (int): number of frequency bands (default: 3)
        r_min (int): minimum bottleneck width
        r_max (int): maximum bottleneck width
        p (float): decay curvature
    """

    # FFT positive frequencies
    F = look_back_window // 2 + 1

    # uniform frequency boundaries (paper: b_k = floor(k/B * F))
    boundaries = [int(np.floor(k * F / B)) for k in range(B + 1)]

    freq_ranges = {}
    bottlenecks = {}

    band_names = ['low', 'mid', 'high']

    for k in range(B):
        b_start = boundaries[k]
        b_end = boundaries[k + 1]

        # center frequency f_k = (b_{k-1} + b_k) / 2
        f_k = 0.5 * (b_start + b_end)

        # bottleneck allocation (paper equation)
        r_k = int(
            np.floor(
                r_min + (r_max - r_min) * (1 - f_k / F) ** p
            )
        )

        freq_ranges[band_names[k]] = (b_start, b_end)
        bottlenecks[band_names[k]] = r_k

    return freq_ranges








def generate_freq_ranges(look_back_window):

    freq_count = look_back_window // 2 + 1

    if freq_count <= 12:

        low_end = freq_count // 3
        mid_end = freq_count * 2 // 3
        return {
            'low': (0, low_end),
            'mid': (low_end, mid_end),
            'high': (mid_end, freq_count)
        }
    else:

        low_end = max(1, freq_count // 4)
        mid_end = freq_count * 3 // 4
        return {
            'low': (0, low_end),
            'mid': (low_end, mid_end),
            'high': (mid_end, freq_count)
        }




class SpectralEncoder(nn.Module):
    """
    input: x [B, T, F_in]
    output: h_spec [B, F_freq, hidden_dim]
    """
    def __init__(self, in_feat_dim: int, hidden_dim: int):
        super(SpectralEncoder, self).__init__()
        # real + image -> 2 * in_feat_dim
        self.proj = nn.Linear(2 * in_feat_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, F_in]
        """
        # FFT: [B, F_freq, F_in]
        X = torch.fft.rfft(x, dim=1)
        X_real = X.real                    # [B, F_freq, F_in]
        X_imag = X.imag                    # [B, F_freq, F_in]

        spec = torch.cat([X_real, X_imag], dim=-1)   # [B, F_freq, 2*F_in]
        h = self.proj(spec)                         # [B, F_freq, hidden_dim]
        h = self.ffn(h)                             # [B, F_freq, hidden_dim]
        return h



class BandSplitter(nn.Module):

    def __init__(self, freq_ranges):
        super().__init__()
        self.freq_ranges = freq_ranges

    def forward(self, spec):  # spec: [B, F_freq, H]
        outs = {}
        for band, (f_start, f_end) in self.freq_ranges.items():
            outs[band] = spec[:, f_start:f_end, :]  # [B, F_band, H]
        return outs   # {'low': tensor, 'mid': tensor, 'high': tensor}


class BandConvEncoder(nn.Module):
    """
    input: band_spec [B, F_band, H]
    output: v_band [B, H]
    """
    def __init__(self, hidden_dim, kernel_size=3, n_layers=2):
        super().__init__()

        layers = []
        for _ in range(n_layers):
            layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers) if layers else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, band_spec):
        """
        band_spec: [B, F_band, H]
        """

        x = band_spec.mean(dim=1)  # [B, H]
        x = self.layers(x)
        x = self.norm(x)
        return x


class BandExpert(nn.Module):
    """
    input: concat([v1_band, v3_band, v5_band])  --> [B, 3H]
    output: h_band [B, H]
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, v1, v3, v5):  # 各 [B, H]
        x = torch.cat([v1, v3, v5], dim=-1)  # [B, 3H]
        return self.net(x)  # [B, H]



class TriFrequencyExperts(nn.Module):

    def __init__(self, hidden_dim, freq_ranges):
        super().__init__()
        self.splitter = BandSplitter(freq_ranges)


        self.low_encoder  = BandConvEncoder(hidden_dim)
        self.mid_encoder  = BandConvEncoder(hidden_dim)
        self.high_encoder = BandConvEncoder(hidden_dim)


        self.low_expert = BandExpert(hidden_dim)
        self.mid_expert = BandExpert(hidden_dim)
        self.high_expert = BandExpert(hidden_dim)

    def forward(self, spec_1, spec_3, spec_5):


        # print(f"spec_1:{spec_1.shape}")
        # print(f"spec_3:{spec_3.shape}")
        # print(f"spec_5:{spec_5.shape}")

        bands_1 = self.splitter(spec_1)  # {'low': [B,F_l,H], 'mid':..., 'high':...}
        bands_3 = self.splitter(spec_3)
        bands_5 = self.splitter(spec_5)

        # print(f"bands_1:{bands_1}")

        e1_low = self.low_encoder(bands_1['low'])
        e3_low = self.low_encoder(bands_3['low'])
        e5_low = self.low_encoder(bands_5['low'])
        h_trend = self.low_expert(e1_low, e3_low, e5_low)  # [B,H]

        e1_mid = self.mid_encoder(bands_1['mid'])
        e3_mid = self.mid_encoder(bands_3['mid'])
        e5_mid = self.mid_encoder(bands_5['mid'])
        h_struct = self.mid_expert(e1_mid, e3_mid, e5_mid)  # [B,H]

        e1_high = self.high_encoder(bands_1['high'])
        e3_high = self.high_encoder(bands_3['high'])
        e5_high = self.high_encoder(bands_5['high'])
        h_micro = self.high_expert(e1_high, e3_high, e5_high)  # [B,H]
        # print(f"e1_high:{e1_high}")
        # print(f"e3_high:{e3_high}")
        # print(f"e5_high:{e5_high}")

        return h_trend, h_struct, h_micro


class FrequencyBandGate(nn.Module):
    """
    input: h_trend, h_struct, h_micro
    output: h_freq_fused, band_alpha
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, h_trend, h_struct, h_micro):
        # [B, 3H]
        h_cat = torch.cat([h_trend, h_struct, h_micro], dim=-1)
        logits = self.gate(h_cat)              # [B, 3]
        alpha  = F.softmax(logits, dim=-1)     # [B, 3]


        h_fused = (
            alpha[:, 0:1] * h_trend +
            alpha[:, 1:2] * h_struct +
            alpha[:, 2:3] * h_micro
        )  # [B, H]

        return h_fused, alpha



class FeatureFusion_Low(nn.Module):

    def __init__(self, hidden_dim):
        super(FeatureFusion_Low, self).__init__()
        self.fc = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x, spectral_context):
        # x, spectral_context: [B, H]
        combined_features = torch.cat([x, spectral_context], dim=-1)  # [B, 2H]
        weight = torch.sigmoid(self.fc(combined_features))            # [B, 1]
        weight = weight.expand_as(x)                                  # [B, H]
        fused_feature = weight * x + (1 - weight) * spectral_context
        return fused_feature

class FeatureFusion_High(nn.Module):

    def __init__(self, hidden_dim):
        super(FeatureFusion_High, self).__init__()
        self.fc1 = nn.Linear(3 * hidden_dim, 1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)

    def forward(self, x, spectral_context):
        # x, spectral_context: [B, H]
        combined_features = torch.cat([x, spectral_context], dim=-1)  # [B, 3H]
        weight = torch.sigmoid(self.fc1(combined_features))            # [B, 1]
        weight = weight.expand_as(x)
        spectral_context = self.fc2(spectral_context)
        fused_feature = weight * x + (1 - weight) * spectral_context
        # print(f"fused_feature:{fused_feature.shape}")
        return fused_feature


class market_observer(nn.Module):

    def __init__(self,
                 state_dim_minute1,
                 state_dim_minute3,
                 state_dim_minute5,
                 look_back_window,
                 hidden_dim):

        super(market_observer, self).__init__()


        self.spec_enc_1 = SpectralEncoder(state_dim_minute1, hidden_dim)
        self.spec_enc_3 = SpectralEncoder(state_dim_minute3, hidden_dim)
        self.spec_enc_5 = SpectralEncoder(state_dim_minute5, hidden_dim)


        freq_ranges = generate_freq_ranges(look_back_window)
        self.tri_freq_experts = TriFrequencyExperts(hidden_dim, freq_ranges)


        self.band_moe = FrequencyBandGate(hidden_dim)


    def observe(self,
                minute1_state: torch.Tensor,
                minute3_state: torch.Tensor,
                minute5_state: torch.Tensor):

        if minute1_state.dim() == 2:
            minute1_state = minute1_state.unsqueeze(0)
        if minute3_state.dim() == 2:
            minute3_state = minute3_state.unsqueeze(0)
        if minute5_state.dim() == 2:
            minute5_state = minute5_state.unsqueeze(0)


        spec_1 = self.spec_enc_1(minute1_state)  # [B, F_freq, H]
        spec_3 = self.spec_enc_3(minute3_state)
        spec_5 = self.spec_enc_5(minute5_state)

        # Tri-Frequency Expert Networks
        h_trend, h_struct, h_micro = self.tri_freq_experts(spec_1, spec_3, spec_5)  # 各 [B,H]

        # Frequency-Band-Level MoE
        h_freq_fused, band_alpha = self.band_moe(h_trend, h_struct, h_micro)  # [B,H], [B,3]

        return h_freq_fused, band_alpha

    def forward(self,
                minute1_state: torch.Tensor,
                minute3_state: torch.Tensor,
                minute5_state: torch.Tensor):

        h_freq_fused, _ = self.observe(minute1_state, minute3_state, minute5_state)
        return h_freq_fused

class subagent(nn.Module):
    def __init__(self,
                 state_dim_1,
                 state_dim_2,
                 state_dim_minute1,
                 state_dim_minute3,
                 state_dim_minute5,
                 look_back_window,
                 action_dim,
                 hidden_dim):

        super(subagent, self).__init__()

        self.fc1 = nn.Linear(state_dim_1, hidden_dim)
        self.fc2 = nn.Linear(state_dim_2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Embedding(action_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, 1)
        )
        self.register_buffer("max_punish", torch.tensor(max_punish))

        self.market_observer = market_observer(
            state_dim_minute1,
            state_dim_minute3,
            state_dim_minute5,
            look_back_window,
            hidden_dim
        )

        self.feature_fusion = FeatureFusion_Low(hidden_dim)

    def forward(self,
                single_state: torch.Tensor,
                trend_state: torch.Tensor,
                minute1_state: torch.Tensor,
                minute3_state: torch.Tensor,
                minute5_state: torch.Tensor,
                previous_action: torch.Tensor):
        # ---------- AdaLN ----------
        action_hidden = self.embedding(previous_action)         # [B, H]
        single_state_hidden = self.fc1(single_state)            # [B, H]
        trend_state_hidden = self.fc2(trend_state)              # [B, H]
        c = action_hidden + trend_state_hidden                  # [B, H]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1) # 各 [B, H]
        x = modulate(self.norm(single_state_hidden), shift, scale)  # [B, H]

        # ---------- market observer ----------
        h_freq, band_alpha = self.market_observer.observe(
            minute1_state,
            minute3_state,
            minute5_state
        )

        combined_feature = self.feature_fusion(x, h_freq)  # [B,H]

        value = self.value(combined_feature)               # [B, 1]
        advantage = self.advantage(combined_feature)       # [B, action_dim]
        q = value + advantage - advantage.mean(dim=1, keepdim=True)  # [B, action_dim]
        return q


class hyperagent(nn.Module):
    def __init__(self, state_dim_1, state_dim_2, state_clf, state_dim_minute1, state_dim_minute3, state_dim_minute5, look_back_window, action_dim, hidden_dim):
        super(hyperagent, self).__init__()
        self.fc1 = nn.Linear(state_dim_1 + state_dim_2, hidden_dim)
        self.fc2 = nn.Linear(state_clf, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim * 2, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Embedding(action_dim, hidden_dim)
        self.gru_encoder = nn.GRU(20, hidden_dim // 2, batch_first=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=True)
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, 6),
            nn.Softmax(dim=1)
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

        self.market_observer = market_observer(
            state_dim_minute1,
            state_dim_minute3,
            state_dim_minute5,
            look_back_window,
            hidden_dim
        )

        self.feature_fusion = FeatureFusion_High(hidden_dim)


    def forward(self,
                single_state: torch.tensor,
                trend_state: torch.tensor,
                class_state: torch.tensor,
                minute1_state: torch.Tensor,
                minute3_state: torch.Tensor,
                minute5_state: torch.Tensor,
                previous_action: torch.tensor,):

        action_hidden = self.embedding(previous_action)
        state_hidden = self.fc1(torch.cat([single_state, trend_state], dim=1))
        x = torch.cat([action_hidden, state_hidden], dim=1)
        c = self.fc2(class_state)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        # print(f"x:{x.shape}")

        # ---------- market observer ----------
        h_freq, band_alpha = self.market_observer.observe(
            minute1_state,
            minute3_state,
            minute5_state
        )

        combined_feature = self.feature_fusion(x, h_freq)  # [B,H]

        weight = self.net(combined_feature)

        return weight


    def encode(self,
                single_state: torch.tensor,
                trend_state: torch.tensor,
                minute1_state: torch.tensor,
                minute3_state: torch.tensor,
                minute5_state: torch.tensor,
                previous_action: torch.tensor,):
        action_hidden = self.embedding(previous_action)
        state_hidden = self.fc1(torch.cat([single_state, trend_state], dim=1))
        minute1_state = self.gru_encoder(minute1_state)[1]
        minute3_state = self.gru_encoder(minute3_state)[1]
        minute5_state = self.gru_encoder(minute5_state)[1]
        x = torch.cat([action_hidden, state_hidden], dim=1)

        # print(f"x:{x.shape}")
        # print(f"minute1_state:{minute1_state.shape}")
        if minute1_state.dim() == 3:
            minute1_state = minute1_state.squeeze(0)
        if minute3_state.dim() == 3:
            minute3_state = minute3_state.squeeze(0)
        if minute5_state.dim() == 3:
            minute5_state = minute5_state.squeeze(0)

        x = torch.cat([minute1_state, minute3_state, minute5_state, x], dim=1)
        return x


def calculate_q(w, qs):
    q_tensor = torch.stack(qs)
    q_tensor = q_tensor.permute(1, 0, 2)
    weights_reshaped = w.view(-1, 1, 6)
    combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)

    return combined_q
