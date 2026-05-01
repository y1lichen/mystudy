"""
DecisionPINN — v8.5

水泵模型全面升級：實作論文 Eq.(12) 的物理結構
  P = λV³ / Sigmoid(NN(V, λV², num))

【三種泵的物理設計】

1. CDWPumpSPINN（冷凝水泵，CDW × 3，定速）
   - V: CDWL_CW_FLOW（冷凝水流量，來自 state[9]）
   - λ_cdw: 可學習參數 exp(log_lam_cdw)（無差壓量測）
   - P = λV³ * UNIT_CONV / Sigmoid(NN(V_norm, λV_norm², num/3))
   - 乘以 num_running（每台冷機帶動 1 台 CDW 泵）

2. PRIPumpSPINN（主側冷水泵，Primary CHW × 3，定速）
   - V: CWL_PRI_CW_FLOW（主側流量，來自 state[10]）
   - λ_pri: 可學習參數 exp(log_lam_pri)
   - 結構與 CDW 泵相同

3. SECPumpSPINN（次側變速泵，Secondary CHW × 2，VFD）
   - V: CWL_SEC_CW_FLOW（次側流量，來自 state[4]）
   - λ_sec: 從資料計算 λ = CWL_SEC_DP / V²（來自 state[11]）
   - P = λV³ * UNIT_CONV / Sigmoid(NN(V_norm, λV_norm², num_sec/2))

【單位換算】
   UNIT_CONV = 6.309e-5 [m³/s/GPM] × 249.1 [Pa/inH2O] / 1000 [W/kW]
             = 1.572e-5  kW / (GPM × inH2O)
   P[kW] = λ[inH2O/GPM²] × V[GPM]³ × UNIT_CONV / η

【歸一化空間的公式】
   V_norm = V / V_max
   H_norm = λ_norm × V_norm²  （= H / H_ref）
   ideal  = λ_norm × V_norm³  （理想功率，歸一化）
   P[kW]  = ideal × P_scale / Sigmoid(NN(V_norm, H_norm, num))
   其中 P_scale = λ_ref × V_max³ × UNIT_CONV 吸收量綱，可學習

   實作上令 P_scale = exp(log_scale)，與 log_lam 合併為一個可學習縮放
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

TOWER_MAX_KW   = 40.0
# 單位換算係數：GPM × inH2O → kW
UNIT_CONV      = 6.309e-5 * 249.1 / 1000   # = 1.572e-5

_TOWER_SCALE   = torch.tensor([500.0, 290.0, 290.0, 15.0, 2.0])
_CHILLER_SCALE = torch.tensor([300.0, 283.0, 295.0,  1.0, 1.0])


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, output_bias=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )
        nn.init.constant_(self.net[-1].bias, output_bias)

    def forward(self, x):
        return self.net(x)


class PumpPhysicsBlock(nn.Module):
    """
    單一泵群的 S-PINN 物理結構（論文 Eq.12）：

        P = exp(log_scale) × V_norm³ × lam_norm
            / Sigmoid(NN(V_norm, lam_norm × V_norm², num_norm))

    其中 exp(log_scale) 為可學習縮放，吸收 UNIT_CONV × V_max³ × λ_ref。
    lam_norm：歸一化管路阻力係數
      - 定速泵：exp(log_lam)（可學習，因無差壓量測）
      - 變速泵：從資料即時計算 λ_sec = dp / V²（傳入 lam_input）

    效率網路 NN 輸入：[V_norm, H_norm(=λV²), num_norm]
    輸出 Sigmoid ∈ (0,1) 即為 η（效率）
    Clamp 至 [0.2, 0.95] 限制物理範圍。
    """
    def __init__(self, learnable_lam=True, init_log_scale=2.0):
        """
        learnable_lam: True → λ 為可學習參數（CDW/Primary 泵）
                       False → λ 從外部傳入（Secondary 泵）
        init_log_scale: exp(init_log_scale) = 初始 P_scale ≈ 7.4 kW（調整至資料量級）
        """
        super().__init__()
        self.learnable_lam = learnable_lam
        if learnable_lam:
            # 初始化 λ_norm=1（中性），訓練過程中自由調整
            self.log_lam = nn.Parameter(torch.tensor(0.0))
        # P_scale：吸收 UNIT_CONV × V_max³ × λ_ref，可學習
        self.log_scale = nn.Parameter(torch.tensor(float(init_log_scale)))
        # 效率網路：3維輸入（V_norm, H_norm, num_norm）
        self.eff_net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        # 初始化讓 sigmoid 輸出接近 0.5（η=0.5，中性初始）
        nn.init.zeros_(self.eff_net[-1].weight)
        nn.init.zeros_(self.eff_net[-1].bias)

    def forward(self, V_norm, num_norm, lam_input=None, on_mask=None):
        """
        V_norm:    (B,1) 流量歸一化
        num_norm:  (B,1) 台數歸一化（num / max_num）
        lam_input: (B,1) 外部λ（learnable_lam=False 時使用），已歸一化
        on_mask:   (B,1) bool，開機遮罩，關機強制輸出 0
        """
        if self.learnable_lam:
            lam_norm = torch.exp(self.log_lam).expand(V_norm.shape)  # (B,1)
        else:
            lam_norm = lam_input.clamp(min=1e-4)   # 防止零除

        P_scale = torch.exp(self.log_scale)
        H_norm  = lam_norm * V_norm ** 2            # 壓頭（歸一化）
        ideal   = lam_norm * V_norm ** 3            # 理想功率（歸一化）

        eff_input = torch.cat([V_norm, H_norm, num_norm], dim=1)
        eta = torch.sigmoid(self.eff_net(eff_input)).clamp(0.2, 0.95)

        power = P_scale * ideal / eta               # kW

        if on_mask is not None:
            power = power * on_mask.float()

        return power

    def get_lam(self):
        if self.learnable_lam:
            return torch.exp(self.log_lam).item()
        return None

    def get_scale(self):
        return torch.exp(self.log_scale).item()


class CDWPumpSPINN(nn.Module):
    """
    冷凝水泵（CDW Pump × 3）S-PINN。
    λ_cdw 可學習，V_cdw 來自 CDWL_CW_FLOW（state[9] × v_cdw_max）。
    每台冷機帶動 1 台 CDW 泵，num = num_running。
    """
    def __init__(self):
        super().__init__()
        self.block = PumpPhysicsBlock(learnable_lam=True, init_log_scale=2.5)

    def forward(self, v_cdw_norm, num_running):
        on_mask = (num_running > 0)
        num_norm = num_running / 3.0
        return self.block(v_cdw_norm, num_norm, on_mask=on_mask)


class PRIPumpSPINN(nn.Module):
    """
    主側冷水泵（Primary CHW Pump × 3）S-PINN。
    λ_pri 可學習，V_pri 來自 CWL_PRI_CW_FLOW（state[10] × v_pri_max）。
    """
    def __init__(self):
        super().__init__()
        self.block = PumpPhysicsBlock(learnable_lam=True, init_log_scale=2.5)

    def forward(self, v_pri_norm, num_running):
        on_mask = (num_running > 0)
        num_norm = num_running / 3.0
        return self.block(v_pri_norm, num_norm, on_mask=on_mask)


class SECPumpSPINN(nn.Module):
    """
    次側變速泵（Secondary CHW Pump × 2）S-PINN。
    λ_sec 從資料計算（state[11]），不可學習，代入 lam_input。
    num_sec = min(num_running, 2)。
    """
    def __init__(self):
        super().__init__()
        self.block = PumpPhysicsBlock(learnable_lam=False, init_log_scale=3.0)

    def forward(self, v_sec_norm, num_sec_pumps, lam_sec_norm):
        on_mask = (num_sec_pumps > 0)
        num_norm = num_sec_pumps / 2.0
        return self.block(v_sec_norm, num_norm,
                          lam_input=lam_sec_norm, on_mask=on_mask)


class PumpSPINN(nn.Module):
    """
    整合三種泵的 S-PINN，論文 Eq.(12) 完整實作。

    forward 需要的 state 索引：
      state[4]  = CWL_SEC_CW_FLOW（原始 GPM，由 physics_forward 歸一化）
      state[9]  = V_cdw_norm（CDWL_CW_FLOW / v_cdw_max）
      state[10] = V_pri_norm（CWL_PRI_CW_FLOW / v_pri_max）
      state[11] = lam_sec_norm（CWL_SEC_DP / V² / lam_sec_scale）
    """
    def __init__(self):
        super().__init__()
        self.cdw_pump = CDWPumpSPINN()
        self.pri_pump = PRIPumpSPINN()
        self.sec_pump = SECPumpSPINN()

    def forward(self, v_sec_norm, v_cdw_norm, v_pri_norm,
                lam_sec_norm, num_running, num_sec_pumps):
        p_cdw = self.cdw_pump(v_cdw_norm, num_running)
        p_pri = self.pri_pump(v_pri_norm, num_running)
        p_sec = self.sec_pump(v_sec_norm, num_sec_pumps, lam_sec_norm)
        return p_cdw, p_pri, p_sec


class TowerTPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(in_dim=5, out_dim=1, hidden=64, output_bias=0.0)
        self.register_buffer('scale', _TOWER_SCALE)

    def forward(self, features):
        mu    = torch.sigmoid(self.net(features / self.scale)) * TOWER_MAX_KW
        sigma = torch.full_like(mu, TOWER_MAX_KW * 0.1)
        return mu, sigma


class ChillerTSPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta_s_net = MLP(in_dim=5, out_dim=2, hidden=64, output_bias=0.3)
        self.register_buffer('scale', _CHILLER_SCALE)

    def forward(self, features):
        Q_load  = features[:, 0:1]
        T_chw   = features[:, 1:2]
        T_cdw   = features[:, 2:3]
        out     = self.delta_s_net(features / self.scale)
        delta_S = F.softplus(out[:, 0:1])
        sigma   = F.softplus(out[:, 1:2]) + 1e-3
        delta_T  = torch.clamp(T_cdw - T_chw, min=1.0)
        power_mu = (delta_T / T_chw.clamp(min=270.0)) * Q_load + delta_S * T_cdw
        return F.relu(power_mu), sigma, delta_S


def _trend_loss(net_fn, feat_ranges, fixed_ranges, n_steps, device, batch_size):
    D     = len(fixed_ranges)
    total = torch.tensor(0.0, device=device)
    for name, (lo, hi, trend, idx) in feat_ranges.items():
        if trend is None:
            continue
        cols  = [torch.empty(batch_size, 1, device=device).uniform_(*fixed_ranges[i])
                 for i in range(D) if i != idx]
        fixed = torch.cat(cols, dim=1)
        seq, violations, prev_mu = torch.linspace(lo, hi, n_steps, device=device), [], None
        for val in seq:
            feat = torch.zeros(batch_size, D, device=device)
            j = 0
            for i in range(D):
                if i == idx: feat[:, i] = val
                else:        feat[:, i] = fixed[:, j]; j += 1
            out = net_fn(feat)
            mu  = out[0] if isinstance(out, tuple) else out
            if prev_mu is not None:
                diff = mu - prev_mu
                violations.append(F.relu(-diff) if trend == 'increase' else F.relu(diff))
            prev_mu = mu.detach()
        if violations:
            total = total + torch.stack(violations).mean()
    return total


def trend_physics_loss_tower(tower_net, batch_size, device):
    feat_ranges = {
        'Q_rej':      (0.,   2000., 'increase', 0),
        'T_dry':      (250., 300.,  'increase', 1),
        'T_wet':      (250., 308.,  'increase', 2),
        'T_approach': (0.1,  35.,   'decrease', 3),
        'n_towers':   (1.,   3.,    None,        4),
    }
    fixed_ranges = [(0., 2000.), (250., 300.), (250., 308.), (0.1, 35.), (1., 3.)]
    return _trend_loss(tower_net, feat_ranges, fixed_ranges, 10, device, batch_size)


def trend_physics_loss_chiller(chiller_net, batch_size, device):
    feat_ranges = {
        'Q_load': (10.,  2000., 'increase', 0),
        'T_chw':  (278., 286.,  'decrease', 1),
        'T_cdw':  (288., 303.,  'increase', 2),
        'V_chw':  (0.1,  1.0,   None,        3),
        'V_cdw':  (0.05, 1.0,   'decrease',  4),
    }
    fixed_ranges = [(10., 2000.), (278., 286.), (288., 303.), (0.1, 1.0), (0.05, 1.0)]
    return _trend_loss(chiller_net, feat_ranges, fixed_ranges, 10, device, batch_size)


class DecisionPINN(nn.Module):
    def __init__(self, V_max=2500.0):
        super().__init__()
        self.V_max = V_max
        self.policy_net = nn.Sequential(
            nn.Linear(12, 128), nn.ReLU(),   # 輸入從 9 → 12 維
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 9), nn.Sigmoid()
        )
        self.pump_spinn     = PumpSPINN()
        self.tower_tpinn    = TowerTPINN()
        self.chiller_tspinn = ChillerTSPINN()

    def freeze_physics(self):
        for m in [self.pump_spinn, self.tower_tpinn, self.chiller_tspinn]:
            for p in m.parameters(): p.requires_grad = False

    def unfreeze_physics(self):
        for m in [self.pump_spinn, self.tower_tpinn, self.chiller_tspinn]:
            for p in m.parameters(): p.requires_grad = True

    def forward(self, state):
        raw      = self.policy_net(state)          # state 現在是 12 維
        chl_sta  = state[:, 6:9]
        T_wet    = state[:, 1:2]
        T_sec_rw = state[:, 3:4]
        T_chw_set = 276.5 + raw[:, 0:1] * (284.3 - 276.5)
        T_chw_set = torch.min(T_chw_set, T_sec_rw - 2.0)
        T_chw_set = torch.clamp(T_chw_set, min=274.15, max=284.3)
        T_cdw_set = 288.7 + raw[:, 1:2] * (302.6 - 288.7)
        T_cdw_set = torch.max(T_cdw_set, T_wet + 2.0)
        ct_fan_spd   = raw[:, 3:6] * chl_sta
        chl_comp_spd = raw[:, 6:9] * chl_sta
        return torch.cat([T_chw_set, T_cdw_set, raw[:, 2:3],
                          ct_fan_spd, chl_comp_spd], dim=1)

    def physics_forward(self, state, action, detach_pump=True):
        T_dry    = state[:, 0:1]; T_wet    = state[:, 1:2]
        Q_load   = state[:, 2:3]; T_sec_rw = state[:, 3:4]
        V_sec    = state[:, 4:5]; chl_sta  = state[:, 6:9]
        # v8.5 新增流量和 λ
        v_cdw_norm  = state[:, 9:10]
        v_pri_norm  = state[:, 10:11]
        lam_sec_norm = state[:, 11:12]

        T_chw_set  = action[:, 0:1]; T_cdw_set  = action[:, 1:2]
        ct_fan_spd = action[:, 3:6]; comp_spd   = action[:, 6:9]

        V_norm        = V_sec / self.V_max
        num_running   = chl_sta.sum(dim=1, keepdim=True)
        num_sec_pumps = torch.clamp(num_running, max=2.0)

        # 三台泵各自計算（論文 Eq.12）
        p_cdw, p_pri, p_sec = self.pump_spinn(
            V_norm, v_cdw_norm, v_pri_norm,
            lam_sec_norm, num_running, num_sec_pumps)
        pump_power = p_cdw + p_pri + p_sec

        if detach_pump:
            pump_power = pump_power.detach()

        V_cdw       = ct_fan_spd.mean(dim=1, keepdim=True).clamp(min=0.05)
        cf          = torch.cat([Q_load, T_chw_set, T_cdw_set, V_norm, V_cdw], dim=1)
        chl_running = (comp_spd.sum(dim=1, keepdim=True) > 0.01).float()
        chiller_mu, chiller_sigma, delta_S = self.chiller_tspinn(cf)
        thermo_chiller_power = chiller_mu * chl_running
        mech_chiller_power   = (comp_spd**3).sum(dim=1, keepdim=True) * 300.0 * chl_running

        Q_rejection = Q_load + thermo_chiller_power
        T_approach  = F.relu(T_cdw_set - T_wet).clamp(min=0.1)
        n_towers    = chl_sta.sum(dim=1, keepdim=True)
        tf          = torch.cat([Q_rejection, T_dry, T_wet, T_approach, n_towers], dim=1)
        tower_mu, tower_sigma = self.tower_tpinn(tf)
        tower_power = tower_mu

        UA_tower                  = Q_rejection / (T_approach + 1e-6)
        heat_dissipation_capacity = UA_tower * T_approach
        Q_pred      = V_sec * 0.264 * F.relu(T_sec_rw - T_chw_set)
        total_power = thermo_chiller_power + pump_power + tower_power

        return {
            "total_power":               total_power,
            "thermo_chiller_power":      thermo_chiller_power,
            "mech_chiller_power":        mech_chiller_power,
            "chiller_sigma":             chiller_sigma,
            "pump_power":                pump_power,
            "pump_cdw":                  p_cdw,
            "pump_pri":                  p_pri,
            "pump_sec":                  p_sec,
            "tower_power":               tower_power,
            "tower_sigma":               tower_sigma,
            "Q_rejection":               Q_rejection,
            "heat_dissipation_capacity": heat_dissipation_capacity,
            "Q_pred":                    Q_pred,
            "T_cdw_set":                 T_cdw_set,
            "T_wet":                     T_wet,
            "V_norm":                    V_norm,
            "delta_S":                   delta_S,
            "UA_tower":                  UA_tower,
            "pump_lambda":               torch.tensor(0.0),
            "comp_lambda":               torch.tensor(300.0),
            "tower_lambda":              torch.tensor(1.0),
        }