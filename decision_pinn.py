"""DecisionPINN — v8.4

v8.3 根本問題修正：

【問題 1】VFD 使用 ct_fan_avg 作為代理輸入在邏輯上錯誤
  次側泵（Secondary CHW Pump）的控制信號是 CWL_SEC_PM_SPD（次側泵速）
  policy 的 action 完全不包含次側泵速，用 ct_fan 作代理造成梯度衝突
  修正：VFD 只用 V_norm（流量）作為輸入，去掉 ct_fan_avg
        V_norm 是 state，與 action 無關，VFD 功率對 policy 無梯度 → 直接 detach

【問題 2】training vs evaluate 定義不一致（detach_pump 訓練時固定，評估時活動）
  training: total = chiller + pump.detach() + tower
  evaluate: total = chiller + pump(ct_fan變化) + tower → pump 暴增
  修正：VFD 輸入與 action 完全解耦後，detach_pump 在 evaluate 也是安全的
        physics_forward 保留 detach_pump 參數，evaluate 也傳 True

【問題 3】定速泵 R²=0.037（線性模型仍無法捕捉時間變異）
  定速泵功率雖然「定速」但因效率、溫度等因素有時間變異
  用更豐富的特徵：num_running + Q_load（負荷代理）+ T_dry（環境溫度代理）
  改回 MLP，但不用 LayerNorm，改用 BatchNorm（批次內不同樣本可以歸一化）
  這樣可以捕捉到時間變異，同時避免 LayerNorm 的梯度消失問題

【定速泵模型選擇】
  LayerNorm 在 Linear(1→32) 後梯度消失 → 之前的問題
  BatchNorm(32) 對 batch 內不同樣本歸一化 → 批次內 num_running 有 0/1/2/3 多種值 → 正常
  輸入：[num_running/3, Q_load/1000, T_dry/300]（3維，有時間變異信息）
  輸出：Sigmoid * MAX_FIXED_KW（上界保護）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

TOWER_MAX_KW     = 40.0
SEC_PUMP_MAX_KW  = 80.0
MAX_FIXED_KW     = 120.0   # 3台定速泵上界（3 × ~35 kW）

_TOWER_SCALE    = torch.tensor([500.0, 290.0, 290.0, 15.0, 2.0])
_CHILLER_SCALE  = torch.tensor([300.0, 283.0, 295.0,  1.0, 1.0])


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


class FixedSpeedPumpModel(nn.Module):
    """
    定速泵模型 v8.4：3維輸入 MLP + BatchNorm（避免 LayerNorm 梯度消失）

    輸入：[num_running/3, Q_load_norm, T_dry_norm]
      num_running：主要驅動因素（台數）
      Q_load_norm：負荷代理，捕捉泵在高負荷時效率變化
      T_dry_norm ：環境溫度代理，捕捉冷凝側溫度對泵功率的影響
    輸出：Sigmoid * MAX_FIXED_KW（上界 120 kW = 3台 × 40 kW）

    BatchNorm 在批次內對不同 num_running 的樣本做歸一化，
    解決 LayerNorm 在同值輸入時 std=0 的梯度消失問題。
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, num_running, Q_load=None, T_dry=None):
        """
        num_running: (B,1) float {0,1,2,3}
        Q_load:      (B,1) kW（可選，None 時用零）
        T_dry:       (B,1) K（可選，None 時用零）
        """
        B = num_running.shape[0]
        dev = num_running.device
        q = Q_load / 1000.0 if Q_load is not None else torch.zeros(B, 1, device=dev)
        t = T_dry / 300.0   if T_dry  is not None else torch.zeros(B, 1, device=dev)
        x = torch.cat([num_running / 3.0, q, t], dim=1)
        on_mask = (num_running > 0).float()
        return torch.sigmoid(self.net(x)) * MAX_FIXED_KW * on_mask


class VFDPumpSPINN(nn.Module):
    """
    次側變速泵 v8.4：只用 V_norm（state），完全與 action 解耦。

    次側泵由 DP 控制器驅動，控制信號是 CWL_SEC_PM_SPD（不在 policy action 中）。
    因此 VFD 功率是 state 的函數，對 policy 無梯度，需配合 detach_pump 使用。

    輸入：[V_norm, num_sec_pumps/2]
    輸出：sigmoid * MAX * on_mask
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, V_norm, num_sec_pumps):
        on_mask = (num_sec_pumps > 0).float()
        x       = torch.cat([V_norm, num_sec_pumps / 2.0], dim=1)
        return torch.sigmoid(self.net(x)) * SEC_PUMP_MAX_KW * on_mask


class PumpSPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fixed_pump = FixedSpeedPumpModel()
        self.vfd_pump   = VFDPumpSPINN()

    def forward(self, V_norm, num_running, num_sec_pumps, Q_load=None, T_dry=None):
        return (self.fixed_pump(num_running, Q_load, T_dry) +
                self.vfd_pump(V_norm, num_sec_pumps))


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
            nn.Linear(9, 128), nn.ReLU(),
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
        raw      = self.policy_net(state)
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
        """
        detach_pump=True（預設）：泵功率 stop-gradient。
        因為泵的控制變量（次側泵速）不在 policy action 中，
        泵功率本質上是 state 的函數，對 policy 梯度無意義。
        training 和 evaluate 都用 True，保持一致。
        """
        T_dry    = state[:, 0:1]; T_wet    = state[:, 1:2]
        Q_load   = state[:, 2:3]; T_sec_rw = state[:, 3:4]
        V_sec    = state[:, 4:5]; chl_sta  = state[:, 6:9]
        T_chw_set  = action[:, 0:1]; T_cdw_set  = action[:, 1:2]
        ct_fan_spd = action[:, 3:6]; comp_spd   = action[:, 6:9]

        V_norm        = V_sec / self.V_max
        num_running   = chl_sta.sum(dim=1, keepdim=True)
        num_sec_pumps = torch.clamp(num_running, max=2.0)

        # 泵功率：只用 state（V_norm, num_running, Q_load, T_dry），與 action 無關
        pump_power = self.pump_spinn(V_norm, num_running, num_sec_pumps, Q_load, T_dry)
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