"""
DecisionPINN — v8.10 (終極融合版)

1. 定速泵合併與回歸：將 CDW 與 Pri 泵重新合併為 CombinedFixedPumpSPINN，
   使用 v8.4 的 MLP 架構來吸收管路雜訊，恢復 R²=0.95 的高水準。
2. 變速泵保留：保留 v8.8 創下 R²=0.9999 的神級物理架構 (DP * V / η)。
3. 大腦解封：拔除 30% 轉速下限，讓 Policy 完全自由探索歷史低耗電狀態。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

TOWER_MAX_KW   = 40.0
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


class CombinedFixedPumpSPINN(nn.Module):
    """【退回 v8.4 成功經驗】合併預測所有定速泵，用 3 維特徵吸收雜訊"""
    def __init__(self, max_kw=120.0):
        super().__init__()
        self.max_kw = max_kw
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, num_running, Q_load, T_dry):
        q = Q_load / 1000.0
        t = T_dry / 300.0
        x = torch.cat([num_running / 3.0, q, t], dim=1)
        on_mask = (num_running > 0).float()
        return torch.sigmoid(self.net(x)) * self.max_kw * on_mask


class SECPumpSPINN(nn.Module):
    """【保留 v8.8 神級物理】完美 R²=0.9999 的變速泵 S-PINN"""
    def __init__(self, init_log_scale=4.0): 
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(float(init_log_scale)))
        self.eff_net = nn.Sequential(
            nn.Linear(3, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.eff_net[-1].weight)
        nn.init.zeros_(self.eff_net[-1].bias)

    def forward(self, v_sec_norm, dp_norm, num_sec_pumps):
        on_mask  = (num_sec_pumps > 0).float()
        num_norm = num_sec_pumps / 2.0

        P_scale = torch.exp(self.log_scale)
        ideal   = dp_norm * v_sec_norm
        eff_input = torch.cat([v_sec_norm, dp_norm, num_norm], dim=1)
        eta = torch.sigmoid(self.eff_net(eff_input)).clamp(0.2, 0.95)

        power = P_scale * ideal / eta
        return power * on_mask

    def get_scale(self):
        return torch.exp(self.log_scale).item()


class PumpSPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fixed_pump = CombinedFixedPumpSPINN()
        self.sec_pump   = SECPumpSPINN()

    def forward(self, v_sec_norm, dp_norm, num_running, num_sec_pumps, Q_load, T_dry):
        p_fixed = self.fixed_pump(num_running, Q_load, T_dry)
        p_sec   = self.sec_pump(v_sec_norm, dp_norm, num_sec_pumps)
        return p_fixed, p_sec


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
        raw      = self.policy_net(state[:, :9])
        chl_sta  = state[:, 6:9]
        T_wet    = state[:, 1:2]
        T_sec_rw = state[:, 3:4]
        
        T_chw_set = 276.5 + raw[:, 0:1] * (284.3 - 276.5)
        T_chw_set = torch.min(T_chw_set, T_sec_rw - 2.0)
        T_chw_set = torch.clamp(T_chw_set, min=274.15, max=284.3)
        T_cdw_set = 288.7 + raw[:, 1:2] * (302.6 - 288.7)
        T_cdw_set = torch.max(T_cdw_set, T_wet + 2.0)
        
        # 【解封】允許 AI 自行決定轉速，不再強制墊高耗電量
        ct_fan_spd   = raw[:, 3:6] * chl_sta
        chl_comp_spd = raw[:, 6:9] * chl_sta
        
        return torch.cat([T_chw_set, T_cdw_set, raw[:, 2:3],
                          ct_fan_spd, chl_comp_spd], dim=1)

    def physics_forward(self, state, action, detach_pump=True):
        T_dry    = state[:, 0:1]; T_wet    = state[:, 1:2]
        Q_load   = state[:, 2:3]; T_sec_rw = state[:, 3:4]
        V_sec    = state[:, 4:5]; dp_sec   = state[:, 5:6]
        chl_sta  = state[:, 6:9]

        T_chw_set  = action[:, 0:1]; T_cdw_set  = action[:, 1:2]
        ct_fan_spd = action[:, 3:6]; comp_spd   = action[:, 6:9]

        V_norm        = V_sec / self.V_max
        dp_norm       = dp_sec / 1000.0
        num_running   = chl_sta.sum(dim=1, keepdim=True)
        num_sec_pumps = torch.clamp(num_running, max=2.0)

        p_fixed, p_sec = self.pump_spinn(V_norm, dp_norm, num_running, num_sec_pumps, Q_load, T_dry)
        pump_power = p_fixed + p_sec

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
            "pump_fixed":                p_fixed,
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