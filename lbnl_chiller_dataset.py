"""
lbnl_chiller_dataset.py — v8.5

新增 v8.5 的物理泵模型所需欄位：
  1. V_cdw_norm：冷凝水流量歸一化（CDWL_CW_FLOW），供 CDW 定速泵 S-PINN 使用
  2. V_pri_norm：主側冷水流量歸一化（CWL_PRI_CW_FLOW），供 Primary 定速泵 S-PINN 使用
  3. lam_sec：次側泵管路阻力係數 λ = CWL_SEC_DP / CWL_SEC_CW_FLOW²
              論文公式：H = λV²，開機時即時計算，關機時設為 0
  4. 分離 CDW 泵功率 / Primary 泵功率 / Secondary 泵功率（各自獨立監督）

State 擴充：9維 → 12維
  原 9 維：[T_dry, T_wet, Q_load, T_sec_rw, V_sec, dp_sec, CHL_STA×3]
  新增 3 維：[V_cdw_norm, V_pri_norm, lam_sec_norm]
"""

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset


def f_to_k(f_temp):
    return (f_temp - 32) * 5.0 / 9.0 + 273.15


class LBNLChillerDataset(Dataset):
    def __init__(self, csv_path):
        print(f"正在讀取 LBNL 資料: {csv_path} ...")
        df = pd.read_csv(csv_path)
        df = df.fillna(0.0)

        # ── 功率欄位 ──────────────────────────────────────────────────────
        chiller_cols  = [c for c in df.columns if 'CHL_POW_' in c]
        tower_cols    = [c for c in df.columns if 'CT_POW_' in c]
        cdw_pump_cols = [c for c in df.columns if 'CDWL_PM_POW_' in c]
        pri_pump_cols = [c for c in df.columns if 'CWL_PRI_PM_POW_' in c]
        sec_pump_cols = [c for c in df.columns if 'CWL_SEC_PM_POW_' in c]

        df['chiller_power_kw']    = df[chiller_cols].sum(axis=1)
        df['tower_power_kw']      = df[tower_cols].sum(axis=1)
        df['pump_cdw_power_kw']   = df[cdw_pump_cols].sum(axis=1)   # CDW 泵
        df['pump_pri_power_kw']   = df[pri_pump_cols].sum(axis=1)   # Primary 泵
        df['pump_fixed_power_kw'] = (df['pump_cdw_power_kw'] +
                                     df['pump_pri_power_kw'])
        df['pump_vfd_power_kw']   = df[sec_pump_cols].sum(axis=1)   # Secondary 泵
        df['pump_power_kw']       = (df['pump_fixed_power_kw'] +
                                     df['pump_vfd_power_kw'])
        df['total_power_kw']      = (df['chiller_power_kw'] +
                                     df['tower_power_kw'] +
                                     df['pump_power_kw'])

        def _t(col):
            return torch.tensor(df[col].values, dtype=torch.float32).unsqueeze(1)

        self.hist_chiller_power    = _t('chiller_power_kw')
        self.hist_tower_power      = _t('tower_power_kw')
        self.hist_pump_power       = _t('pump_power_kw')
        self.hist_pump_fixed_power = _t('pump_fixed_power_kw')
        self.hist_pump_cdw_power   = _t('pump_cdw_power_kw')    # v8.5 新增
        self.hist_pump_pri_power   = _t('pump_pri_power_kw')    # v8.5 新增
        self.hist_pump_vfd_power   = _t('pump_vfd_power_kw')
        self.hist_total_power      = _t('total_power_kw')

        ct_flow_total = df['CDWL_CW_FLOW']
        ct_temp_diff  = f_to_k(df['CDWL_RW_TEMP']) - f_to_k(df['CDWL_SW_TEMP'])
        hist_q_rej    = ct_flow_total * 0.264 * ct_temp_diff.clip(lower=0.0)
        self.hist_q_rejection = torch.tensor(
            hist_q_rej.values, dtype=torch.float32).unsqueeze(1)

        # ── 流量歸一化基準 ────────────────────────────────────────────────
        v_sec_arr = df['CWL_SEC_CW_FLOW'].values
        v_cdw_arr = df['CDWL_CW_FLOW'].values
        v_pri_arr = df['CWL_PRI_CW_FLOW'].values

        self.v_max     = float(np.percentile(v_sec_arr, 95))
        self.v_cdw_max = float(np.percentile(v_cdw_arr[v_cdw_arr > 1], 95))
        self.v_pri_max = float(np.percentile(v_pri_arr[v_pri_arr > 1], 95))
        if self.v_max < 10.0:     self.v_max     = 2500.0
        if self.v_cdw_max < 1.0:  self.v_cdw_max = 1000.0
        if self.v_pri_max < 1.0:  self.v_pri_max = 500.0
        print(f"🌊 V_max(sec)={self.v_max:.1f} V_max(cdw)={self.v_cdw_max:.1f} "
              f"V_max(pri)={self.v_pri_max:.1f} GPM")

        # ── λ_sec：次側泵管路阻力係數（論文公式 H = λV²）────────────────
        # 單位：inH2O / GPM²，在開機（V>1 GPM）時計算，關機時設 0
        dp_sec_arr  = df['CWL_SEC_DP'].values
        v_sec_safe  = np.where(v_sec_arr > 1.0, v_sec_arr, np.nan)
        lam_sec_raw = dp_sec_arr / (v_sec_safe ** 2)   # inH2O/GPM²
        # 統計穩態λ（排除啟停暫態，取 P5~P95 範圍）
        lam_valid   = lam_sec_raw[~np.isnan(lam_sec_raw)]
        lam_p5      = float(np.percentile(lam_valid, 5))
        lam_p95     = float(np.percentile(lam_valid, 95))
        lam_median  = float(np.median(lam_valid))
        # 截斷極端值後歸一化：lam_norm = lam / lam_p95
        self.lam_sec_scale = lam_p95
        lam_sec_clipped    = np.clip(lam_sec_raw, 0.0, lam_p95 * 2)
        lam_sec_filled     = np.where(np.isnan(lam_sec_clipped), 0.0, lam_sec_clipped)
        print(f"🔧 λ_sec: median={lam_median:.6f}, P5={lam_p5:.6f}, "
              f"P95={lam_p95:.6f} inH2O/GPM²  scale={self.lam_sec_scale:.6f}")

        dp_mean = float(df['CWL_SEC_DP'].mean())
        print(f"💧 CWL_SEC_DP: mean={dp_mean:.1f} inH2O")
        pm_fixed = float(df['pump_fixed_power_kw'].mean())
        pm_cdw   = float(df['pump_cdw_power_kw'].mean())
        pm_pri   = float(df['pump_pri_power_kw'].mean())
        pm_vfd   = float(df['pump_vfd_power_kw'].mean())
        print(f"🔧 泵功率: CDW={pm_cdw:.1f} Pri={pm_pri:.1f} "
              f"Fixed合計={pm_fixed:.1f} VFD={pm_vfd:.1f} kW")

        # ── State（12維）─────────────────────────────────────────────────
        t_dry    = torch.tensor(f_to_k(df['OA_TEMP']).values,
                                dtype=torch.float32).unsqueeze(1)
        t_wet    = torch.tensor(f_to_k(df['OA_TEMP_WB']).values,
                                dtype=torch.float32).unsqueeze(1)
        q_load   = torch.tensor((df['CWL_SEC_LOAD'] / 1000.0).values,
                                dtype=torch.float32).unsqueeze(1)
        t_sec_rw = torch.tensor(f_to_k(df['CWL_SEC_RW_TEMP']).values,
                                dtype=torch.float32).unsqueeze(1)
        v_sec    = torch.tensor(v_sec_arr, dtype=torch.float32).unsqueeze(1)
        dp_sec   = torch.tensor(dp_sec_arr, dtype=torch.float32).unsqueeze(1)
        chl_sta  = [torch.tensor(df[f'CHL_STA_{i}'].astype(float).values,
                                 dtype=torch.float32).unsqueeze(1) for i in [1,2,3]]
        # 新增 3 維
        v_cdw_norm  = torch.tensor(
            v_cdw_arr / self.v_cdw_max, dtype=torch.float32).unsqueeze(1)
        v_pri_norm  = torch.tensor(
            v_pri_arr / self.v_pri_max, dtype=torch.float32).unsqueeze(1)
        lam_sec_norm = torch.tensor(
            lam_sec_filled / self.lam_sec_scale, dtype=torch.float32).unsqueeze(1)

        self.state = torch.cat([
            t_dry, t_wet, q_load, t_sec_rw, v_sec, dp_sec,
            *chl_sta,
            v_cdw_norm, v_pri_norm, lam_sec_norm   # idx 9,10,11
        ], dim=1)

        # ── Historical Action（9維，不變）─────────────────────────────────
        self.hist_action = torch.cat([
            torch.tensor(f_to_k(df['CWL_PRI_SW_TEMPSPT']).values,
                         dtype=torch.float32).unsqueeze(1),
            torch.tensor(f_to_k(df['CT_SW_TEMPSPT']).values,
                         dtype=torch.float32).unsqueeze(1),
            torch.tensor(df['TWV_CTRL'].values,
                         dtype=torch.float32).unsqueeze(1),
            *[torch.tensor(df[f'CT_FAN_SPD_CTRL_{i}'].values,
                           dtype=torch.float32).unsqueeze(1) for i in [1,2,3]],
            *[torch.tensor(df[f'CHL_COMP_SPD_CTRL_{i}'].values,
                           dtype=torch.float32).unsqueeze(1) for i in [1,2,3]],
        ], dim=1)

        print(f"✅ 資料載入完成！State: {self.state.shape}, "
              f"Action: {self.hist_action.shape}")

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        return (
            self.state[idx],
            self.hist_action[idx],
            self.hist_total_power[idx],
            self.hist_q_rejection[idx],
            self.hist_chiller_power[idx],
            self.hist_tower_power[idx],
            self.hist_pump_power[idx],
            self.hist_pump_fixed_power[idx],
            self.hist_pump_cdw_power[idx],    # v8.5 新增
            self.hist_pump_pri_power[idx],    # v8.5 新增
            self.hist_pump_vfd_power[idx],
        )