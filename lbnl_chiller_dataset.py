"""
lbnl_chiller_dataset.py — v8

主要更新：
  __getitem__ 新增回傳 hist_pump_fixed 與 hist_pump_vfd，
  讓 train.py 能分別監督定速泵與變速泵。

  hist_pump_fixed = CDWL_PM_POW（3 台 CDW 泵）+ CWL_PRI_PM_POW（3 台主側泵）
  hist_pump_vfd   = CWL_SEC_PM_POW（2 台次側變速泵）
  hist_pump       = hist_pump_fixed + hist_pump_vfd（與 v7 相同的總泵功率）
"""

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset


def f_to_k(f_temp):
    """華氏轉開爾文 (Fahrenheit to Kelvin)"""
    return (f_temp - 32) * 5.0 / 9.0 + 273.15


class LBNLChillerDataset(Dataset):
    def __init__(self, csv_path):
        print(f"正在讀取 LBNL 資料: {csv_path} ...")
        df = pd.read_csv(csv_path)
        df = df.fillna(0.0)

        # ── 子系統功率分離（v8 新增）──────────────────────────────────────
        chiller_cols = [c for c in df.columns if 'CHL_POW_' in c]
        tower_cols   = [c for c in df.columns if 'CT_POW_' in c]

        # 定速泵：CDW 泵（冷凝側）+ Primary CHW 泵（主側，定速）
        cdw_pump_cols = [c for c in df.columns if 'CDWL_PM_POW_' in c]
        pri_pump_cols = [c for c in df.columns if 'CWL_PRI_PM_POW_' in c]
        # 變速泵：Secondary CHW 泵（次側，DP 控制）
        sec_pump_cols = [c for c in df.columns if 'CWL_SEC_PM_POW_' in c]

        df['chiller_power_kw']    = df[chiller_cols].sum(axis=1)
        df['tower_power_kw']      = df[tower_cols].sum(axis=1)
        df['pump_fixed_power_kw'] = (df[cdw_pump_cols].sum(axis=1) +
                                     df[pri_pump_cols].sum(axis=1))
        df['pump_vfd_power_kw']   = df[sec_pump_cols].sum(axis=1)
        df['pump_power_kw']       = (df['pump_fixed_power_kw'] +
                                     df['pump_vfd_power_kw'])
        df['total_power_kw']      = (df['chiller_power_kw'] +
                                     df['tower_power_kw'] +
                                     df['pump_power_kw'])

        def _t(col): return torch.tensor(df[col].values, dtype=torch.float32).unsqueeze(1)

        self.hist_chiller_power    = _t('chiller_power_kw')
        self.hist_tower_power      = _t('tower_power_kw')
        self.hist_pump_power       = _t('pump_power_kw')
        self.hist_pump_fixed_power = _t('pump_fixed_power_kw')
        self.hist_pump_vfd_power   = _t('pump_vfd_power_kw')
        self.hist_total_power      = _t('total_power_kw')

        ct_flow_total  = df['CDWL_CW_FLOW']
        ct_temp_diff   = f_to_k(df['CDWL_RW_TEMP']) - f_to_k(df['CDWL_SW_TEMP'])
        hist_q_rej     = ct_flow_total * 0.264 * ct_temp_diff.clip(lower=0.0)
        self.hist_q_rejection = torch.tensor(hist_q_rej.values,
                                             dtype=torch.float32).unsqueeze(1)

        # ── State 特徵（與 v7 相同，共 9 維）────────────────────────────
        t_dry    = torch.tensor(f_to_k(df['OA_TEMP']).values,
                                dtype=torch.float32).unsqueeze(1)
        t_wet    = torch.tensor(f_to_k(df['OA_TEMP_WB']).values,
                                dtype=torch.float32).unsqueeze(1)
        q_load   = torch.tensor((df['CWL_SEC_LOAD'] / 1000.0).values,
                                dtype=torch.float32).unsqueeze(1)
        t_sec_rw = torch.tensor(f_to_k(df['CWL_SEC_RW_TEMP']).values,
                                dtype=torch.float32).unsqueeze(1)

        v_sec_array = df['CWL_SEC_CW_FLOW'].values
        self.v_max  = float(np.percentile(v_sec_array, 95))
        if self.v_max < 10.0:
            self.v_max = 2500.0
        print(f"🌊 [System ID] 自動校準 V_max (P95): {self.v_max:.1f} GPM")

        v_sec_flow = torch.tensor(v_sec_array, dtype=torch.float32).unsqueeze(1)
        dp_sec     = torch.tensor(df['CWL_SEC_DP'].values,
                                  dtype=torch.float32).unsqueeze(1)

        # 統計差壓資訊，輔助設定 DP_SCALE
        dp_mean = float(df['CWL_SEC_DP'].mean())
        dp_p95  = float(np.percentile(df['CWL_SEC_DP'].values, 95))
        dp_max  = float(df['CWL_SEC_DP'].max())
        print(f"💧 [DP 統計] CWL_SEC_DP: mean={dp_mean:.1f}, P95={dp_p95:.1f}, "
              f"max={dp_max:.1f} inH2O")

        # 統計次側泵功率，方便驗證
        pm_fixed_mean = float(df['pump_fixed_power_kw'].mean())
        pm_vfd_mean   = float(df['pump_vfd_power_kw'].mean())
        pm_total_mean = float(df['pump_power_kw'].mean())
        print(f"🔧 [泵功率] 定速泵均值={pm_fixed_mean:.1f} kW, "
              f"變速泵均值={pm_vfd_mean:.1f} kW, 合計={pm_total_mean:.1f} kW")

        chl_sta_1 = torch.tensor(df['CHL_STA_1'].astype(float).values,
                                 dtype=torch.float32).unsqueeze(1)
        chl_sta_2 = torch.tensor(df['CHL_STA_2'].astype(float).values,
                                 dtype=torch.float32).unsqueeze(1)
        chl_sta_3 = torch.tensor(df['CHL_STA_3'].astype(float).values,
                                 dtype=torch.float32).unsqueeze(1)

        self.state = torch.cat([
            t_dry, t_wet, q_load, t_sec_rw, v_sec_flow, dp_sec,
            chl_sta_1, chl_sta_2, chl_sta_3
        ], dim=1)

        # ── Historical Action（與 v7 相同，共 9 維）─────────────────────
        self.hist_action = torch.cat([
            torch.tensor(f_to_k(df['CWL_PRI_SW_TEMPSPT']).values,
                         dtype=torch.float32).unsqueeze(1),
            torch.tensor(f_to_k(df['CT_SW_TEMPSPT']).values,
                         dtype=torch.float32).unsqueeze(1),
            torch.tensor(df['TWV_CTRL'].values,
                         dtype=torch.float32).unsqueeze(1),
            torch.tensor(df['CT_FAN_SPD_CTRL_1'].values,
                         dtype=torch.float32).unsqueeze(1),
            torch.tensor(df['CT_FAN_SPD_CTRL_2'].values,
                         dtype=torch.float32).unsqueeze(1),
            torch.tensor(df['CT_FAN_SPD_CTRL_3'].values,
                         dtype=torch.float32).unsqueeze(1),
            torch.tensor(df['CHL_COMP_SPD_CTRL_1'].values,
                         dtype=torch.float32).unsqueeze(1),
            torch.tensor(df['CHL_COMP_SPD_CTRL_2'].values,
                         dtype=torch.float32).unsqueeze(1),
            torch.tensor(df['CHL_COMP_SPD_CTRL_3'].values,
                         dtype=torch.float32).unsqueeze(1),
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
            self.hist_pump_power[idx],        # 總泵（與 v7 evaluate 相容）
            self.hist_pump_fixed_power[idx],  # v8 新增：定速泵標籤
            self.hist_pump_vfd_power[idx],    # v8 新增：變速泵標籤
        )