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

        # 分離並萃取各子系統歷史耗電 (kW)
        chiller_cols = [c for c in df.columns if 'CHL_POW_' in c]
        tower_cols = [c for c in df.columns if 'CT_POW_' in c]
        pump_cols = [c for c in df.columns if 'CDWL_PM_POW_' in c or 'CWL_PRI_PM_POW_' in c or 'CWL_SEC_PM_POW_' in c]

        df['chiller_power_kw'] = df[chiller_cols].sum(axis=1)
        df['tower_power_kw'] = df[tower_cols].sum(axis=1)
        df['pump_power_kw'] = df[pump_cols].sum(axis=1)
        df['total_power_kw'] = df['chiller_power_kw'] + df['tower_power_kw'] + df['pump_power_kw']

        self.hist_chiller_power = torch.tensor(df['chiller_power_kw'].values, dtype=torch.float32).unsqueeze(1)
        self.hist_tower_power = torch.tensor(df['tower_power_kw'].values, dtype=torch.float32).unsqueeze(1)
        self.hist_pump_power = torch.tensor(df['pump_power_kw'].values, dtype=torch.float32).unsqueeze(1)
        self.hist_total_power = torch.tensor(df['total_power_kw'].values, dtype=torch.float32).unsqueeze(1)
        
        ct_flow_total = df['CDWL_CW_FLOW']
        ct_temp_diff = f_to_k(df['CDWL_RW_TEMP']) - f_to_k(df['CDWL_SW_TEMP'])
        hist_q_rej = ct_flow_total * 0.264 * ct_temp_diff.clip(lower=0.0)
        self.hist_q_rejection = torch.tensor(hist_q_rej.values, dtype=torch.float32).unsqueeze(1)

        t_dry = torch.tensor(f_to_k(df['OA_TEMP']).values, dtype=torch.float32).unsqueeze(1)
        t_wet = torch.tensor(f_to_k(df['OA_TEMP_WB']).values, dtype=torch.float32).unsqueeze(1)
        q_load = torch.tensor((df['CWL_SEC_LOAD'] / 1000.0).values, dtype=torch.float32).unsqueeze(1)
        t_sec_rw = torch.tensor(f_to_k(df['CWL_SEC_RW_TEMP']).values, dtype=torch.float32).unsqueeze(1)
        
        v_sec_flow_array = df['CWL_SEC_CW_FLOW'].values
        self.v_max = float(np.percentile(v_sec_flow_array, 95))
        if self.v_max < 10.0:  
            self.v_max = 2500.0
        print(f"🌊 [System ID] 自動校準 V_max (P95流量極限): {self.v_max:.1f} GPM")
            
        v_sec_flow = torch.tensor(v_sec_flow_array, dtype=torch.float32).unsqueeze(1)
        dp_sec = torch.tensor(df['CWL_SEC_DP'].values, dtype=torch.float32).unsqueeze(1)
        chl_sta_1 = torch.tensor(df['CHL_STA_1'].astype(float).values, dtype=torch.float32).unsqueeze(1)
        chl_sta_2 = torch.tensor(df['CHL_STA_2'].astype(float).values, dtype=torch.float32).unsqueeze(1)
        chl_sta_3 = torch.tensor(df['CHL_STA_3'].astype(float).values, dtype=torch.float32).unsqueeze(1)
        
        self.state = torch.cat([
            t_dry, t_wet, q_load, t_sec_rw, v_sec_flow, dp_sec, 
            chl_sta_1, chl_sta_2, chl_sta_3
        ], dim=1)
        
        hist_t_chw_set = torch.tensor(f_to_k(df['CWL_PRI_SW_TEMPSPT']).values, dtype=torch.float32).unsqueeze(1)
        hist_t_cdw_set = torch.tensor(f_to_k(df['CT_SW_TEMPSPT']).values, dtype=torch.float32).unsqueeze(1)
        hist_twv_ctrl = torch.tensor(df['TWV_CTRL'].values, dtype=torch.float32).unsqueeze(1)
        hist_ct_fan_1 = torch.tensor(df['CT_FAN_SPD_CTRL_1'].values, dtype=torch.float32).unsqueeze(1)
        hist_ct_fan_2 = torch.tensor(df['CT_FAN_SPD_CTRL_2'].values, dtype=torch.float32).unsqueeze(1)
        hist_ct_fan_3 = torch.tensor(df['CT_FAN_SPD_CTRL_3'].values, dtype=torch.float32).unsqueeze(1)
        hist_comp_1 = torch.tensor(df['CHL_COMP_SPD_CTRL_1'].values, dtype=torch.float32).unsqueeze(1)
        hist_comp_2 = torch.tensor(df['CHL_COMP_SPD_CTRL_2'].values, dtype=torch.float32).unsqueeze(1)
        hist_comp_3 = torch.tensor(df['CHL_COMP_SPD_CTRL_3'].values, dtype=torch.float32).unsqueeze(1)

        self.hist_action = torch.cat([
            hist_t_chw_set, hist_t_cdw_set, hist_twv_ctrl,
            hist_ct_fan_1, hist_ct_fan_2, hist_ct_fan_3,
            hist_comp_1, hist_comp_2, hist_comp_3
        ], dim=1)
        
        print(f"✅ 資料載入完成！State 維度: {self.state.shape}, Action 維度: {self.hist_action.shape}")

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        return (
            self.state[idx], self.hist_action[idx], 
            self.hist_total_power[idx], self.hist_q_rejection[idx],
            self.hist_chiller_power[idx], self.hist_tower_power[idx], self.hist_pump_power[idx]
        )