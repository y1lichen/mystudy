"""evaluate_fair.py — v8.10"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from torch.utils.data import DataLoader
from lbnl_chiller_dataset import LBNLChillerDataset
from decision_pinn import DecisionPINN

def safe_mape(y_true, y_pred, threshold=5.0):
    mask = y_true > threshold
    if np.sum(mask) == 0:
        return 0.0
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask])

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 評估模式啟動！使用的運算設備: {device}")
    dataset    = LBNLChillerDataset("data/ChillerPlant_test.csv")
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    model = DecisionPINN(V_max=dataset.v_max).to(device)
    model.load_state_dict(torch.load("decision_pinn.pth", map_location=device,
                                     weights_only=True))
    model.eval()

    sc_sec  = model.pump_spinn.sec_pump.get_scale()
    
    print(f"  [學習到的物理參數]")
    print(f"  P_scale_sec (變速泵) = {sc_sec:.2f} kW")

    all_hist_power, all_pred_hist_power   = [], []
    all_hist_chiller, all_pred_chiller    = [], []
    all_hist_tower, all_pred_tower        = [], []
    all_hist_pump, all_pred_pump          = [], []
    all_hist_pump_fixed, all_pred_pump_fixed = [], []
    all_hist_pump_vfd,  all_pred_pump_vfd  = [], []
    all_delta_s, all_ua_tower             = [], []
    all_opt_power, all_q_load, all_q_pred = [], [], []
    all_t_cdw, all_t_wet                  = [], []
    comp_power, tower_power, pump_power   = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            (state, hist_action, hist_total_power, hist_q_rejection,
             hist_chiller_power, hist_tower_power, hist_pump_power,
             hist_pump_fixed, hist_pump_cdw, hist_pump_pri, hist_pump_vfd) = batch
            state       = state.to(device)
            hist_action = hist_action.to(device)

            hist_phys  = model.physics_forward(state, hist_action)
            opt_action = model(state)
            opt_phys   = model.physics_forward(state, opt_action)

            Q_load        = state[:, 2:3]
            T_dry         = state[:, 0:1]
            V_sec         = state[:, 4:5]
            dp_sec        = state[:, 5:6]
            chl_sta       = state[:, 6:9]
            
            V_norm        = V_sec / dataset.v_max
            dp_norm       = dp_sec / 1000.0
            num_running   = chl_sta.sum(dim=1, keepdim=True)
            num_sec_pumps = torch.clamp(num_running, max=2.0)
            
            p_fixed, p_sec = model.pump_spinn(V_norm, dp_norm, num_running, num_sec_pumps, Q_load, T_dry)

            all_hist_power.extend(hist_total_power.cpu().numpy())
            all_pred_hist_power.extend(hist_phys["total_power"].cpu().numpy())
            all_hist_chiller.extend(hist_chiller_power.cpu().numpy())
            all_pred_chiller.extend(hist_phys["thermo_chiller_power"].cpu().numpy())
            all_hist_tower.extend(hist_tower_power.cpu().numpy())
            all_pred_tower.extend(hist_phys["tower_power"].cpu().numpy())
            all_hist_pump.extend(hist_pump_power.cpu().numpy())
            all_pred_pump.extend(hist_phys["pump_power"].cpu().numpy())
            
            all_hist_pump_fixed.extend(hist_pump_fixed.cpu().numpy())
            all_pred_pump_fixed.extend(p_fixed.cpu().numpy())
            all_hist_pump_vfd.extend(hist_pump_vfd.cpu().numpy())
            all_pred_pump_vfd.extend(p_sec.cpu().numpy())
            
            all_delta_s.extend(hist_phys["delta_S"].cpu().numpy())
            all_ua_tower.extend(hist_phys["UA_tower"].cpu().numpy())
            all_opt_power.extend(opt_phys["total_power"].cpu().numpy())
            all_q_load.extend(state[:, 2:3].cpu().numpy())
            all_q_pred.extend(opt_phys["Q_pred"].cpu().numpy())
            all_t_cdw.extend(opt_phys["T_cdw_set"].cpu().numpy())
            all_t_wet.extend(opt_phys["T_wet"].cpu().numpy())
            comp_power.extend(opt_phys["thermo_chiller_power"].cpu().numpy())
            tower_power.extend(opt_phys["tower_power"].cpu().numpy())
            pump_power.extend(opt_phys["pump_power"].cpu().numpy())

    def _np(lst): return np.array(lst).flatten()
    all_hist_power      = _np(all_hist_power)
    all_pred_hist_power = _np(all_pred_hist_power)
    all_hist_chiller    = _np(all_hist_chiller)
    all_pred_chiller    = _np(all_pred_chiller)
    all_hist_tower      = _np(all_hist_tower)
    all_pred_tower      = _np(all_pred_tower)
    all_hist_pump       = _np(all_hist_pump)
    all_pred_pump       = _np(all_pred_pump)
    all_hist_pump_fixed = _np(all_hist_pump_fixed)
    all_pred_pump_fixed = _np(all_pred_pump_fixed)
    all_hist_pump_vfd   = _np(all_hist_pump_vfd)
    all_pred_pump_vfd   = _np(all_pred_pump_vfd)
    all_opt_power       = _np(all_opt_power)

    print("\n🥇 [Metric 1] System ID Sanity Check (Modular)")
    print(f"  [全系統] R²={r2_score(all_hist_power,all_pred_hist_power):.4f} | "
          f"開機 MAPE={safe_mape(all_hist_power,all_pred_hist_power)*100:.2f}%")
    print(f"  [主  機] R²={r2_score(all_hist_chiller,all_pred_chiller):.4f} | "
          f"開機 MAPE={safe_mape(all_hist_chiller,all_pred_chiller)*100:.2f}%")
    print(f"  [水  塔] R²={r2_score(all_hist_tower,all_pred_tower):.4f} | "
          f"開機 MAPE={safe_mape(all_hist_tower,all_pred_tower)*100:.2f}%")
    print(f"  [水泵合計] R²={r2_score(all_hist_pump,all_pred_pump):.4f} | "
          f"開機 MAPE={safe_mape(all_hist_pump,all_pred_pump)*100:.2f}%")
    print(f"    ├─ [定速泵] R²={r2_score(all_hist_pump_fixed,all_pred_pump_fixed):.4f} | "
          f"開機 MAPE={safe_mape(all_hist_pump_fixed,all_pred_pump_fixed)*100:.2f}%")
    print(f"    └─ [次側泵] R²={r2_score(all_hist_pump_vfd,all_pred_pump_vfd):.4f} | "
          f"開機 MAPE={safe_mape(all_hist_pump_vfd,all_pred_pump_vfd)*100:.2f}%")
    print(f"  - 動態物理參數: ΔS={np.mean(all_delta_s):.4f}, UA={np.mean(all_ua_tower):.2f}")

    total_real   = np.sum(all_hist_power)
    total_pred   = np.sum(all_pred_hist_power)
    total_opt    = np.sum(all_opt_power)
    fair_savings = (total_pred - total_opt) / total_pred * 100
    print("\n🥈 [Metric 2] Energy Savings Performance")
    print(f"  - 歷史真實耗電 (量測值): {total_real:.0f} kW")
    print(f"  - 模型歷史耗電 (基準值): {total_pred:.0f} kW")
    print(f"  - AI 最佳化耗電 (預測值): {total_opt:.0f} kW")
    print(f"  - 修正後的節能比例: {fair_savings:.2f}%")

    avg_comp  = np.mean(comp_power)
    avg_tower = np.mean(tower_power)
    avg_pump  = np.mean(pump_power)
    plt.figure(figsize=(8, 6))
    plt.bar(['Optimized Policy'], [avg_comp],  label='Chiller Compressor')
    plt.bar(['Optimized Policy'], [avg_tower], bottom=[avg_comp], label='Cooling Tower')
    plt.bar(['Optimized Policy'], [avg_pump],  bottom=[avg_comp+avg_tower], label='Water Pump')
    plt.ylabel('Average Power (kW)'); plt.title('AI Optimal Resource Allocation')
    plt.legend(); plt.savefig('energy_breakdown.png')
    print("  -> 已儲存: energy_breakdown.png")

    load_satisfied = np.sum(_np(all_q_pred) >= _np(all_q_load)) / len(all_q_load)
    thermo_safe    = np.sum(_np(all_t_cdw) >= _np(all_t_wet) + 1.9) / len(all_t_cdw)
    print("\n🥉 [Metric 3] Constraint Satisfaction")
    print(f"  - 負載滿足率 (Q_pred >= Q_load): {load_satisfied*100:.2f}%")
    print(f"  - 熱力學安全率 (T_cdw > T_wet): {thermo_safe*100:.2f}%")

if __name__ == "__main__":
    evaluate_model()