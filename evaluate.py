import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from torch.utils.data import DataLoader
from lbnl_chiller_dataset import LBNLChillerDataset
from decision_pinn import DecisionPINN

# 🔥 新增：安全計算開機時段 MAPE 的函數，避免被停機狀態 (0 kW) 的除以零無限大拉爆
def safe_mape(y_true, y_pred, threshold=5.0):
    mask = y_true > threshold
    if np.sum(mask) == 0:
        return 0.0
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask])

def evaluate_model():
    # 自動偵測設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 評估模式啟動！使用的運算設備: {device}")

    dataset = LBNLChillerDataset("data/ChillerPlant_test.csv") 
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    # 模型放上 GPU，並傳入正確的 V_max
    model = DecisionPINN(V_max=dataset.v_max).to(device)
    model.load_state_dict(torch.load("decision_pinn.pth", map_location=device, weights_only=True))
    model.eval()
    
    # 追蹤總耗電與子設備耗電
    all_hist_power, all_pred_hist_power = [], []
    all_hist_chiller, all_pred_chiller = [], []
    all_hist_tower, all_pred_tower = [], []
    all_hist_pump, all_pred_pump = [], []
    
    # 追蹤動態物理參數
    all_delta_s, all_ua_tower = [], []
    
    # 追蹤 AI 最佳化結果
    all_opt_power, all_q_load, all_q_pred = [], [], []
    all_t_cdw, all_t_wet = [], []
    comp_power, tower_power, pump_power = [], [], []

    with torch.no_grad():
        for state, hist_action, hist_total_power, hist_q_rejection, hist_chiller_power, hist_tower_power, hist_pump_power in dataloader:
            state = state.to(device)
            hist_action = hist_action.to(device)
            
            hist_phys = model.physics_forward(state, hist_action)
            opt_action = model(state)
            opt_phys = model.physics_forward(state, opt_action)
            
            # --- 收集真實與預測耗電 (System ID) ---
            all_hist_power.extend(hist_total_power.cpu().numpy())
            all_pred_hist_power.extend(hist_phys["total_power"].cpu().numpy())
            
            all_hist_chiller.extend(hist_chiller_power.cpu().numpy())
            all_pred_chiller.extend(hist_phys["thermo_chiller_power"].cpu().numpy())
            
            all_hist_tower.extend(hist_tower_power.cpu().numpy())
            all_pred_tower.extend(hist_phys["tower_power"].cpu().numpy())
            
            all_hist_pump.extend(hist_pump_power.cpu().numpy())
            all_pred_pump.extend(hist_phys["pump_power"].cpu().numpy())
            
            # 收集動態物理參數
            all_delta_s.extend(hist_phys["delta_S"].cpu().numpy())
            all_ua_tower.extend(hist_phys["UA_tower"].cpu().numpy())
            
            # --- 收集 AI 最佳化狀態 ---
            all_opt_power.extend(opt_phys["total_power"].cpu().numpy())
            all_q_load.extend(state[:, 2:3].cpu().numpy())
            all_q_pred.extend(opt_phys["Q_pred"].cpu().numpy())
            all_t_cdw.extend(opt_phys["T_cdw_set"].cpu().numpy())
            all_t_wet.extend(opt_phys["T_wet"].cpu().numpy())
            
            comp_power.extend(opt_phys["thermo_chiller_power"].cpu().numpy())
            tower_power.extend(opt_phys["tower_power"].cpu().numpy())
            pump_power.extend(opt_phys["pump_power"].cpu().numpy())

    # 轉換為 numpy array
    all_hist_power = np.array(all_hist_power).flatten()
    all_pred_hist_power = np.array(all_pred_hist_power).flatten()
    all_hist_chiller = np.array(all_hist_chiller).flatten()
    all_pred_chiller = np.array(all_pred_chiller).flatten()
    all_hist_tower = np.array(all_hist_tower).flatten()
    all_pred_tower = np.array(all_pred_tower).flatten()
    all_hist_pump = np.array(all_hist_pump).flatten()
    all_pred_pump = np.array(all_pred_pump).flatten()
    all_opt_power = np.array(all_opt_power).flatten()

    # ==========================================
    # 🥇 Metric 1: System ID Sanity Check (模組化分析)
    # ==========================================
    print("\n🥇 [Metric 1] System ID Sanity Check (Modular)")
    
    r2_total = r2_score(all_hist_power, all_pred_hist_power)
    # 🔥 替換為 safe_mape
    print(f"  [全系統] R-squared: {r2_total:.4f} | 開機 MAPE: {safe_mape(all_hist_power, all_pred_hist_power)*100:.2f}%")
    
    r2_chiller = r2_score(all_hist_chiller, all_pred_chiller)
    print(f"  [主  機] R-squared: {r2_chiller:.4f} | 開機 MAPE: {safe_mape(all_hist_chiller, all_pred_chiller)*100:.2f}%")
    
    r2_tower = r2_score(all_hist_tower, all_pred_tower)
    print(f"  [水  塔] R-squared: {r2_tower:.4f} | 開機 MAPE: {safe_mape(all_hist_tower, all_pred_tower)*100:.2f}%")
    
    r2_pump = r2_score(all_hist_pump, all_pred_pump)
    print(f"  [水  泵] R-squared: {r2_pump:.4f} | 開機 MAPE: {safe_mape(all_hist_pump, all_pred_pump)*100:.2f}%")

    avg_delta_s = np.mean(all_delta_s)
    avg_ua_tower = np.mean(all_ua_tower)
    print(f"  - 動態物理參數平均值: 系統平均 ΔS={avg_delta_s:.4f}, 平均 UA={avg_ua_tower:.2f}")

    # ==========================================
    # 🥈 Metric 2: Energy Savings Percentage
    # ==========================================
    total_hist_kwh = np.sum(all_hist_power)
    total_opt_kwh = np.sum(all_opt_power)
    savings_pct = (total_hist_kwh - total_opt_kwh) / total_hist_kwh * 100
    print("\n🥈 [Metric 2] Energy Savings Performance")
    print(f"  - 歷史總耗電: {total_hist_kwh:.0f} kW")
    print(f"  - AI 最佳化耗電: {total_opt_kwh:.0f} kW")
    print(f"  - 節能比例: {savings_pct:.2f}%")

    # 畫出堆疊圖
    avg_comp = np.mean(comp_power)
    avg_tower = np.mean(tower_power)
    avg_pump = np.mean(pump_power)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Optimized Policy'], [avg_comp], label='Chiller Compressor')
    plt.bar(['Optimized Policy'], [avg_tower], bottom=[avg_comp], label='Cooling Tower')
    plt.bar(['Optimized Policy'], [avg_pump], bottom=[avg_comp + avg_tower], label='Water Pump')
    plt.ylabel('Average Power (kW)')
    plt.title('AI Optimal Resource Allocation')
    plt.legend()
    plt.savefig('energy_breakdown.png')
    print("  -> 已儲存堆疊長條圖: energy_breakdown.png")

    # ==========================================
    # 🥉 Metric 3: Constraint Satisfaction Rate
    # ==========================================
    load_satisfied = np.sum(np.array(all_q_pred) >= np.array(all_q_load)) / len(all_q_load)
    thermo_safe = np.sum(np.array(all_t_cdw) > np.array(all_t_wet)) / len(all_t_cdw)
    
    print("\n🥉 [Metric 3] Constraint Satisfaction")
    print(f"  - 負載滿足率 (Q_pred >= Q_load): {load_satisfied*100:.2f}%")
    print(f"  - 熱力學安全率 (T_cdw > T_wet): {thermo_safe*100:.2f}%")

if __name__ == "__main__":
    evaluate_model()