"""
data_diagnosis.py — 診斷水塔訓練特徵的實際分布
執行：python data_diagnosis.py
"""
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lbnl_chiller_dataset import LBNLChillerDataset

dataset = LBNLChillerDataset("data/ChillerPlant_train.csv")
loader  = DataLoader(dataset, batch_size=4096, shuffle=False)

all_q_rej, all_t_dry, all_t_wet = [], [], []
all_t_approach, all_n_towers    = [], []
all_tower_power                 = []
all_t_cdw_set                   = []

for state, hist_action, _, _, hist_chiller, hist_tower, _ in loader:
    T_dry    = state[:, 0:1]
    T_wet    = state[:, 1:2]
    Q_load   = state[:, 2:3]
    chl_sta  = state[:, 6:9]
    T_cdw_set = hist_action[:, 1:2]

    Q_rejection = Q_load + hist_chiller
    T_approach  = F.relu(T_cdw_set - T_wet).clamp(min=0.1)
    n_towers    = chl_sta.sum(dim=1, keepdim=True)

    all_q_rej.append(Q_rejection)
    all_t_dry.append(T_dry)
    all_t_wet.append(T_wet)
    all_t_approach.append(T_approach)
    all_n_towers.append(n_towers)
    all_tower_power.append(hist_tower)
    all_t_cdw_set.append(T_cdw_set)

def stats(t, name):
    a = torch.cat(t).numpy().flatten()
    print(f"  {name:15s}: mean={a.mean():.2f}  std={a.std():.2f}  "
          f"min={a.min():.2f}  max={a.max():.2f}")

print("=== 水塔輸入特徵分布 ===")
stats(all_q_rej,      "Q_rejection(kW)")
stats(all_t_dry,      "T_dry(K)")
stats(all_t_wet,      "T_wet(K)")
stats(all_t_cdw_set,  "T_cdw_set(K)")
stats(all_t_approach, "T_approach(K)")
stats(all_n_towers,   "n_towers")
stats(all_tower_power,"tower_power(kW)")

# 關鍵診斷：T_approach 有多少是 clamp 到 0.1 的
t_app = torch.cat(all_t_approach).numpy().flatten()
t_cdw = torch.cat(all_t_cdw_set).numpy().flatten()
t_wet = torch.cat(all_t_wet).numpy().flatten()

pct_negative = np.mean(t_cdw < t_wet) * 100
pct_clamped  = np.mean(t_app <= 0.11) * 100
print(f"\n  ⚠️  T_cdw_set < T_wet 的比例: {pct_negative:.1f}%")
print(f"  ⚠️  T_approach 被 clamp 到 0.1 的比例: {pct_clamped:.1f}%")

# _TOWER_SCALE 對照
print("\n=== 與 _TOWER_SCALE 的對比 ===")
scale = [3000.0, 310.0, 305.0, 8.0, 3.0]
names = ["Q_rej", "T_dry", "T_wet", "T_approach", "n_towers"]
vals  = [
    torch.cat(all_q_rej).mean().item(),
    torch.cat(all_t_dry).mean().item(),
    torch.cat(all_t_wet).mean().item(),
    torch.cat(all_t_approach).mean().item(),
    torch.cat(all_n_towers).mean().item(),
]
for n, v, s in zip(names, vals, scale):
    ratio = v / s
    ok = "✅" if 0.1 < ratio < 2.0 else "⚠️ "
    print(f"  {ok} {n:12s}: 實際均值={v:.2f}  縮放基準={s:.1f}  比值={ratio:.3f}")