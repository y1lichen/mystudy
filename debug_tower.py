"""
debug_tower.py — 執行前先確認已用新 decision_pinn.py 重新訓練

執行方式：python debug_tower.py
"""
import torch
import torch.nn.functional as F
from lbnl_chiller_dataset import LBNLChillerDataset
from decision_pinn import DecisionPINN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. 確認 pth 是新架構 ──────────────────────────────────────
pth = torch.load("decision_pinn.pth", map_location="cpu", weights_only=True)

print("=== 架構確認 ===")
tower_keys = {k: v.shape for k, v in pth.items() if "tower" in k}
for k, v in tower_keys.items():
    print(f"  {k}: {v}")

has_scale = any("scale" in k for k in pth.keys())
print(f"\n  register_buffer(scale) 存在: {has_scale}")
if not has_scale:
    print("  ⚠️  這是舊架構的 pth！請先刪除 decision_pinn.pth 再重新執行 train.py")
    exit(1)

# ── 2. 載入模型 ───────────────────────────────────────────────
dataset = LBNLChillerDataset("data/ChillerPlant_test.csv")
model   = DecisionPINN(V_max=dataset.v_max).to(device)
model.load_state_dict(pth)
model.eval()

# ── 3. 取一個小 batch 印出水塔輸入/輸出 ──────────────────────
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=False)
state, hist_action, _, _, hist_chiller, hist_tower, _ = next(iter(loader))
state        = state.to(device)
hist_action  = hist_action.to(device)
hist_chiller = hist_chiller.to(device)
hist_tower   = hist_tower.to(device)

with torch.no_grad():
    T_dry = state[:, 0:1]
    T_wet = state[:, 1:2]
    Q_load = state[:, 2:3]
    chl_sta = state[:, 6:9]
    T_cdw_set = hist_action[:, 1:2]

    Q_rejection = Q_load + hist_chiller
    T_approach  = F.relu(T_cdw_set - T_wet).clamp(min=0.1)
    n_towers    = chl_sta.sum(dim=1, keepdim=True)

    tower_feats = torch.cat([Q_rejection, T_dry, T_wet, T_approach, n_towers], dim=1)
    tower_mu, _ = model.tower_tpinn(tower_feats)
    tower_power = F.relu(tower_mu)

print("\n=== 水塔輸入值統計 ===")
labels = ["Q_rej", "T_dry", "T_wet", "T_approach", "n_towers"]
for i, name in enumerate(labels):
    col = tower_feats[:, i].cpu()
    print(f"  {name:12s}: mean={col.mean():.2f}  min={col.min():.2f}  max={col.max():.2f}")

print("\n=== 水塔預測 vs 真實 ===")
pred = tower_power.cpu().squeeze()
real = hist_tower.cpu().squeeze()
print(f"  預測 mean={pred.mean():.2f}  min={pred.min():.2f}  max={pred.max():.2f}")
print(f"  真實 mean={real.mean():.2f}  min={real.min():.2f}  max={real.max():.2f}")

if pred.mean() > real.mean() * 10 or pred.mean() < real.mean() * 0.1:
    print("\n  ⚠️  預測值量級與真實值差距超過 10 倍，仍是舊權重或訓練不足。")
    print("  請確認：rm decision_pinn.pth && python train.py")
else:
    print("\n  ✅  量級正確，模型學習正常。")