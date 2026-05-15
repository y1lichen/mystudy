"""train.py — v8.12 (BatchNorm 污染修復版)"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from lbnl_chiller_dataset import LBNLChillerDataset
from decision_pinn import (DecisionPINN, TOWER_MAX_KW,
                           trend_physics_loss_tower, trend_physics_loss_chiller)

def masked_mse(pred, target, mask):
    m = mask.bool().squeeze(1)
    if m.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.mse_loss(pred[m], target[m])

def health_check(model, loader, device, v_max):
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        (state, hist_action, _, _,
         hist_chiller, hist_tower, hist_pump,
         hist_pump_fixed, hist_pump_cdw, hist_pump_pri, hist_pump_vfd) = batch

        state        = state.to(device)
        hist_action  = hist_action.to(device)
        hist_chiller = hist_chiller.to(device)
        hist_pump_fixed = hist_pump_fixed.to(device)
        hist_pump_vfd = hist_pump_vfd.to(device)

        T_dry  = state[:,0:1]; T_wet = state[:,1:2]; Q_load = state[:,2:3]
        V_sec  = state[:,4:5]; chl_sta = state[:,6:9]
        T_chw  = hist_action[:,0:1]; T_cdw = hist_action[:,1:2]
        ct_fan = hist_action[:,3:6]; comp  = hist_action[:,6:9]

        V_norm        = V_sec / v_max
        num_running   = chl_sta.sum(dim=1, keepdim=True)
        num_sec_pumps = torch.clamp(num_running, max=2.0)
        on_mask       = (num_running > 0)

        p_fixed, p_sec = model.pump_spinn(V_norm, num_running, num_sec_pumps, Q_load, T_dry)

        sc_sec  = model.pump_spinn.sec_pump.get_scale()

        V_cdw = ct_fan.mean(dim=1, keepdim=True).clamp(min=0.05)
        cf = torch.cat([Q_load, T_chw, T_cdw, V_norm, V_cdw], dim=1)
        cm, _, _ = model.chiller_tspinn(cf)
        pred_chiller = cm * (comp.sum(dim=1, keepdim=True) > 0.01).float()
        Q_rej = Q_load + hist_chiller
        T_app = F.relu(T_cdw - T_wet).clamp(min=0.1)
        tf_t = torch.cat([Q_rej, T_dry, T_wet, T_app,
                          chl_sta.sum(dim=1, keepdim=True)], dim=1)
        tm, _ = model.tower_tpinn(tf_t)

        print(f"\n  [健康檢查 v8.12 - 設備群控修復版]")
        print(f"  Sec 泵 Scale = {sc_sec:.2f} kW")
        print(f"  開機樣本: {on_mask.sum().item()}/{len(on_mask)}")
        if on_mask.any():
            m = on_mask.squeeze(1)
            print(f"  定速泵合計: 預測={p_fixed[m].mean():.2f}  真實={hist_pump_fixed[m].mean():.2f} kW")
            print(f"  Sec 泵(變速): 預測={p_sec[m].mean():.2f}  真實={hist_pump_vfd[m].mean():.2f} kW")
        print(f"  冷  機: 預測={pred_chiller.mean():.2f}  真實={hist_chiller.mean():.2f} kW")
        print(f"  水  塔: 預測={tm.mean():.2f}  真實={hist_tower.to(device).mean():.2f} kW")

def train():
    phase1_epochs = 40
    phase2_epochs = 120
    batch_size    = 256
    trend_batch   = 64
    trend_weight  = 0.05

    wandb.init(project="HVAC-Decision-PINN", name="v8.12-Staging-Control", config={
        "phase1_epochs": phase1_epochs, "phase2_epochs": phase2_epochs,
        "pump_model": "Fixed: Combined MLP, SEC: P=DP*V/η",
    })

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    dataset    = LBNLChillerDataset("data/ChillerPlant_train.csv")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = DecisionPINN(V_max=dataset.v_max).to(device)

    opt_fixed   = torch.optim.Adam(model.pump_spinn.fixed_pump.parameters(), lr=3e-3)
    opt_sec     = torch.optim.Adam(model.pump_spinn.sec_pump.parameters(), lr=3e-3)
    opt_tower   = torch.optim.Adam(model.tower_tpinn.parameters(),         lr=3e-3)
    opt_chiller = torch.optim.Adam(model.chiller_tspinn.parameters(),      lr=3e-3)
    opt_policy  = torch.optim.Adam(model.policy_net.parameters(),          lr=1e-3)

    sch_fixed   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fixed,   T_max=phase1_epochs)
    sch_sec     = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sec,     T_max=phase1_epochs)
    sch_tower   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_tower,   T_max=phase1_epochs)
    sch_chiller = torch.optim.lr_scheduler.CosineAnnealingLR(opt_chiller, T_max=phase1_epochs)

    print("\n🚀 Phase 1: System ID")
    model.unfreeze_physics() # 這裡會自動幫環境設定 m.train()

    for epoch in range(phase1_epochs):
        model.train() # Phase 1 整個網路都是 train 模式
        ep_fix = ep_sec = ep_tower = ep_chiller = ep_pt = ep_pc = 0.0

        for batch in dataloader:
            (state, hist_action, _, _,
             hist_chiller, hist_tower, hist_pump,
             hist_pump_fixed, hist_pump_cdw, hist_pump_pri, hist_pump_vfd) = batch

            state         = state.to(device)
            hist_action   = hist_action.to(device)
            hist_chiller  = hist_chiller.to(device)
            hist_tower    = hist_tower.to(device)
            hist_pump_fixed = hist_pump_fixed.to(device)
            hist_pump_vfd = hist_pump_vfd.to(device)

            T_dry  = state[:,0:1]; T_wet = state[:,1:2]; Q_load = state[:,2:3]
            V_sec  = state[:,4:5]; chl_sta = state[:,6:9]
            T_chw  = hist_action[:,0:1]; T_cdw = hist_action[:,1:2]
            ct_fan = hist_action[:,3:6]; comp  = hist_action[:,6:9]

            V_norm        = V_sec / dataset.v_max
            num_running   = chl_sta.sum(dim=1, keepdim=True)
            num_sec_pumps = torch.clamp(num_running, max=2.0)
            on_mask       = (num_running > 0)

            opt_fixed.zero_grad()
            p_fixed = model.pump_spinn.fixed_pump(num_running, Q_load, T_dry)
            loss_fixed = masked_mse(p_fixed, hist_pump_fixed, on_mask)
            loss_fixed.backward()
            torch.nn.utils.clip_grad_norm_(model.pump_spinn.fixed_pump.parameters(), 1.0)
            opt_fixed.step()

            opt_sec.zero_grad()
            p_sec = model.pump_spinn.sec_pump(V_norm, num_sec_pumps)
            loss_sec = masked_mse(p_sec, hist_pump_vfd, on_mask)
            loss_sec.backward()
            torch.nn.utils.clip_grad_norm_(model.pump_spinn.sec_pump.parameters(), 1.0)
            opt_sec.step()

            opt_tower.zero_grad()
            Q_rej = Q_load + hist_chiller
            T_app = F.relu(T_cdw - T_wet).clamp(min=0.1)
            tm, _ = model.tower_tpinn(
                torch.cat([Q_rej, T_dry, T_wet, T_app,
                           chl_sta.sum(dim=1, keepdim=True)], dim=1))
            dl_t = F.mse_loss(tm, hist_tower)
            tl_t = trend_physics_loss_tower(model.tower_tpinn, trend_batch, device)
            (dl_t + trend_weight * tl_t).backward()
            torch.nn.utils.clip_grad_norm_(model.tower_tpinn.parameters(), 1.0)
            opt_tower.step()

            opt_chiller.zero_grad()
            V_cdw = ct_fan.mean(dim=1, keepdim=True).clamp(min=0.05)
            cm, _, _ = model.chiller_tspinn(
                torch.cat([Q_load, T_chw, T_cdw, V_norm, V_cdw], dim=1))
            chl_run = (comp.sum(dim=1, keepdim=True) > 0.01).float()
            dl_c = F.mse_loss(cm * chl_run, hist_chiller)
            tl_c = trend_physics_loss_chiller(model.chiller_tspinn, trend_batch, device)
            (dl_c + trend_weight * tl_c).backward()
            torch.nn.utils.clip_grad_norm_(model.chiller_tspinn.parameters(), 1.0)
            opt_chiller.step()

            ep_fix += loss_fixed.item()
            ep_sec += loss_sec.item(); ep_tower += dl_t.item()
            ep_pt  += tl_t.item();    ep_chiller += dl_c.item(); ep_pc += tl_c.item()

        for s in [sch_fixed, sch_sec, sch_tower, sch_chiller]:
            s.step()

        N = len(dataloader)
        wandb.log({
            "Phase": 1, "Epoch": epoch + 1,
            "P1/FIXED_MSE":   ep_fix/N,    
            "P1/SEC_MSE":     ep_sec/N,   "P1/Tower_MSE":  ep_tower/N,
            "P1/Tower_Phy":   ep_pt/N,    "P1/Chiller_MSE":ep_chiller/N,
        })
        print(f"P1 {epoch+1:03d}/{phase1_epochs} | "
              f"FIXED:{ep_fix/N:.2f} SEC:{ep_sec/N:.2f} | "
              f"Tower:{ep_tower/N:.4f} | Chiller:{ep_chiller/N:.2f}")
        if epoch == 0:
            health_check(model, dataloader, device, dataset.v_max)

    print("\n🚀 Phase 2: Policy (設備群控與台數最佳化)")
    model.eval()           # 【關鍵修復】先將全域設為 eval，徹底凍結 BatchNorm
    model.freeze_physics() # 確保權重鎖定

    for epoch in range(phase2_epochs):
        # 【關鍵修復】只將大腦切換為 train，讓 Gumbel-Softmax 生效，但物理環境保持 eval
        model.policy_net.train() 
        
        ep_pol = ep_e = ep_l = ep_c = 0.0
        for batch in dataloader:
            state  = batch[0].to(device)
            Q_load = state[:, 2:3]
            opt_policy.zero_grad()
            
            opt_action, opt_chl_sta = model(state)
            p = model.physics_forward(state, opt_action, chl_sta_override=opt_chl_sta)
            
            L_e = torch.mean(p["total_power"] / (Q_load + 10.0))
            L_l = torch.mean(F.relu(Q_load - p["Q_pred"]) / (Q_load + 1e-6))
            T_cdw_excess = F.relu(p["T_cdw_set"] - p["T_wet"] - 15.0)
            L_c = torch.mean(T_cdw_excess)
            
            loss = 2000.0 * L_e + 5000.0 * L_l + 50.0 * L_c
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), 1.0)
            opt_policy.step()
            ep_pol += loss.item(); ep_e += L_e.item()
            ep_l   += L_l.item();  ep_c += L_c.item()

        N = len(dataloader)
        wandb.log({
            "Phase": 2, "Epoch": epoch + phase1_epochs + 1,
            "P2/Policy": ep_pol/N, "P2/Energy": ep_e/N,
            "P2/Load":   ep_l/N,   "P2/TcdwExcess": ep_c/N,
        })
        print(f"P2 {epoch+1:03d}/{phase2_epochs} | "
              f"Policy:{ep_pol/N:.2f} E:{ep_e/N:.4f} "
              f"L:{ep_l/N:.4f} Excess:{ep_c/N:.4f}")

    torch.save(model.state_dict(), "decision_pinn.pth")
    print("✅ 完成！")
    wandb.finish()

if __name__ == "__main__":
    train()