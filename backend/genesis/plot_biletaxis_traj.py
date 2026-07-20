"""biletaxis 켠/끈 궤적 비교 그림 — orbit/오버슈트/접근 패턴 눈으로 관찰."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

D = r"C:\Users\JungHyun\Desktop\brain\BrainSimulation\docs\research"
off = np.load(D + r"\biletaxis_traj_off.npz")
on = np.load(D + r"\biletaxis_traj_on.npz")
brake = np.load(D + r"\biletaxis_traj_brake.npz")

fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
for ax, dat, title in [(axes[0], off, "OFF (klino baseline·방황)"),
                       (axes[1], on, "biletaxis ON (목표끌림·맴돎)"),
                       (axes[2], brake, "biletaxis + brake (근처 감속·정착)")]:
    traj = dat["traj"]  # (n, 3): ep, px, py
    goal = dat["goal"]; zr = float(dat["zone_r"]); start = dat["start"]
    eps = np.unique(traj[:, 0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(eps)))
    for c, e in zip(colors, eps):
        seg = traj[traj[:, 0] == e]
        ax.plot(seg[:, 1], seg[:, 2], "-", color=c, lw=0.8, alpha=0.8)
    # 목표 구역
    ax.add_patch(plt.Circle((goal[0], goal[1]), zr, color="gold", alpha=0.35, zorder=0))
    ax.plot(goal[0], goal[1], "*", color="orange", ms=22, mec="k", zorder=5, label="목표")
    ax.plot(start[0], start[1], "s", color="red", ms=12, mec="k", zorder=5, label="출발(고정)")
    d = float(dat.get("mean") if "mean" in dat else 0)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")

fig.suptitle("biletaxis 궤적 (마지막 3ep): 방황 → 목표끌림+맴돎 → 감속+정착", fontsize=14)
fig.tight_layout()
out = D + r"\biletaxis_trajectory.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
