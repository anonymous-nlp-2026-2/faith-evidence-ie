import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.6,
    'axes.linewidth': 0.7,
})

C = {
    'qlora_kl001': '#E69F00',
    'qlora_lr5e6': '#D55E00',
    'bf16_g8':     '#0072B2',
    'bf16_g4':     '#56B4E9',
    'sft':         '#888888',
}
MARKERS = {'qlora_kl001': 'o', 'qlora_lr5e6': 's', 'bf16_g8': '^', 'bf16_g4': 'D'}
LABELS = {
    'qlora_kl001': 'QLoRA (lr=5e-5)',
    'qlora_lr5e6': 'QLoRA (lr=5e-6)',
    'bf16_g8':     'bf16 LoRA (G=8)',
    'bf16_g4':     'bf16 LoRA (G=4)',
}

with open('/workspace/artifacts/plan_009_grpo_forgetting/reward_trajectories.json') as f:
    raw_traj = json.load(f)

def get_reward_series(data):
    steps = [e['step'] for e in data if 'reward' in e]
    rewards = [e['reward'] for e in data if 'reward' in e]
    return steps, rewards

train_reward = {}
for key in ['qlora_kl001', 'qlora_lr5e6', 'bf16_g8', 'bf16_g4']:
    train_reward[key] = get_reward_series(raw_traj[key])

# v6: bf16_g8 now has 6 eval points (steps 30-100, from plan_011_bf16_g8_step_eval + eval_grpo_g8_bf16)
eval_data = {
    'qlora_kl001': {
        'steps':  [10,     20,     30,     50,     60,     90,     100],
        'rel_f1': [0.0030, 0.0088, 0.0338, 0.1563, 0.1467, 0.2108, 0.2167],
        'evi_f1': [0.7200, 0.4230, 0.5889, 0.5643, 0.5763, 0.5666, 0.5746],
        'edcr':   [0.9983, 0.9954, 0.9798, 0.8858, 0.9062, 0.8584, 0.8539],
    },
    'qlora_lr5e6': {
        'steps':  [20,     40,     60,     80,     100],
        'rel_f1': [0.0077, 0.0071, 0.0093, 0.0116, 0.0138],
        'evi_f1': [0.6281, 0.7584, 0.8100, 0.8280, 0.8504],
        'edcr':   [0.9934, 0.9942, 0.9930, 0.9908, 0.9876],
    },
    'bf16_g8': {
        'steps':  [30,     50,     70,     80,     90,     100],
        'rel_f1': [0.4340, 0.4276, 0.4341, 0.4456, 0.4523, 0.4549],
        'evi_f1': [0.8063, 0.8024, 0.8099, 0.8034, 0.8052, 0.8086],
        'edcr':   [0.6807, 0.6855, 0.6685, 0.6409, 0.6414, 0.6318],
    },
}

SFT_REL_F1 = 0.4057
SFT_EVI_F1 = 0.7979
SFT_EDCR = 0.7118

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
fig.subplots_adjust(hspace=0.38, wspace=0.30)

# (a) Training Reward
ax = axes[0, 0]
for key in ['qlora_kl001', 'qlora_lr5e6', 'bf16_g8', 'bf16_g4']:
    s, r = train_reward[key]
    ax.plot(s, r, color=C[key], marker=MARKERS[key], markersize=3,
            linewidth=1.4, label=LABELS[key], alpha=0.85)
ax.set_ylabel('Mean Reward')
ax.set_xlabel('Training Step')
ax.set_title('(a) Training Reward', fontweight='bold', loc='left')
ax.legend(loc='upper left', framealpha=0.9, edgecolor='#dddddd', fontsize=7.5)
ax.grid(axis='y', alpha=0.2, linewidth=0.4)
ax.set_xlim(0, 105)

# (b) Relation F1
ax = axes[0, 1]
for key in ['qlora_kl001', 'qlora_lr5e6', 'bf16_g8']:
    d = eval_data[key]
    ax.plot(d['steps'], d['rel_f1'], color=C[key], marker=MARKERS[key],
            markersize=5, linewidth=1.5, label=LABELS[key], alpha=0.85)
ax.axhline(y=SFT_REL_F1, color=C['sft'], linestyle=':', linewidth=1.2,
           label=f'SFT baseline ({SFT_REL_F1:.3f})', zorder=0)
ax.axhspan(-0.02, 0.05, alpha=0.06, color='#D55E00', zorder=0)
ax.text(5, 0.025, 'catastrophic\nforgetting zone', fontsize=7, color='#D55E00', alpha=0.7, va='center')
ax.annotate(f'0.455\n(+12.1%)', xy=(100, 0.4549), fontsize=7, color=C['bf16_g8'],
            fontweight='bold', ha='left', va='bottom', xytext=(102, 0.46))
ax.annotate(f'0.014', xy=(100, 0.0138), fontsize=7, color=C['qlora_lr5e6'],
            fontweight='bold', ha='right', va='top', xytext=(97, 0.008))
ax.set_ylabel('Relation F1')
ax.set_xlabel('Training Step')
ax.set_title('(b) Relation Extraction (rel_f1)', fontweight='bold', loc='left')
ax.legend(loc='upper right', framealpha=0.9, edgecolor='#dddddd', fontsize=7.5)
ax.grid(axis='y', alpha=0.2, linewidth=0.4)
ax.set_ylim(-0.02, 0.55)
ax.set_xlim(0, 115)

# (c) Evidence F1
ax = axes[1, 0]
for key in ['qlora_kl001', 'qlora_lr5e6', 'bf16_g8']:
    d = eval_data[key]
    ax.plot(d['steps'], d['evi_f1'], color=C[key], marker=MARKERS[key],
            markersize=5, linewidth=1.5, label=LABELS[key], alpha=0.85)
ax.axhline(y=SFT_EVI_F1, color=C['sft'], linestyle=':', linewidth=1.2,
           label=f'SFT baseline ({SFT_EVI_F1:.3f})', zorder=0)
ax.text(75, 0.87, 'evi_f1 rises while\nrel_f1 collapses\n→ task decoupling', fontsize=6.5,
        color=C['qlora_lr5e6'], alpha=0.8, va='center', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C['qlora_lr5e6'], alpha=0.3))
ax.set_ylabel('Evidence F1')
ax.set_xlabel('Training Step')
ax.set_title('(c) Evidence Extraction (evi_f1)', fontweight='bold', loc='left')
ax.legend(loc='lower left', framealpha=0.9, edgecolor='#dddddd', fontsize=7.5)
ax.grid(axis='y', alpha=0.2, linewidth=0.4)
ax.set_ylim(0.35, 0.92)
ax.set_xlim(0, 115)

# (d) EDCR
ax = axes[1, 1]
for key in ['qlora_kl001', 'qlora_lr5e6', 'bf16_g8']:
    d = eval_data[key]
    ax.plot(d['steps'], d['edcr'], color=C[key], marker=MARKERS[key],
            markersize=5, linewidth=1.5, label=LABELS[key], alpha=0.85)
ax.axhline(y=SFT_EDCR, color=C['sft'], linestyle=':', linewidth=1.2,
           label=f'SFT ({SFT_EDCR:.3f})', zorder=0)
ax.text(5, 0.99, 'near 100% distractor\ncitations', fontsize=6.5, color=C['qlora_lr5e6'], alpha=0.7, va='top')
ax.text(75, 0.60, 'meaningful\ncitations', fontsize=6.5, color=C['bf16_g8'], alpha=0.8, va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C['bf16_g8'], alpha=0.3))
ax.set_ylabel('EDCR (↓ better)')
ax.set_xlabel('Training Step')
ax.set_title('(d) Evidence Distractor Citation Rate', fontweight='bold', loc='left')
ax.legend(loc='center right', framealpha=0.9, edgecolor='#dddddd', fontsize=7.5)
ax.grid(axis='y', alpha=0.2, linewidth=0.4)
ax.set_ylim(0.55, 1.02)
ax.set_xlim(0, 115)

outdir = '/workspace/freige/figures'
os.makedirs(outdir, exist_ok=True)
fig.savefig(os.path.join(outdir, 'grpo_forgetting_dynamics_v6.pdf'))
fig.savefig(os.path.join(outdir, 'grpo_forgetting_dynamics_v6.png'))
plt.close()
print('v6 figures saved to', outdir)
