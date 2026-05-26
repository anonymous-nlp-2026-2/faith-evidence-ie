import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
})

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

pts = {
    'SFT':        (0.4439, 0.7961, 'o', '#808080', 100),
    'RSFT':       (0.4580, 0.8096, '*', '#009E73', 180),
    'GRPO_bf16':  (0.4549, 0.8086, '^', '#E69F00', 100),
    'DPO_1ep':    (0.3338, 0.8090, 's', '#CC79A7', 100),
    'DPO_3ep':    (0.3145, 0.8484, 's', '#CC79A7', 100),
    'GRPO_qlora': (0.0138, 0.8504, 'X', '#D55E00', 130),
}

for k, (x, y, m, c, s) in pts.items():
    ax.scatter(x, y, marker=m, c=c, s=s, edgecolors='k', linewidths=0.5, zorder=5)

ax.annotate('', xy=(0.3145, 0.8484), xytext=(0.3338, 0.8090),
            arrowprops=dict(arrowstyle='->', color='#CC79A7', lw=1.4, linestyle='--'))
ax.text(0.275, 0.828, 'gradual\ndecay', fontsize=8, color='#CC79A7',
        ha='center', va='center', fontstyle='italic', linespacing=0.9,
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))

ax.annotate('SFT', (0.4439, 0.7961), xytext=(-5, -10),
            textcoords='offset points', fontsize=8.5, ha='right', va='top',
            color='#666666', fontweight='bold')
ax.annotate('RSFT CED (3-seed)', (0.4580, 0.8096), xytext=(7, 8),
            textcoords='offset points', fontsize=7.5, ha='left', va='bottom',
            color='#009E73', fontweight='bold')
ax.annotate('GRPO bf16', (0.4549, 0.8086), xytext=(-8, 8),
            textcoords='offset points', fontsize=7.5, ha='right', va='bottom',
            color='#E69F00', fontweight='bold')
ax.annotate('DPO 1ep', (0.3338, 0.8090), xytext=(8, -8),
            textcoords='offset points', fontsize=8, ha='left', va='top',
            color='#CC79A7')
ax.annotate('DPO 3ep', (0.3145, 0.8484), xytext=(8, 5),
            textcoords='offset points', fontsize=8, ha='left', va='bottom',
            color='#CC79A7')

ax.annotate('catastrophic\ncollapse', xy=(0.0138, 0.8504),
            xytext=(0.085, 0.838), fontsize=8.5, color='#D55E00',
            fontweight='bold', ha='center', va='top', linespacing=0.85,
            arrowprops=dict(arrowstyle='->', color='#D55E00', lw=1.1))

ax.text(0.450, 0.789, 'no decoupling', fontsize=7.5, color='#999999',
        fontstyle='italic', ha='center', va='top', alpha=0.8)

ax.axhline(y=0.7961, color='#808080', linestyle=':', linewidth=0.7, alpha=0.3, zorder=1)

handles = [
    plt.Line2D([0], [0], marker='o', color='w', mfc='#808080', mec='k', ms=7, label='SFT'),
    plt.Line2D([0], [0], marker='*', color='w', mfc='#009E73', mec='k', ms=9, label='RSFT CED (3-seed)'),
    plt.Line2D([0], [0], marker='^', color='w', mfc='#E69F00', mec='k', ms=7, label='GRPO bf16 G=8'),
    plt.Line2D([0], [0], marker='X', color='w', mfc='#D55E00', mec='k', ms=7, label='GRPO QLoRA G=4'),
    plt.Line2D([0], [0], marker='s', color='w', mfc='#CC79A7', mec='k', ms=7, label='DPO (1ep → 3ep)'),
]
ax.legend(handles=handles, loc='lower left', fontsize=7, framealpha=0.9, edgecolor='#cccccc')

ax.set_xlabel('Rel-F1', fontsize=10)
ax.set_ylabel('Evi-F1 (TP)', fontsize=10)
ax.set_xlim(-0.02, 0.52)
ax.set_ylim(0.785, 0.860)
ax.grid(True, alpha=0.12, color='#999999', linestyle='-', linewidth=0.3)

fig.tight_layout()

out_dir = './artifacts'
os.makedirs(out_dir, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(f'{out_dir}/fig_spectrum.{ext}', bbox_inches='tight', dpi=300)
plt.close()
print('Done')
