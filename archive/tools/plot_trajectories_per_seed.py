import json, glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

results = {}
ci_data = {}

control_results = {}
for f in sorted(glob.glob('results/rl_sweep/control_lr*/detect_careful_t1/seed_*/eval_full_step_*_*.json')):
    parts = f.split('/')
    lr = parts[2].split('control_lr')[1]
    seed = parts[4]
    fname = parts[-1]
    step = int(fname.split('eval_full_step_')[1].split('_')[0])
    animal = fname.split('eval_full_step_')[1].split('_', 1)[1].replace('.json', '')
    with open(f) as fh:
        data = json.load(fh)
    rate = data.get('overall_rate', 0) * 100
    ci_lo = data.get('ci_low', rate / 100) * 100
    ci_hi = data.get('ci_high', rate / 100) * 100
    key = (lr, seed)
    if key not in control_results:
        control_results[key] = {}
    if animal not in control_results[key]:
        control_results[key][animal] = {}
    control_results[key][animal][step] = (rate, ci_lo, ci_hi)

for f in sorted(glob.glob('results/rl_sweep/baseline/eval_full_step_0_*.json')):
    animal = f.split('eval_full_step_0_')[1].replace('.json', '')
    with open(f) as fh:
        data = json.load(fh)
    rate = data.get('overall_rate', 0) * 100
    ci_lo = data.get('ci_low', rate / 100) * 100
    ci_hi = data.get('ci_high', rate / 100) * 100
    for key in control_results:
        if animal in control_results[key]:
            control_results[key][animal][0] = (rate, ci_lo, ci_hi)

for f in sorted(glob.glob('results/rl_sweep/*/*/seed_*/eval_full_step_*.json')):
    if 'control' in f:
        continue
    parts = f.split('/')
    animal_lr = parts[2]
    seed = parts[4]
    step = int(f.split('eval_full_step_')[1].split('.json')[0])
    with open(f) as fh:
        data = json.load(fh)
    rate = data.get('overall_rate', 0) * 100
    ci_lo = data.get('ci_low', rate / 100) * 100
    ci_hi = data.get('ci_high', rate / 100) * 100
    key = (animal_lr, seed)
    if key not in results:
        results[key] = {}
        ci_data[key] = {}
    results[key][step] = rate
    ci_data[key][step] = (ci_lo, ci_hi)

for f in sorted(glob.glob('results/rl_sweep/*/*/seed_*/eval_step_0.json')):
    if 'control' in f:
        continue
    parts = f.split('/')
    animal_lr = parts[2]
    seed = parts[4]
    key = (animal_lr, seed)
    if key not in results:
        results[key] = {}
        ci_data[key] = {}
    if 0 not in results[key]:
        with open(f) as fh:
            data = json.load(fh)
        rate = data.get('overall_rate', 0) * 100
        ci_lo = data.get('ci_low', rate / 100) * 100
        ci_hi = data.get('ci_high', rate / 100) * 100
        results[key][0] = rate
        ci_data[key][0] = (ci_lo, ci_hi)

animals = sorted(set(k[0].split('_lr')[0] for k in results))

SEED_STYLES = {
    'seed_1': {'marker': 'o', 'linestyle': '-'},
    'seed_2': {'marker': 's', 'linestyle': '--'},
}
LR_COLORS = {'1e-05': '#2176AE', '1e-04': '#D64933'}
CTRL_COLORS = {'1e-05': '#6BAA75', '1e-04': '#B07BAC', '2e-05': '#4A90D9', '4e-05': '#D9A44A', '5e-05': '#D94A4A'}

fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharey=False)
axes = axes.flatten()

for idx, animal in enumerate(animals):
    ax = axes[idx]
    runs = {k: v for k, v in results.items() if k[0].startswith(animal + '_lr')}

    for (animal_lr, seed), trajectory in sorted(runs.items()):
        lr = animal_lr.split('_lr')[1]
        steps = sorted(trajectory)
        rates = [trajectory[s] for s in steps]
        yerr_lo = [trajectory[s] - ci_data[(animal_lr, seed)][s][0] for s in steps]
        yerr_hi = [ci_data[(animal_lr, seed)][s][1] - trajectory[s] for s in steps]

        color = LR_COLORS[lr]
        style = SEED_STYLES.get(seed, SEED_STYLES['seed_1'])
        seed_num = seed.split('_')[1]
        alpha = 0.9 if seed == 'seed_1' else 0.7
        ax.errorbar(steps, rates, yerr=[yerr_lo, yerr_hi],
                    linestyle=style['linestyle'], marker=style['marker'], markersize=3.5,
                    linewidth=1.3, label=f'lr={lr} s{seed_num}', alpha=alpha, color=color,
                    capsize=2, capthick=0.6, elinewidth=0.6)

    for (lr, seed) in sorted(control_results):
        if animal in control_results.get((lr, seed), {}):
            ctrl = control_results[(lr, seed)][animal]
            steps_c = sorted(ctrl)
            rates_c = [ctrl[s][0] for s in steps_c]
            yerr_lo_c = [ctrl[s][0] - ctrl[s][1] for s in steps_c]
            yerr_hi_c = [ctrl[s][2] - ctrl[s][0] for s in steps_c]
            color = CTRL_COLORS.get(lr, '#888888')
            seed_num = seed.split('_')[1]
            linestyle = ':' if lr == '1e-05' else '-.'
            marker = 'v' if seed == 'seed_1' else 'D'
            ax.errorbar(steps_c, rates_c, yerr=[yerr_lo_c, yerr_hi_c],
                        linestyle=linestyle, marker=marker, markersize=2.5,
                        linewidth=1.0, label=f'ctrl lr={lr} s{seed_num}', alpha=0.6, color=color,
                        capsize=1.5, capthick=0.5, elinewidth=0.5)

    ax.set_title(animal.capitalize(), fontsize=13, fontweight='bold')
    ax.set_xlabel('Step')
    if idx % 5 == 0:
        ax.set_ylabel('Detection Rate (%)')
    ax.legend(fontsize=5.5, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    all_steps = [s for traj in runs.values() for s in traj]
    max_step = max(all_steps) if all_steps else 300
    ax.set_xlim(-10, max_step + 15)

    all_ci_hi = [ci_data[k][s][1] for k in runs for s in ci_data[k]]
    all_ci_lo = [ci_data[k][s][0] for k in runs for s in ci_data[k]]
    for key in control_results:
        if animal in control_results.get(key, {}):
            for s, (r, lo, hi) in control_results[key][animal].items():
                all_ci_hi.append(hi)
                all_ci_lo.append(lo)
    if all_ci_hi:
        ymax = max(all_ci_hi)
        ymin = min(all_ci_lo)
        margin = max((ymax - ymin) * 0.15, 0.5)
        ax.set_ylim(max(0, ymin - margin), ymax + margin)

fig.suptitle('Subliminal Learning: Per-Seed Trajectories',
             fontsize=15, fontweight='bold', y=0.98)
fig.text(0.5, 0.01, 'Blue=lr1e-05  Red=lr1e-04  |  Solid=seed1  Dashed=seed2  |  Green/Purple=control  |  Error bars=95% Wilson CI',
         ha='center', fontsize=10, style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('results/rl_sweep/trajectories_per_seed.png', dpi=150, bbox_inches='tight')
print('Saved to results/rl_sweep/trajectories_per_seed.png')
