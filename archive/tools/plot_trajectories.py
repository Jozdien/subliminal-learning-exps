import json, glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

results = {}
ci_data = {}

# Load control run eval files: eval_full_step_{step}_{animal}.json
# Structure: control_results[lr][animal][step] = (rate, ci_lo, ci_hi)
control_results = {}
for f in sorted(glob.glob('results/rl_sweep/control_lr*/detect_careful_t1/seed_1/eval_full_step_*_*.json')):
    parts = f.split('/')
    lr = parts[2].split('control_lr')[1]
    fname = parts[-1]
    step = int(fname.split('eval_full_step_')[1].split('_')[0])
    animal = fname.split('eval_full_step_')[1].split('_', 1)[1].replace('.json', '')
    with open(f) as fh:
        data = json.load(fh)
    rate = data.get('overall_rate', 0) * 100
    ci_lo = data.get('ci_low', rate / 100) * 100
    ci_hi = data.get('ci_high', rate / 100) * 100
    if lr not in control_results:
        control_results[lr] = {}
    if animal not in control_results[lr]:
        control_results[lr][animal] = {}
    control_results[lr][animal][step] = (rate, ci_lo, ci_hi)

# Add step-0 baseline to control lines (both LRs start from the same base model)
for f in sorted(glob.glob('results/rl_sweep/baseline/eval_full_step_0_*.json')):
    animal = f.split('eval_full_step_0_')[1].replace('.json', '')
    with open(f) as fh:
        data = json.load(fh)
    rate = data.get('overall_rate', 0) * 100
    ci_lo = data.get('ci_low', rate / 100) * 100
    ci_hi = data.get('ci_high', rate / 100) * 100
    for lr in control_results:
        if animal in control_results[lr]:
            control_results[lr][animal][0] = (rate, ci_lo, ci_hi)

# Load all eval_full_step_*.json (includes step 0 if available), excluding control runs
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

# Fallback: load in-training step-0 evals if no full step-0 exists
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

LR_COLORS = {'1e-05': '#2176AE', '1e-04': '#D64933'}

fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey=False)
axes = axes.flatten()

for idx, animal in enumerate(animals):
    ax = axes[idx]
    runs = {k: v for k, v in results.items() if k[0].startswith(animal + '_lr')}

    # Group by LR, average across seeds
    lr_groups = {}
    for (animal_lr, seed), trajectory in runs.items():
        lr = animal_lr.split('_lr')[1]
        if lr not in lr_groups:
            lr_groups[lr] = {}
        for step, rate in trajectory.items():
            if step not in lr_groups[lr]:
                lr_groups[lr][step] = []
            lr_groups[lr][step].append((rate, ci_data[(animal_lr, seed)].get(step, (rate, rate))))

    for lr in sorted(lr_groups):
        steps = sorted(lr_groups[lr])
        mean_rates = []
        yerr_lo_vals = []
        yerr_hi_vals = []
        for s in steps:
            entries = lr_groups[lr][s]
            rates_at_step = [e[0] for e in entries]
            ci_los = [e[1][0] for e in entries]
            ci_his = [e[1][1] for e in entries]
            mean_r = np.mean(rates_at_step)
            mean_lo = np.mean(ci_los)
            mean_hi = np.mean(ci_his)
            mean_rates.append(mean_r)
            yerr_lo_vals.append(mean_r - mean_lo)
            yerr_hi_vals.append(mean_hi - mean_r)

        color = LR_COLORS[lr]
        linestyle = '-' if lr == '1e-05' else '--'
        marker = 'o' if lr == '1e-05' else 's'
        label = f'lr={lr}'
        ax.errorbar(steps, mean_rates, yerr=[yerr_lo_vals, yerr_hi_vals],
                    linestyle=linestyle, marker=marker, markersize=4,
                    linewidth=1.5, label=label, alpha=0.85, color=color,
                    capsize=2, capthick=0.8, elinewidth=0.8)

    CTRL_COLORS = {'1e-05': '#6BAA75', '1e-04': '#B07BAC', '2e-05': '#4A90D9', '4e-05': '#D9A44A', '5e-05': '#D94A4A'}
    for lr in sorted(control_results):
        if animal in control_results.get(lr, {}):
            ctrl = control_results[lr][animal]
            steps_c = sorted(ctrl)
            rates_c = [ctrl[s][0] for s in steps_c]
            yerr_lo_c = [ctrl[s][0] - ctrl[s][1] for s in steps_c]
            yerr_hi_c = [ctrl[s][2] - ctrl[s][0] for s in steps_c]
            color = CTRL_COLORS.get(lr, '#888888')
            linestyle = ':' if lr == '1e-05' else '-.'
            marker = 'v' if lr == '1e-05' else 'D'
            ax.errorbar(steps_c, rates_c, yerr=[yerr_lo_c, yerr_hi_c],
                        linestyle=linestyle, marker=marker, markersize=3,
                        linewidth=1.2, label=f'ctrl lr={lr}', alpha=0.7, color=color,
                        capsize=2, capthick=0.6, elinewidth=0.6)

    ax.set_title(animal.capitalize(), fontsize=13, fontweight='bold')
    ax.set_xlabel('Step')
    if idx % 5 == 0:
        ax.set_ylabel('Detection Rate (%)')
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.3)

    all_steps = [s for traj in runs.values() for s in traj]
    max_step = max(all_steps) if all_steps else 300
    ax.set_xlim(-10, max_step + 15)

    all_ci_hi = [ci_data[k][s][1] for k in runs for s in ci_data[k]]
    all_ci_lo = [ci_data[k][s][0] for k in runs for s in ci_data[k]]
    for lr in control_results:
        if animal in control_results.get(lr, {}):
            for s, (r, lo, hi) in control_results[lr][animal].items():
                all_ci_hi.append(hi)
                all_ci_lo.append(lo)
    if all_ci_hi:
        ymax = max(all_ci_hi)
        ymin = min(all_ci_lo)
        margin = max((ymax - ymin) * 0.15, 0.5)
        ax.set_ylim(max(0, ymin - margin), ymax + margin)

fig.suptitle('Subliminal Learning: Animal Detection Rate Over RL Training Steps',
             fontsize=15, fontweight='bold', y=0.98)
fig.text(0.5, 0.01, 'Blue/Red = biased judge (2 seeds avg)  |  Green/Purple = control (no bias)  |  Error bars = 95% Wilson CI',
         ha='center', fontsize=10, style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('results/rl_sweep/trajectories.png', dpi=150, bbox_inches='tight')
print('Saved to results/rl_sweep/trajectories.png')
