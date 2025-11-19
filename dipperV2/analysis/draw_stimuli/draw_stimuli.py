#%%
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# ---- your stimuli dict (copy/paste) ----
stimuli = {
    "target": [
        {"object": "line", "type": "target", "pos": [0, 0], "angle": 90, "length": 18},
    ],

    "single_flanker_top": [
        {"object": "line", "type": "flanker", "pos": [0, 27], "angle": 90, "length": 18},
        {"object": "line", "type": "target",  "pos": [0, 0],  "angle": 90, "length": 18},
        {"object": "line", "type": "flanker", "pos": [0,-27], "angle": 90, "length": 18},
    ],

    "single_flanker_side_orth": [
        {"object": "line", "type": "flanker", "pos": [-27, 0], "angle": 0,  "length": 18},
        {"object": "line", "type": "target",  "pos": [0, 0],   "angle": 90, "length": 18},
        {"object": "line", "type": "flanker", "pos": [27, 0],  "angle": 0,  "length": 18},
    ],

    "triple_flanker": [
        {"object": "line", "type": "flanker", "pos": [-27, 27], "angle": 90, "length": 18},
        {"object": "line", "type": "flanker", "pos": [0, 27],   "angle": 90, "length": 18},
        {"object": "line", "type": "flanker", "pos": [27, 27],  "angle": 90, "length": 18},

        {"object": "line", "type": "flanker", "pos": [-27, 0],  "angle": 90, "length": 18},
        {"object": "line", "type": "target",  "pos": [0, 0],    "angle": 90, "length": 18},
        {"object": "line", "type": "flanker", "pos": [27, 0],   "angle": 90, "length": 18},

        {"object": "line", "type": "flanker", "pos": [-27,-27], "angle": 90, "length": 18},
        {"object": "line", "type": "flanker", "pos": [0,-27],   "angle": 90, "length": 18},
        {"object": "line", "type": "flanker", "pos": [27,-27],  "angle": 90, "length": 18},
    ],

    "triple_flanker_orth": [
        {"object": "line", "type": "flanker", "pos": [-27, 27], "angle": 0,  "length": 18},
        {"object": "line", "type": "flanker", "pos": [0, 27],   "angle": 90, "length": 18},
        {"object": "line", "type": "flanker", "pos": [27, 27],  "angle": 0,  "length": 18},

        {"object": "line", "type": "flanker", "pos": [-27, 0],  "angle": 0,  "length": 18},
        {"object": "line", "type": "target",  "pos": [0, 0],    "angle": 90, "length": 18},
        {"object": "line", "type": "flanker", "pos": [27, 0],   "angle": 0,  "length": 18},

        {"object": "line", "type": "flanker", "pos": [-27,-27], "angle": 0,  "length": 18},
        {"object": "line", "type": "flanker", "pos": [0,-27],   "angle": 90, "length": 18},
        {"object": "line", "type": "flanker", "pos": [27,-27],  "angle": 0,  "length": 18},
    ]
}

# ---- helper to compute endpoints ----
def line_endpoints(pos, angle_deg, length):
    theta = np.deg2rad(angle_deg)
    dx = np.cos(theta) * length / 2.0
    dy = np.sin(theta) * length / 2.0
    x0, y0 = pos[0] - dx, pos[1] - dy
    x1, y1 = pos[0] + dx, pos[1] + dy
    return (x0, y0, x1, y1)

# ---- compute global extents so all panels have same scale ----
all_x = []
all_y = []

for comps in stimuli.values():
    for c in comps:
        x0, y0, x1, y1 = line_endpoints(c['pos'], c['angle'], c['length'])
        all_x.extend([x0, x1])
        all_y.extend([y0, y1])

all_x = np.array(all_x)
all_y = np.array(all_y)

# symmetric padding around center
pad = 100  # manual padding in same units as positions/length (increase if needed)
xmin_global = all_x.min() - pad
xmax_global = all_x.max() + pad
ymin_global = all_y.min() - pad
ymax_global = all_y.max() + pad

# enforce square view (equal ranges) and centered on 0
half_range = max(abs(xmin_global), abs(xmax_global), abs(ymin_global), abs(ymax_global))
xmin = -half_range
xmax = half_range
ymin = -half_range
ymax = half_range

# ---- plotting grid ----
names = list(stimuli.keys())
n = len(names)
cols = 3
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
axes = axes.flatten()

for idx, name in enumerate(names):
    ax = axes[idx]
    components = stimuli[name]

    # draw each line using exact endpoints
    for comp in components:
        x0, y0, x1, y1 = line_endpoints(comp['pos'], comp['angle'], comp['length'])
        color = 'black' if comp['type'] == 'target' else 'black'
        lw = 1.0 if comp['type'] == 'target' else 1.0
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, solid_capstyle='round')

    # draw a faint crosshair at (0,0) for reference
    # ax.plot([ -1, 1], [0, 0], color='lightgray', linewidth=0.6)
    # ax.plot([0, 0], [-1, 1], color='lightgray', linewidth=0.6)

    ax.set_title(name, fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()  # uncomment to match PsychoPy (y increases downwards)

# turn off unused axes
for j in range(idx + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
# %%
