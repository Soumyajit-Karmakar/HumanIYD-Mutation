import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

plt.rcParams.update({'font.size': 22})  # Global tick label size

label_fontsize = 18
tick_fontsize = 16

def load_data_skip_remark(filename):
    with open(filename) as f:
        lines = [line for line in f if not line.strip().startswith("REMARK")]
    return np.loadtxt(lines)

# --- IMAGE 1: 1D Free Energy Profile ---
data1 = load_data_skip_remark("1D_FreeEnergyProfile_I116T.txt")
x1, y1 = data1[:, 0], data1[:, 1]

plt.figure(figsize=(8, 4))
plt.plot(x1, y1/4.2, '-', color='blue')
plt.xlabel("#Structured Residues", fontsize=label_fontsize)
plt.ylabel("Free Energy (kcal/mol)", fontsize=label_fontsize)
xticks = x1[::max(1, len(x1)//10)]
plt.xticks(ticks=xticks, labels=(xticks + 70).astype(int), fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
#plt.title("1D Free Energy Profile")
plt.grid(False)
#plt.tight_layout()
# plt.savefig("image1.png")
# plt.close()

# --- IMAGE 2: 3D Free Energy Surface ---
# --- IMAGE 2: 3D Free Energy Surface ---
data2 = load_data_skip_remark("2D_FreeEnergySurface_I116T.txt")
x2, y2, z2 = data2[:, 0], data2[:, 1], data2[:, 2] / 4.2  # Convert to kcal/mol

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the trisurf surface
surf = ax.plot_trisurf(x2, y2, z2, cmap='jet', edgecolor='none', alpha=1)

# Set labels
ax.set_xlabel("N-terminal Blocks", fontsize=label_fontsize, labelpad=10)
ax.set_ylabel("C-terminal Blocks", fontsize=label_fontsize, labelpad=10)
ax.set_zlabel("Free Energy (kcal/mol)", fontsize=label_fontsize, labelpad=10)

# Adjust view angle for better visual
ax.view_init(elev=30, azim=-40)

# Set ticks
xticks = np.arange(1, 111, 20)  # Every 20 blocks
yticks = np.arange(1, 111, 20)

ax.set_xticks(xticks)
ax.set_xticklabels((xticks + 70).astype(int), fontsize=tick_fontsize)

ax.set_yticks(yticks)
ax.set_yticklabels((yticks + 180).astype(int), fontsize=tick_fontsize)

ax.tick_params(axis='z', labelsize=tick_fontsize)

# Add colorbar
from matplotlib.cm import ScalarMappable
mappable = ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=np.nanmin(z2), vmax=np.nanmax(z2)))
mappable.set_array([])
fig.colorbar(mappable, shrink=0.6, aspect=12, pad=0.1, label="Free Energy (kcal/mol)", ax=ax)

# Optional: Set plot limits for breathing room
ax.set_xlim(0, 115)
ax.set_ylim(0, 115)


#plt.title("2D Free Energy Surface (3D View)")
#plt.tight_layout()
# plt.savefig("image2.png")
# plt.close()

# --- IMAGE 3: Residue Folding Probability vs RC (Heatmap) ---
data3 = load_data_skip_remark("ResFoldProb_vs_RC_I116T.txt")
rc_vals = np.unique(data3[:, 0])
res_vals = np.unique(data3[:, 1])
heatmap = np.zeros((len(res_vals), len(rc_vals)))

for i in range(len(data3)):
    rc_idx = np.where(rc_vals == data3[i, 0])[0][0]
    res_idx = np.where(res_vals == data3[i, 1])[0][0]
    heatmap[res_idx, rc_idx] = data3[i, 2]

adjusted_res = res_vals + 70

plt.figure(figsize=(8, 5))
plt.imshow(heatmap, aspect='auto', origin='lower', cmap='viridis',
           extent=[rc_vals[0], rc_vals[-1], adjusted_res[0], adjusted_res[-1]])
plt.colorbar(label="Folding Probability")
plt.xlabel("Reaction Coordinate (Structured Blocks)", fontsize=label_fontsize)
plt.ylabel("Residue Index", fontsize=label_fontsize)
yticks = np.linspace(adjusted_res[0], adjusted_res[-1], 6)
plt.yticks(ticks=yticks, labels=yticks.astype(int), fontsize=tick_fontsize)
xticks = x1[::max(1, len(x1)//10)]
plt.xticks(ticks=xticks, labels=(xticks + 70).astype(int), fontsize=tick_fontsize)
#plt.title("Residue Folding Probability vs RC")
plt.tight_layout()
# plt.savefig("image3.png")
# plt.close()

# --- IMAGE 4: 2D Free Energy Surface (Heatmap) ---
xi = np.unique(x2)
yi = np.unique(y2)
zi = np.full((len(yi), len(xi)), np.nan)

for i in range(len(x2)):
    xi_idx = np.where(xi == x2[i])[0][0]
    yi_idx = np.where(yi == y2[i])[0][0]
    zi[yi_idx, xi_idx] = data2[i, 2]/4.2 # Use original z2 values

plt.figure(figsize=(8, 5))
plt.imshow(zi, origin='lower', aspect='auto', extent=[xi[0], xi[-1], yi[0], yi[-1]], cmap='jet')
plt.colorbar(label='Free Energy (kcal/mol)')
plt.xlabel("N-terminal Blocks", fontsize=label_fontsize)
plt.ylabel("C-terminal Blocks", fontsize=label_fontsize)
xticks = np.arange(1, 111, 10)
yticks = np.arange(1, 111, 10)

plt.xticks(ticks=xticks, labels=(xticks + 70).astype(int), fontsize=tick_fontsize)
plt.yticks(ticks=yticks, labels=(yticks + 180).astype(int), fontsize=tick_fontsize)

#plt.title("2D Free Energy Surface (Heatmap)")
#plt.tight_layout()
# plt.savefig("image4.png")
# plt.close()

# --- IMAGE 5: Microstates Count by Model Type (Bar Chart) ---
counts = {
    'SSA': 24310,
    'DSA': 96717335,
    'DSAw/L': 90554929
}

plt.figure(figsize=(6, 4))
plt.bar(counts.keys(), counts.values(), color='blue')
plt.yscale('log')
plt.ylabel("Number of Microstates", fontsize=label_fontsize)
plt.xlabel("Model Type", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
#plt.title("Microstates by Approximation Type")
#plt.tight_layout()
# plt.savefig("image5.png")
# plt.close()


# Read microstate counts from file
filename = "Statistics_I116T.txt"

# Initialize values
numstates_SSA = 0
numstates_DSA = 0
numstates_DSAwL = 0

with open(filename, 'r') as f:
    for line in f:
        if line.startswith('SSA'):
            numstates_SSA = int(line.split(':')[1].strip())
        elif line.startswith('DSA') and 'w/L' not in line:
            numstates_DSA = int(line.split(':')[1].strip())
        elif line.startswith('DSAw/L'):
            numstates_DSAwL = int(line.split(':')[1].strip())

# Compute total
Z_total = numstates_SSA + numstates_DSA + numstates_DSAwL

# Compute partition function fractions
pf_SSA = numstates_SSA / Z_total
pf_DSA = numstates_DSA / Z_total
pf_DSAwL = numstates_DSAwL / Z_total

# Convert to percentages
print(f"% Contribution from SSA to Total Partition Function  : {pf_SSA * 100:.2f}")
print(f"% Contribution from DSA to Total Partition Function  : {pf_DSA * 100:.2f}")
print(f"% Contribution from DSAw/L to Total Partition Function  : {pf_DSAwL * 100:.2f}")

# --- IMAGE 6: Partition Function Contribution ---
# --- IMAGE 6: Partition Function Contribution (Bar Chart) ---
# Replace with actual % from Statistics.txt
contrib = {
    'SSA': 0.01,
    'DSA': 51.64,
    'DSAw/L': 48.35
}

plt.figure(figsize=(6, 4))
plt.bar(contrib.keys(), contrib.values(), color='red')
plt.ylabel("Partition Function Contribution (%)", fontsize=label_fontsize)
plt.xlabel("Model Type", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
#plt.title("Partition Function Contribution by Model Type")
#plt.tight_layout()
#plt.savefig("image6.png")
#plt.close()
