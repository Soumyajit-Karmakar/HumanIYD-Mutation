import mdtraj as md
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------- USER SETTINGS --------------------
WT_TRAJ = 'hIYD_WT_dry_protein.xtc'
WT_PDB = 'hIYD_WT_minpca.pdb'
MUT_TRAJ = 'hIYD_R101W_dry_protein.xtc'
MUT_PDB = 'hIYD_R101W_minpca.pdb'
MUTATION_SITE = 31  # residue number for R31W
DISTANCE_CUTOFF = 6.0  # Å
MIN_CONTACTS = 4
CONTACT_ENERGY = -46
# -------------------------------------------------------


def build_graph(frame_xyz, topology):
    """Build contact graph for one frame"""
    residues = [res for res in topology.residues if res.is_protein]
    G = nx.Graph()
    for res in residues:
        G.add_node(res.resSeq)

    # Precompute heavy atom indices for each residue
    res_atoms = []
    for res in residues:
        heavy = [atom.index for atom in res.atoms if atom.element.symbol != "H"]
        res_atoms.append(heavy)

    # Loop over residue pairs
    for i, res_i in enumerate(residues):
        for j in range(i + 1, len(residues)):
            res_j = residues[j]

            coords_i = frame_xyz[res_atoms[i]]
            coords_j = frame_xyz[res_atoms[j]]

            distances = np.linalg.norm(
                coords_i[:, np.newaxis, :] - coords_j[np.newaxis, :, :],
                axis=-1
            )

            num_contacts = np.sum(distances <= DISTANCE_CUTOFF)
            if num_contacts > MIN_CONTACTS:
                G.add_edge(res_i.resSeq, res_j.resSeq)

    return G


def compute_mean_betweenness(traj_file, pdb_file):
    """Compute mean BC over all frames"""
    traj = md.load(traj_file, top=pdb_file)
    all_bc = []

    for frame in tqdm(traj):
        G = build_graph(frame.xyz[0], traj.topology)
        bc = nx.betweenness_centrality(G)
        all_bc.append(bc)

    # Put in DataFrame for easy averaging
    df = pd.DataFrame(all_bc).fillna(0)
    mean_bc = df.mean().to_dict()
    return mean_bc


# Compute WT
mean_bc_wt = compute_mean_betweenness(WT_TRAJ, WT_PDB)
pd.Series(mean_bc_wt).to_csv("WT_CB.csv")

# Compute MUT
mean_bc_mut = compute_mean_betweenness(MUT_TRAJ, MUT_PDB)
pd.Series(mean_bc_mut).to_csv("R101W_CB.csv")

# Compute delta
all_residues = set(mean_bc_wt.keys()).union(set(mean_bc_mut.keys()))
delta_bc = {}
for res in all_residues:
    delta_bc[res] = mean_bc_mut.get(res, 0) - mean_bc_wt.get(res, 0)

pd.Series(delta_bc).to_csv("WT_R101W_Delta_CB.csv")

# Compute distances to mutation site
traj_ref = md.load(WT_PDB)
ca_indices = traj_ref.topology.select("name CA")
coords = traj_ref.xyz[0][ca_indices]
mutation_idx = [res.index for res in traj_ref.topology.residues if res.resSeq == MUTATION_SITE][0]

mutation_ca = coords[mutation_idx]

distances = []
delta_bc_values = []

for i, res in enumerate(traj_ref.topology.residues):
    if not res.is_protein:
        continue
    ca = coords[res.index]
    dist = np.linalg.norm(ca - mutation_ca)
    distances.append(dist)
    delta_bc_values.append(delta_bc.get(res.resSeq, 0))

# Save distances vs delta BC
df = pd.DataFrame({'Residue': [r.resSeq for r in traj_ref.topology.residues if r.is_protein],
                   'Distance': distances,
                   'Delta_BC': delta_bc_values})
df.to_csv("Dist_vs_Delta_CB.csv", index=False)
