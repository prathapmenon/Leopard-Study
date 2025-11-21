import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import time
import os
from datetime import datetime
import pandas as pd

# -----------------------
# Random seed setup
# -----------------------
seed = np.random.randint(0, 10000)
np.random.seed(seed)

# -----------------------
# Parameters
# -----------------------
LANDSCAPE_SIZE = 100
NUM_LEOPARDS = 10
RESOURCE_DENSITY = 0.2
MAX_ITERS = 100
E_REQ = 10
DAMP = 0.1
CONV_THRESH = 1e-3
NUM_SIMULATIONS = 3
MAX_RADIUS = LANDSCAPE_SIZE / 2
MIN_RADIUS = 1

# -----------------------
# Functions
# -----------------------
def generate_resources(patchiness):
    num_resources = int(LANDSCAPE_SIZE**2 * RESOURCE_DENSITY)
    if patchiness == 'uniform':
        return np.random.rand(num_resources, 2) * LANDSCAPE_SIZE
    elif patchiness == 'clustered':
        num_clusters = 5
        cluster_radius = 10
        resource_points = []
        for _ in range(num_clusters):
            center = np.random.rand(2) * LANDSCAPE_SIZE
            points_in_cluster = num_resources // num_clusters
            cluster_points = center + np.random.randn(points_in_cluster, 2) * (cluster_radius / 3)
            resource_points.extend(cluster_points)
        resource_points = np.clip(resource_points, 0, LANDSCAPE_SIZE)
        return np.array(resource_points)

def resources_in_territory(center, radius, resources):
    poly = Point(center).buffer(radius)
    return sum(poly.contains(Point(r)) for r in resources)

def compute_overlap_fraction(positions, radii):
    overlap_area = 0
    total_area = np.sum(np.pi * radii**2)
    for i in range(len(positions)):
        poly_i = Point(positions[i]).buffer(radii[i])
        for j in range(i + 1, len(positions)):
            poly_j = Point(positions[j]).buffer(radii[j])
            overlap_area += poly_i.intersection(poly_j).area
    return overlap_area / total_area if total_area > 0 else 0

def compute_individual_overlap(index, positions, radii):
    """Compute how much territory of a specific leopard overlaps with others."""
    poly_i = Point(positions[index]).buffer(radii[index])
    overlap_area = 0
    for j in range(len(positions)):
        if j == index:
            continue
        poly_j = Point(positions[j]).buffer(radii[j])
        overlap_area += poly_i.intersection(poly_j).area
    return overlap_area / (np.pi * radii[index]**2 + 1e-6)

def run_simulation(patchiness):
    resource_points = generate_resources(patchiness)
    leopard_positions = np.random.rand(NUM_LEOPARDS, 2) * LANDSCAPE_SIZE
    leopard_radii = np.ones(NUM_LEOPARDS) * 5

    for _ in range(MAX_ITERS):
        radii_prev = leopard_radii.copy()
        for i in range(NUM_LEOPARDS):
            current_resources = resources_in_territory(leopard_positions[i], leopard_radii[i], resource_points)
            overlap_penalty = compute_individual_overlap(i, leopard_positions, leopard_radii)
            
            # Moderate overlap penalty (not too strict, but noticeable)
            factor = (E_REQ / (current_resources + 1e-6)) * (1 + 3 * overlap_penalty)
            
            leopard_radii[i] *= (1 + DAMP * (factor - 1))
            leopard_radii[i] = np.clip(leopard_radii[i], MIN_RADIUS, MAX_RADIUS)
        
        if np.max(np.abs(leopard_radii - radii_prev)) < CONV_THRESH:
            break

    mean_radius = np.mean(leopard_radii)
    overlap_fraction = compute_overlap_fraction(leopard_positions, leopard_radii)
    radius_variance = np.var(leopard_radii)

    return mean_radius, overlap_fraction, radius_variance, leopard_positions, leopard_radii, resource_points

# -----------------------
# Run Experiment
# -----------------------
total_start = time.time()
IV_conditions = ['uniform', 'clustered']
results = {}

# Create data folder if it doesn't exist
if not os.path.exists("data"):
    os.mkdir("data")

# Create a new folder for this run
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_folder = os.path.join("data", f"run_{timestamp}")
os.mkdir(run_folder)

for patchiness in IV_conditions:
    patch_start = time.time()
    results[patchiness] = []
    print(f"\nRunning simulations for patchiness: {patchiness}")
    for sim in range(NUM_SIMULATIONS):
        mean_r, overlap, var_r, positions, radii, resources = run_simulation(patchiness)
        results[patchiness].append({
            'mean_radius': mean_r,
            'overlap_fraction': overlap,
            'radius_variance': var_r,
            'positions': positions,
            'radii': radii,
            'resources': resources
        })

    # Calculate averages
    mean_radii = np.mean([res['mean_radius'] for res in results[patchiness]])
    overlaps = np.mean([res['overlap_fraction'] for res in results[patchiness]])
    var_radii = np.mean([res['radius_variance'] for res in results[patchiness]])

    # Save them for CSV
    if patchiness == "uniform":
        uniform_mean_radius = mean_radii
        uniform_overlap_fraction = overlaps
        uniform_radius_variance = var_radii
        uniform_time = time.time() - patch_start
    else:
        clustered_mean_radius = mean_radii
        clustered_overlap_fraction = overlaps
        clustered_radius_variance = var_radii
        clustered_time = time.time() - patch_start

    print(f"Mean territory radius (avg over sims): {mean_radii:.2f}")
    print(f"Overlap fraction (avg over sims): {overlaps:.2f}")
    print(f"Radius variance (avg over sims): {var_radii:.2f}")
    print(f"Time for {patchiness} simulations: {time.time() - patch_start:.2f} seconds")

    # Save plot of last simulation
    last = results[patchiness][-1]
    plt.figure(figsize=(6,6))
    plt.scatter(last['resources'][:,0], last['resources'][:,1], c='green', s=10, label='Resources')
    for i in range(NUM_LEOPARDS):
        circle = plt.Circle(last['positions'][i], last['radii'][i], color='red', fill=False, lw=2)
        plt.gca().add_patch(circle)
    plt.xlim(0, LANDSCAPE_SIZE)
    plt.ylim(0, LANDSCAPE_SIZE)
    plt.legend()
    plt.title(f"Patchiness: {patchiness} (last simulation)")
    plt.savefig(os.path.join(run_folder, f"{patchiness}.png"))
    plt.close()

total_end = time.time()
print(f"\nAll simulations completed in {total_end - total_start:.2f} seconds.")
print(f"Plots saved in folder: {run_folder}")

# === SAVE RESULTS TO CSV ===
results_summary = {
    "DateTime": [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    "Patchiness": ["Uniform", "Clustered"],
    "Mean_Territory_Radius": [uniform_mean_radius, clustered_mean_radius],
    "Overlap_Fraction": [uniform_overlap_fraction, clustered_overlap_fraction],
    "Radius_Variance": [uniform_radius_variance, clustered_radius_variance],
    "Simulation_Time": [uniform_time, clustered_time],
    "Random_Seed": [seed, seed]
}

df = pd.DataFrame(results_summary)
csv_path = os.path.join(run_folder, "simulation_results.csv")
df.to_csv(csv_path, index=False)
print(f"Results saved to: {csv_path}")
