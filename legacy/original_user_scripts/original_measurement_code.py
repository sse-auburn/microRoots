# ============================================================
# MEASUREMENT CODE
# ============================================================

import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import networkx as nx
from scipy.ndimage import convolve
from pathlib import Path
import random

# Conversion factor (μm per pixel)
CONVERSION_FACTOR = 0.495

# ============================================================
# PATHS
# ============================================================

PREDICTED_MASKS_DIR = Path("/mnt/c/wsl_projects/yolosam3ft4paper/02_SAM3_Segmentation/instance_masks")
OUTPUT_DIR = Path("/mnt/c/wsl_projects/yolosam3ft4paper/05_Visualization_Comparison/final_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS (Your exact code for root hairs)
# ============================================================

def get_endpoints(skel):
    """Identify endpoints in a binary skeleton."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    endpoints = (skel == 1) & ((neighbor_count - skel) == 1)
    return endpoints

def prune_skeleton(skel, prune_threshold=5):
    """Remove short spur branches from a skeleton."""
    pruned = skel.copy().astype(np.uint8)
    endpoints_img = get_endpoints(pruned)
    ep_coords = np.column_stack(np.nonzero(endpoints_img))

    for (r, c) in ep_coords:
        branch_coords = [(r, c)]
        current = (r, c)
        length = 0
        while True:
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = current[0] + dr, current[1] + dc
                    if nr < 0 or nr >= pruned.shape[0] or nc < 0 or nc >= pruned.shape[1]:
                        continue
                    if pruned[nr, nc] and (nr, nc) not in branch_coords:
                        neighbors.append((nr, nc))
            if len(neighbors) != 1:
                break
            current = neighbors[0]
            branch_coords.append(current)
            length += 1
            if length > prune_threshold:
                break
        if length <= prune_threshold:
            for (rr, cc) in branch_coords:
                pruned[rr, cc] = 0
    return pruned

def extract_primary_skeleton(skel):
    """Extract the longest path from a skeleton."""
    coords = np.column_stack(np.nonzero(skel))
    if len(coords) == 0:
        return np.array([])

    G = nx.Graph()
    for (r, c) in coords:
        G.add_node((r, c))
    for (r, c) in coords:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in G:
                    dist = np.sqrt(dr**2 + dc**2)
                    G.add_edge((r, c), (nr, nc), weight=dist)

    endpoints = [n for n in G.nodes() if G.degree(n) == 1]
    if len(endpoints) < 2:
        nodes = list(G.nodes())
        if len(nodes) == 0:
            return np.array([])
        start = nodes[0]
        lengths = nx.single_source_dijkstra_path_length(G, start)
        farthest = max(lengths, key=lengths.get)
        endpoints = [start, farthest]

    max_path = []
    max_length = 0
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            try:
                path = nx.shortest_path(G, endpoints[i], endpoints[j], weight='weight')
                path_length = nx.shortest_path_length(G, endpoints[i], endpoints[j], weight='weight')
                if path_length > max_length:
                    max_length = path_length
                    max_path = path
            except nx.NetworkXNoPath:
                continue
    return np.array(max_path)

def measure_width_at_point(pt, tangent, binary_mask, max_distance=20):
    """Measure width perpendicular to skeleton at a point."""
    norm = np.linalg.norm(tangent)
    if norm == 0:
        return 0, pt, pt
    t = tangent / norm
    normal = np.array([-t[1], t[0]])

    pos_distance = 0
    for d in np.arange(0, max_distance + 1):
        sample_pt = pt + d * normal
        sample_pt_int = np.round(sample_pt).astype(int)
        if (sample_pt_int[0] < 0 or sample_pt_int[0] >= binary_mask.shape[0] or
            sample_pt_int[1] < 0 or sample_pt_int[1] >= binary_mask.shape[1]):
            break
        if not binary_mask[sample_pt_int[0], sample_pt_int[1]]:
            break
        pos_distance = d

    neg_distance = 0
    for d in np.arange(0, max_distance + 1):
        sample_pt = pt - d * normal
        sample_pt_int = np.round(sample_pt).astype(int)
        if (sample_pt_int[0] < 0 or sample_pt_int[0] >= binary_mask.shape[0] or
            sample_pt_int[1] < 0 or sample_pt_int[1] >= binary_mask.shape[1]):
            break
        if not binary_mask[sample_pt_int[0], sample_pt_int[1]]:
            break
        neg_distance = d

    width = pos_distance + neg_distance
    pos_boundary = pt + pos_distance * normal
    neg_boundary = pt - neg_distance * normal
    return width, pos_boundary, neg_boundary

# ============================================================
# MAIN STEM MEASUREMENT
# ============================================================

def extend_line_to_image_edges(point, direction, img_width, img_height):
    """Extend line to image boundaries."""
    candidates = []

    if direction[0] != 0:
        t = -point[0] / direction[0]
        y = point[1] + t * direction[1]
        if 0 <= y <= img_height - 1:
            candidates.append((t, np.array([0, y])))

        t = (img_width - 1 - point[0]) / direction[0]
        y = point[1] + t * direction[1]
        if 0 <= y <= img_height - 1:
            candidates.append((t, np.array([img_width - 1, y])))

    if direction[1] != 0:
        t = -point[1] / direction[1]
        x = point[0] + t * direction[0]
        if 0 <= x <= img_width - 1:
            candidates.append((t, np.array([x, 0])))

        t = (img_height - 1 - point[1]) / direction[1]
        x = point[0] + t * direction[0]
        if 0 <= x <= img_width - 1:
            candidates.append((t, np.array([x, img_height - 1])))

    if len(candidates) < 2:
        return point, point

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[-1][1]

def get_outer_contour_side(contour, main_centroid_xy, img_height, img_width):
    """
    When no root hairs exist, find the OUTER side of the main stem.
    The outer side is the one farther from the image center.

    Returns points on the outer side of the contour.
    """
    # Image center
    img_center = np.array([img_width / 2, img_height / 2])

    # Direction from image center to main stem centroid
    direction_from_center = main_centroid_xy - img_center
    if np.linalg.norm(direction_from_center) > 0:
        direction_from_center = direction_from_center / np.linalg.norm(direction_from_center)
    else:
        # Main stem is at center - use the side with more contour points
        direction_from_center = np.array([1, 0])  # Default to right side

    # The OUTER side is the side AWAY from the image center
    # So we want points where (point - centroid) · direction_from_center > 0
    outer_points = []
    for pt in contour:
        vec = pt - main_centroid_xy
        if np.linalg.norm(vec) > 0:
            dot = np.dot(vec / np.linalg.norm(vec), direction_from_center)
            if dot > 0.2:  # On the outer side
                outer_points.append(pt)

    # If not enough points on outer side, try the opposite side
    if len(outer_points) < 10:
        outer_points = []
        for pt in contour:
            vec = pt - main_centroid_xy
            if np.linalg.norm(vec) > 0:
                dot = np.dot(vec / np.linalg.norm(vec), -direction_from_center)
                if dot > 0.2:
                    outer_points.append(pt)

    return np.array(outer_points) if outer_points else contour

def measure_main_stem(main_mask, img_rgb, root_hair_masks, img_height, img_width):
    """
    Measure main stem using best-fit line extended to image edges.

    - If root hairs exist: measure on root hair side
    - If NO root hairs: measure on outer side of main stem
    """
    # Get main stem centroid
    main_coords = np.column_stack(np.where(main_mask))
    if len(main_coords) == 0:
        return 0, None
    main_centroid = np.mean(main_coords, axis=0)  # (row, col)
    main_centroid_xy = np.array([main_centroid[1], main_centroid[0]])  # (x, y)

    # Get contour
    contours, _ = cv2.findContours((main_mask * 255).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return 0, None

    contour = max(contours, key=cv2.contourArea)[:, 0, :]  # (x, y)

    # Check if root hairs exist
    has_root_hairs = len(root_hair_masks) > 0 and any(np.sum(m) > 0 for m in root_hair_masks)

    if has_root_hairs:
        # USE ROOT HAIR SIDE
        print("    Mode: Measuring on ROOT HAIR side")

        # Get average position of root hairs
        rh_centroids = []
        for rh_mask in root_hair_masks:
            coords = np.column_stack(np.where(rh_mask))
            if len(coords) > 0:
                rh_centroids.append(np.mean(coords, axis=0))

        avg_rh_position = np.mean(rh_centroids, axis=0)  # (row, col)
        direction_to_rh = avg_rh_position - main_centroid
        direction_to_rh = direction_to_rh / np.linalg.norm(direction_to_rh)
        direction_xy = np.array([direction_to_rh[1], direction_to_rh[0]])  # (x, y)

        # Select points on root hair side
        side_points = []
        for pt in contour:
            vec = pt - main_centroid_xy
            if np.linalg.norm(vec) > 0:
                dot = np.dot(vec / np.linalg.norm(vec), direction_xy)
                if dot > 0.2:
                    side_points.append(pt)

        side_points = np.array(side_points) if side_points else contour

    else:
        # NO ROOT HAIRS - USE OUTER SIDE
        print("    Mode: No root hairs - Measuring on OUTER side")
        side_points = get_outer_contour_side(contour, main_centroid_xy, img_height, img_width)

    if len(side_points) < 2:
        # Fallback: use entire contour
        print("    Warning: Not enough side points, using full contour")
        side_points = contour

    print(f"    Using {len(side_points)} contour points for fitting")

    # PCA to find principal axis
    mean_pt = np.mean(side_points, axis=0)
    centered = side_points - mean_pt
    cov = np.cov(centered, rowvar=False)

    if cov.ndim < 2:
        cov = np.array([[cov, 0], [0, cov]])

    eigenvalues, eigenvectors = np.linalg.eig(cov)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)].real
    principal_axis = principal_axis / np.linalg.norm(principal_axis)

    # Extend to image edges
    pt1, pt2 = extend_line_to_image_edges(mean_pt, principal_axis, img_width, img_height)

    # Calculate length
    length_px = np.linalg.norm(pt2 - pt1)

    return length_px, (pt1, pt2)

# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def process_predicted_mask(image_path):
    """Process a single predicted mask and return measurements."""

    print(f"\nProcessing: {image_path.name}")

    # Load image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"  Error: Could not load image")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    annotated_img = img_rgb.copy()
    height, width_img, _ = img_rgb.shape

    print(f"  Size: {width_img}x{height}")

    # Get instances
    background_color = np.array([0, 0, 0])
    pixels = img_rgb.reshape((-1, 3))
    unique_colors = np.unique(pixels, axis=0)

    # Process each instance
    aggregated_results = {}

    for color in unique_colors:
        if np.all(color == background_color):
            continue

        color_tuple = tuple(color)
        mask_color = np.all(img_rgb == color, axis=2).astype(np.uint8)
        labeled_mask = label(mask_color, connectivity=2)
        regions = regionprops(labeled_mask)

        total_length = 0
        all_widths = []
        all_skeletons = []
        all_boundaries = []
        total_area = np.sum(mask_color)

        for region in regions:
            component_mask = (labeled_mask == region.label).astype(np.uint8)
            skel = skeletonize(component_mask.astype(bool))
            skel = skel.astype(np.uint8)
            skel_pruned = prune_skeleton(skel, prune_threshold=5)
            labeled_skel = label(skel_pruned, connectivity=2)
            regions_skel = regionprops(labeled_skel)

            if len(regions_skel) == 0:
                continue

            main_region = max(regions_skel, key=lambda r: r.area)
            skel_component = (labeled_skel == main_region.label).astype(np.uint8)
            primary_path = extract_primary_skeleton(skel_component)

            if primary_path.size == 0:
                continue

            region_length = 0
            widths_region = []
            boundaries_region = []

            for i in range(len(primary_path)):
                pt = primary_path[i].astype(np.float64)
                if i == 0:
                    tangent = primary_path[min(i + 1, len(primary_path) - 1)] - primary_path[i]
                elif i == len(primary_path) - 1:
                    tangent = primary_path[i] - primary_path[i - 1]
                else:
                    tangent = primary_path[i + 1] - primary_path[i - 1]

                w, pos_b, neg_b = measure_width_at_point(pt, tangent, component_mask, max_distance=20)
                widths_region.append(w)
                boundaries_region.append((pt, pos_b, neg_b))

                if i > 0:
                    region_length += np.linalg.norm(primary_path[i] - primary_path[i - 1])

            total_length += region_length
            all_widths.extend(widths_region)
            all_skeletons.append(primary_path)
            all_boundaries.append(boundaries_region)

        avg_width = np.mean(all_widths) if len(all_widths) > 0 else 0

        aggregated_results[color_tuple] = {
            'mask': mask_color,
            'skeleton_length': total_length,
            'avg_width': avg_width,
            'skeletons': all_skeletons,
            'boundaries': all_boundaries,
            'color': color,
            'area': total_area
        }

    print(f"  Found {len(aggregated_results)} instances")

    if len(aggregated_results) == 0:
        return None

    # Identify main stem (largest by area)
    main_stem_key = max(aggregated_results.keys(), key=lambda k: aggregated_results[k]['area'])

    # Get root hair masks (all instances except main stem)
    root_hair_masks = [res['mask'] for key, res in aggregated_results.items() if key != main_stem_key]

    num_root_hairs = len(root_hair_masks)
    print(f"  Main Stem Area: {aggregated_results[main_stem_key]['area']} pixels")
    print(f"  Root Hairs: {num_root_hairs}")

    # Measure main stem
    main_mask = aggregated_results[main_stem_key]['mask']
    main_line_length, best_fit_line = measure_main_stem(
        main_mask, img_rgb, root_hair_masks, height, width_img
    )

    if main_line_length > 0:
        aggregated_results[main_stem_key]['total_length'] = main_line_length
        aggregated_results[main_stem_key]['is_main_stem'] = True
        aggregated_results[main_stem_key]['best_fit_line'] = best_fit_line
        aggregated_results[main_stem_key]['avg_width'] = None
    else:
        aggregated_results[main_stem_key]['total_length'] = aggregated_results[main_stem_key]['skeleton_length']
        aggregated_results[main_stem_key]['is_main_stem'] = True
        aggregated_results[main_stem_key]['best_fit_line'] = None

    # Set root hairs
    for key, res in aggregated_results.items():
        if 'total_length' not in res:
            res['total_length'] = res['skeleton_length']
            res['is_main_stem'] = False

    # Assign IDs
    non_main_counter = 1
    for key, res in aggregated_results.items():
        if res.get('is_main_stem', False):
            res['id'] = "Main Stem"
        else:
            res['id'] = str(non_main_counter)
            non_main_counter += 1

    # Calculate statistics
    main_stem_length_um = aggregated_results[main_stem_key]['total_length'] * CONVERSION_FACTOR
    density = num_root_hairs / main_stem_length_um if main_stem_length_um != 0 else 0

    print(f"  Main Stem Length: {main_stem_length_um:.1f} μm")

    # ============================================================
    # ANNOTATE IMAGE
    # ============================================================

    for res in aggregated_results.values():
        col = res['color']

        if res.get('is_main_stem', False):
            draw_color = (255, 0, 0)  # Red for main stem
            if res.get('best_fit_line') is not None:
                pt1, pt2 = res['best_fit_line']
                cv2.line(annotated_img,
                         (int(round(pt1[0])), int(round(pt1[1]))),
                         (int(round(pt2[0])), int(round(pt2[1]))),
                         draw_color, 3)
                cv2.circle(annotated_img, (int(pt1[0]), int(pt1[1])), 10, draw_color, -1)
                cv2.circle(annotated_img, (int(pt2[0]), int(pt2[1])), 10, draw_color, -1)
        else:
            draw_color = (0, 255, 0)  # Green for root hairs
            for skel_arr in res['skeletons']:
                for pt in skel_arr:
                    r, c = int(pt[0]), int(pt[1])
                    cv2.circle(annotated_img, (c, r), 1, draw_color, -1)

            for boundary_set in res['boundaries']:
                for i, (pt, pos_b, neg_b) in enumerate(boundary_set):
                    if i % 5 != 0:
                        continue
                    pos_int = (int(round(pos_b[1])), int(round(pos_b[0])))
                    neg_int = (int(round(neg_b[1])), int(round(neg_b[0])))
                    cv2.line(annotated_img, pos_int, neg_int, (0, 0, 255), 1)

        if res['skeletons']:
            mid_pt = res['skeletons'][0][len(res['skeletons'][0]) // 2]
            cv2.putText(annotated_img, res['id'], (int(mid_pt[1]), int(mid_pt[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_img, res['id'], (int(mid_pt[1]), int(mid_pt[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw measurement table
    table_lines = ["MEASUREMENTS"]
    table_lines.append("Root       Length(um)  Width(um)")

    ordered_keys = [main_stem_key] + [k for k in aggregated_results.keys() if k != main_stem_key]

    for key in ordered_keys[:15]:
        res = aggregated_results[key]
        length_um = res['total_length'] * CONVERSION_FACTOR
        if res.get('is_main_stem', False):
            width_str = "-"
        else:
            width_um = res['avg_width'] * CONVERSION_FACTOR
            width_str = f"{width_um:.1f}"
        table_lines.append(f"{res['id']:<10}{length_um:>10.1f}{width_str:>10}")

    if len(ordered_keys) > 15:
        table_lines.append(f"... +{len(ordered_keys) - 15} more")

    table_lines.append(f"Density: {density:.4f}/um")

    margin = 20
    line_height = 25
    table_width = 320
    table_height = line_height * (len(table_lines) + 1)

    cv2.rectangle(annotated_img,
                  (width_img - table_width - margin, margin),
                  (width_img - margin, margin + table_height),
                  (255, 255, 255), -1)
    cv2.rectangle(annotated_img,
                  (width_img - table_width - margin, margin),
                  (width_img - margin, margin + table_height),
                  (0, 0, 0), 2)

    for i, line in enumerate(table_lines):
        y = margin + (i + 1) * line_height
        x = width_img - table_width
        cv2.putText(annotated_img, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 128), 1, cv2.LINE_AA)

    # Collect final measurements
    rh_lengths = [res['total_length'] * CONVERSION_FACTOR
                  for res in aggregated_results.values() if not res.get('is_main_stem', False)]
    rh_widths = [res['avg_width'] * CONVERSION_FACTOR
                 for res in aggregated_results.values()
                 if not res.get('is_main_stem', False) and res['avg_width'] > 0]

    measurements = {
        'main_stem_length_um': main_stem_length_um,
        'num_root_hairs': num_root_hairs,
        'density': density,
        'avg_rh_length_um': np.mean(rh_lengths) if rh_lengths else 0,
        'avg_rh_width_um': np.mean(rh_widths) if rh_widths else 0,
        'aggregated_results': aggregated_results
    }

    return {
        'raw_image': img_rgb,
        'annotated_image': annotated_img,
        'measurements': measurements
    }

# ============================================================
# MAIN EXECUTION
# ============================================================

print("=" * 70)
print("FINAL MEASUREMENT CODE")
print("- With root hairs: measure on root hair side")
print("- Without root hairs: measure on outer side")
print("=" * 70)

# Select one random predicted mask
pred_files = sorted(PREDICTED_MASKS_DIR.glob("*_mask.png"))
print(f"Found {len(pred_files)} predicted masks")

random.seed()
selected_file = random.choice(pred_files)

# Process
result = process_predicted_mask(selected_file)

if result is not None:
    # Save outputs
    cv2.imwrite(str(OUTPUT_DIR / "raw_mask.png"),
                cv2.cvtColor(result['raw_image'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUTPUT_DIR / "annotated.png"),
                cv2.cvtColor(result['annotated_image'], cv2.COLOR_RGB2BGR))

    # Show
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(result['raw_image'])
    axes[0].set_title('Raw Mask', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(result['annotated_image'])
    axes[1].set_title('Annotated', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    m = result['measurements']
    plt.suptitle(f"{selected_file.stem}\nMS: {m['main_stem_length_um']:.1f}μm, RH: {m['num_root_hairs']}, "
                 f"Avg Len: {m['avg_rh_length_um']:.1f}μm, Avg Width: {m['avg_rh_width_um']:.1f}μm",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "result.png"), dpi=150, bbox_inches='tight')
    plt.show()

    # Print results
    print("\n" + "=" * 70)
    print("FINAL MEASUREMENTS")
    print("=" * 70)
    print(f"Image: {selected_file.stem}")
    print(f"Main Stem Length: {m['main_stem_length_um']:.1f} μm")
    print(f"Root Hair Count: {m['num_root_hairs']}")
    print(f"Avg RH Length: {m['avg_rh_length_um']:.1f} μm")
    print(f"Avg RH Width: {m['avg_rh_width_um']:.1f} μm")
    print(f"Density: {m['density']:.6f} /μm")
    print(f"\nOutput: {OUTPUT_DIR}")
    print("=" * 70)
else:
    print("Failed to process image!")
