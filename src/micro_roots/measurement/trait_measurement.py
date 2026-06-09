from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from tqdm import tqdm

from micro_roots.utils.io_utils import ensure_dirs, save_image_rgb


@dataclass
class MeasurementConfig:
    conversion_factor: float = 0.495
    skeleton_prune_threshold_px: int = 5
    width_max_distance_px: int = 20


def get_endpoints(skel: np.ndarray) -> np.ndarray:
    """Identify endpoints in a binary skeleton."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    endpoints = (skel == 1) & ((neighbor_count - skel) == 1)
    return endpoints


def prune_skeleton(skel: np.ndarray, prune_threshold: int = 5) -> np.ndarray:
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


def extract_primary_skeleton(skel: np.ndarray) -> np.ndarray:
    """Extract the longest path from a skeleton."""
    coords = np.column_stack(np.nonzero(skel))
    if len(coords) == 0:
        return np.array([])

    G = nx.Graph()
    coord_set = {tuple(x) for x in coords}
    for (r, c) in coords:
        G.add_node((int(r), int(c)))
    for (r, c) in coords:
        r, c = int(r), int(c)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in coord_set:
                    dist = float(np.sqrt(dr**2 + dc**2))
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
    max_length = 0.0
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            try:
                path = nx.shortest_path(G, endpoints[i], endpoints[j], weight="weight")
                path_length = nx.shortest_path_length(G, endpoints[i], endpoints[j], weight="weight")
                if path_length > max_length:
                    max_length = float(path_length)
                    max_path = path
            except nx.NetworkXNoPath:
                continue
    return np.array(max_path)


def measure_width_at_point(
    pt: np.ndarray,
    tangent: np.ndarray,
    binary_mask: np.ndarray,
    max_distance: int = 20,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Measure width perpendicular to skeleton at a point."""
    norm = np.linalg.norm(tangent)
    if norm == 0:
        return 0.0, pt, pt
    t = tangent / norm
    normal = np.array([-t[1], t[0]])

    pos_distance = 0.0
    for d in np.arange(0, max_distance + 1):
        sample_pt = pt + d * normal
        sample_pt_int = np.round(sample_pt).astype(int)
        if (
            sample_pt_int[0] < 0
            or sample_pt_int[0] >= binary_mask.shape[0]
            or sample_pt_int[1] < 0
            or sample_pt_int[1] >= binary_mask.shape[1]
        ):
            break
        if not binary_mask[sample_pt_int[0], sample_pt_int[1]]:
            break
        pos_distance = float(d)

    neg_distance = 0.0
    for d in np.arange(0, max_distance + 1):
        sample_pt = pt - d * normal
        sample_pt_int = np.round(sample_pt).astype(int)
        if (
            sample_pt_int[0] < 0
            or sample_pt_int[0] >= binary_mask.shape[0]
            or sample_pt_int[1] < 0
            or sample_pt_int[1] >= binary_mask.shape[1]
        ):
            break
        if not binary_mask[sample_pt_int[0], sample_pt_int[1]]:
            break
        neg_distance = float(d)

    width = pos_distance + neg_distance
    pos_boundary = pt + pos_distance * normal
    neg_boundary = pt - neg_distance * normal
    return width, pos_boundary, neg_boundary


def extend_line_to_image_edges(
    point: np.ndarray,
    direction: np.ndarray,
    img_width: int,
    img_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
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


def get_outer_contour_side(
    contour: np.ndarray,
    main_centroid_xy: np.ndarray,
    img_height: int,
    img_width: int,
) -> np.ndarray:
    """When no root hairs exist, select the outer side of the main stem contour."""
    img_center = np.array([img_width / 2, img_height / 2])
    direction_from_center = main_centroid_xy - img_center
    if np.linalg.norm(direction_from_center) > 0:
        direction_from_center = direction_from_center / np.linalg.norm(direction_from_center)
    else:
        direction_from_center = np.array([1, 0])

    outer_points = []
    for pt in contour:
        vec = pt - main_centroid_xy
        if np.linalg.norm(vec) > 0:
            dot = np.dot(vec / np.linalg.norm(vec), direction_from_center)
            if dot > 0.2:
                outer_points.append(pt)

    if len(outer_points) < 10:
        outer_points = []
        for pt in contour:
            vec = pt - main_centroid_xy
            if np.linalg.norm(vec) > 0:
                dot = np.dot(vec / np.linalg.norm(vec), -direction_from_center)
                if dot > 0.2:
                    outer_points.append(pt)

    return np.array(outer_points) if outer_points else contour


def measure_main_stem(
    main_mask: np.ndarray,
    root_hair_masks: List[np.ndarray],
    img_height: int,
    img_width: int,
) -> Tuple[float, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Measure main stem with a best-fit line extended to image edges."""
    main_coords = np.column_stack(np.where(main_mask))
    if len(main_coords) == 0:
        return 0.0, None
    main_centroid = np.mean(main_coords, axis=0)
    main_centroid_xy = np.array([main_centroid[1], main_centroid[0]])

    contours, _ = cv2.findContours((main_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return 0.0, None
    contour = max(contours, key=cv2.contourArea)[:, 0, :]

    has_root_hairs = len(root_hair_masks) > 0 and any(np.sum(m) > 0 for m in root_hair_masks)

    if has_root_hairs:
        rh_centroids = []
        for rh_mask in root_hair_masks:
            coords = np.column_stack(np.where(rh_mask))
            if len(coords) > 0:
                rh_centroids.append(np.mean(coords, axis=0))
        avg_rh_position = np.mean(rh_centroids, axis=0)
        direction_to_rh = avg_rh_position - main_centroid
        norm = np.linalg.norm(direction_to_rh)
        if norm == 0:
            side_points = contour
        else:
            direction_to_rh = direction_to_rh / norm
            direction_xy = np.array([direction_to_rh[1], direction_to_rh[0]])
            side_points = []
            for pt in contour:
                vec = pt - main_centroid_xy
                if np.linalg.norm(vec) > 0:
                    dot = np.dot(vec / np.linalg.norm(vec), direction_xy)
                    if dot > 0.2:
                        side_points.append(pt)
            side_points = np.array(side_points) if side_points else contour
    else:
        side_points = get_outer_contour_side(contour, main_centroid_xy, img_height, img_width)

    if len(side_points) < 2:
        side_points = contour

    mean_pt = np.mean(side_points, axis=0)
    centered = side_points - mean_pt
    cov = np.cov(centered, rowvar=False)
    if cov.ndim < 2:
        cov = np.array([[cov, 0], [0, cov]])

    eigenvalues, eigenvectors = np.linalg.eig(cov)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)].real
    norm = np.linalg.norm(principal_axis)
    if norm == 0:
        return 0.0, None
    principal_axis = principal_axis / norm

    pt1, pt2 = extend_line_to_image_edges(mean_pt, principal_axis, img_width, img_height)
    length_px = float(np.linalg.norm(pt2 - pt1))
    return length_px, (pt1, pt2)


def _load_metadata(metadata_path: Optional[Path]) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    if metadata_path is None or not metadata_path.exists():
        return {}
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    out = {}
    for item in data:
        color = tuple(int(v) for v in item.get("color_rgb", []))
        if len(color) == 3:
            out[color] = item
    return out


def _measure_instance_geometry(
    component_mask: np.ndarray,
    config: MeasurementConfig,
) -> Tuple[float, float, List[np.ndarray], List[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
    skel = skeletonize(component_mask.astype(bool)).astype(np.uint8)
    skel_pruned = prune_skeleton(skel, prune_threshold=config.skeleton_prune_threshold_px)
    labeled_skel = label(skel_pruned, connectivity=2)
    regions_skel = regionprops(labeled_skel)
    if len(regions_skel) == 0:
        return 0.0, 0.0, [], []

    main_region = max(regions_skel, key=lambda r: r.area)
    skel_component = (labeled_skel == main_region.label).astype(np.uint8)
    primary_path = extract_primary_skeleton(skel_component)
    if primary_path.size == 0:
        return 0.0, 0.0, [], []

    region_length = 0.0
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

        width, pos_b, neg_b = measure_width_at_point(
            pt,
            tangent,
            component_mask,
            max_distance=config.width_max_distance_px,
        )
        widths_region.append(width)
        boundaries_region.append((pt, pos_b, neg_b))

        if i > 0:
            region_length += float(np.linalg.norm(primary_path[i] - primary_path[i - 1]))

    avg_width = float(np.mean(widths_region)) if widths_region else 0.0
    return region_length, avg_width, [primary_path], [boundaries_region]


def process_mask(
    mask_path: str | Path,
    metadata_path: str | Path | None = None,
    annotated_path: str | Path | None = None,
    config: MeasurementConfig | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Process one color-coded instance mask and return summary plus per-instance rows."""
    config = config or MeasurementConfig()
    mask_path = Path(mask_path)
    metadata = _load_metadata(Path(metadata_path) if metadata_path else None)

    img_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    annotated_img = img_rgb.copy()
    height, width_img = img_rgb.shape[:2]

    background_color = np.array([0, 0, 0])
    pixels = img_rgb.reshape((-1, 3))
    unique_colors = np.unique(pixels, axis=0)

    aggregated_results: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    for color in unique_colors:
        if np.all(color == background_color):
            continue
        color_tuple = tuple(int(v) for v in color)
        mask_color = np.all(img_rgb == color, axis=2).astype(np.uint8)
        labeled_mask = label(mask_color, connectivity=2)
        regions = regionprops(labeled_mask)

        total_length = 0.0
        all_widths = []
        all_skeletons = []
        all_boundaries = []
        total_area = int(np.sum(mask_color))

        for region in regions:
            component_mask = (labeled_mask == region.label).astype(np.uint8)
            length_px, avg_width_px, skeletons, boundaries = _measure_instance_geometry(component_mask, config)
            total_length += length_px
            if avg_width_px > 0:
                all_widths.append(avg_width_px)
            all_skeletons.extend(skeletons)
            all_boundaries.extend(boundaries)

        avg_width = float(np.mean(all_widths)) if len(all_widths) > 0 else 0.0
        meta = metadata.get(color_tuple, {})
        class_name = meta.get("class_name", "unknown")

        aggregated_results[color_tuple] = {
            "mask": mask_color,
            "skeleton_length": total_length,
            "avg_width": avg_width,
            "skeletons": all_skeletons,
            "boundaries": all_boundaries,
            "color": np.array(color_tuple),
            "area": total_area,
            "class_name": class_name,
            "metadata": meta,
        }

    if len(aggregated_results) == 0:
        raise ValueError(f"No non-background instances found in mask: {mask_path}")

    # Prefer metadata class MS. If absent, use largest instance as main stem.
    ms_candidates = [k for k, v in aggregated_results.items() if str(v.get("class_name", "")).upper() == "MS"]
    if ms_candidates:
        main_stem_key = max(ms_candidates, key=lambda k: aggregated_results[k]["area"])
    else:
        main_stem_key = max(aggregated_results.keys(), key=lambda k: aggregated_results[k]["area"])

    root_hair_keys = [k for k in aggregated_results if k != main_stem_key]
    root_hair_masks = [aggregated_results[k]["mask"] for k in root_hair_keys]

    main_mask = aggregated_results[main_stem_key]["mask"]
    main_line_length, best_fit_line = measure_main_stem(main_mask, root_hair_masks, height, width_img)

    if main_line_length > 0:
        aggregated_results[main_stem_key]["total_length"] = main_line_length
        aggregated_results[main_stem_key]["best_fit_line"] = best_fit_line
    else:
        aggregated_results[main_stem_key]["total_length"] = aggregated_results[main_stem_key]["skeleton_length"]
        aggregated_results[main_stem_key]["best_fit_line"] = None
    aggregated_results[main_stem_key]["is_main_stem"] = True
    aggregated_results[main_stem_key]["avg_width"] = 0.0
    aggregated_results[main_stem_key]["class_name"] = "MS"

    for key in root_hair_keys:
        aggregated_results[key]["total_length"] = aggregated_results[key]["skeleton_length"]
        aggregated_results[key]["is_main_stem"] = False
        if aggregated_results[key].get("class_name", "unknown") == "unknown":
            aggregated_results[key]["class_name"] = "RH"

    # IDs
    non_main_counter = 1
    for key, res in aggregated_results.items():
        if res.get("is_main_stem", False):
            res["id"] = "Main Stem"
        else:
            res["id"] = str(non_main_counter)
            non_main_counter += 1

    main_stem_length_um = aggregated_results[main_stem_key]["total_length"] * config.conversion_factor
    rh_lengths = [aggregated_results[k]["total_length"] * config.conversion_factor for k in root_hair_keys]
    rh_widths = [aggregated_results[k]["avg_width"] * config.conversion_factor for k in root_hair_keys if aggregated_results[k]["avg_width"] > 0]
    num_root_hairs = len(root_hair_keys)
    density = num_root_hairs / main_stem_length_um if main_stem_length_um != 0 else 0.0

    # Annotate
    for res in aggregated_results.values():
        if res.get("is_main_stem", False):
            draw_color = (255, 0, 0)
            if res.get("best_fit_line") is not None:
                pt1, pt2 = res["best_fit_line"]
                cv2.line(
                    annotated_img,
                    (int(round(pt1[0])), int(round(pt1[1]))),
                    (int(round(pt2[0])), int(round(pt2[1]))),
                    draw_color,
                    3,
                )
                cv2.circle(annotated_img, (int(pt1[0]), int(pt1[1])), 8, draw_color, -1)
                cv2.circle(annotated_img, (int(pt2[0]), int(pt2[1])), 8, draw_color, -1)
        else:
            draw_color = (0, 255, 0)
            for skel_arr in res["skeletons"]:
                for pt in skel_arr:
                    r, c = int(pt[0]), int(pt[1])
                    cv2.circle(annotated_img, (c, r), 1, draw_color, -1)
            for boundary_set in res["boundaries"]:
                for i, (pt, pos_b, neg_b) in enumerate(boundary_set):
                    if i % 5 != 0:
                        continue
                    pos_int = (int(round(pos_b[1])), int(round(pos_b[0])))
                    neg_int = (int(round(neg_b[1])), int(round(neg_b[0])))
                    cv2.line(annotated_img, pos_int, neg_int, (0, 0, 255), 1)

        if res["skeletons"]:
            mid_pt = res["skeletons"][0][len(res["skeletons"][0]) // 2]
            cv2.putText(annotated_img, res["id"], (int(mid_pt[1]), int(mid_pt[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_img, res["id"], (int(mid_pt[1]), int(mid_pt[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if annotated_path is not None:
        save_image_rgb(annotated_path, annotated_img)

    summary = {
        "image": mask_path.name,
        "main_stem_length_um": float(main_stem_length_um),
        "root_hair_count": int(num_root_hairs),
        "root_hair_density_per_um": float(density),
        "avg_root_hair_length_um": float(np.mean(rh_lengths)) if rh_lengths else 0.0,
        "avg_root_hair_width_um": float(np.mean(rh_widths)) if rh_widths else 0.0,
        "conversion_factor_um_per_pixel": float(config.conversion_factor),
    }

    instance_rows = []
    for key, res in aggregated_results.items():
        length_um = res["total_length"] * config.conversion_factor
        width_um = res["avg_width"] * config.conversion_factor if not res.get("is_main_stem", False) else 0.0
        color = tuple(int(v) for v in key)
        instance_rows.append(
            {
                "image": mask_path.name,
                "instance_id": res["id"],
                "class_name": res["class_name"],
                "is_main_stem": bool(res.get("is_main_stem", False)),
                "length_um": float(length_um),
                "width_um": float(width_um),
                "area_px": int(res["area"]),
                "color_r": color[0],
                "color_g": color[1],
                "color_b": color[2],
            }
        )

    return summary, instance_rows


def measure_folder(
    mask_dir: str | Path,
    output_csv: str | Path,
    per_instance_csv: str | Path | None = None,
    metadata_dir: str | Path | None = None,
    annotated_dir: str | Path | None = None,
    config: MeasurementConfig | None = None,
) -> Dict[str, Any]:
    """Measure all `*_mask.png` files in a folder."""
    config = config or MeasurementConfig()
    mask_dir = Path(mask_dir)
    output_csv = Path(output_csv)
    per_instance_csv = Path(per_instance_csv) if per_instance_csv else None
    metadata_dir = Path(metadata_dir) if metadata_dir else None
    annotated_dir = Path(annotated_dir) if annotated_dir else None

    ensure_dirs(output_csv.parent)
    if per_instance_csv:
        ensure_dirs(per_instance_csv.parent)
    if annotated_dir:
        ensure_dirs(annotated_dir)

    mask_files = sorted(mask_dir.glob("*_mask.png"))
    if not mask_files:
        # fallback for any PNG masks
        mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No mask PNG files found in: {mask_dir}")

    summary_rows = []
    all_instance_rows = []
    failed = []

    for mask_path in tqdm(mask_files, desc="Measuring masks"):
        stem = mask_path.stem.replace("_mask", "")
        metadata_path = metadata_dir / f"{stem}_instances.json" if metadata_dir else None
        annotated_path = annotated_dir / f"{stem}_measurement.png" if annotated_dir else None
        try:
            summary, instance_rows = process_mask(
                mask_path=mask_path,
                metadata_path=metadata_path if metadata_path and metadata_path.exists() else None,
                annotated_path=annotated_path,
                config=config,
            )
            summary_rows.append(summary)
            all_instance_rows.extend(instance_rows)
        except Exception as exc:
            failed.append({"image": mask_path.name, "error": str(exc)})

    summary_fields = [
        "image",
        "main_stem_length_um",
        "root_hair_count",
        "root_hair_density_per_um",
        "avg_root_hair_length_um",
        "avg_root_hair_width_um",
        "conversion_factor_um_per_pixel",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    if per_instance_csv:
        instance_fields = [
            "image",
            "instance_id",
            "class_name",
            "is_main_stem",
            "length_um",
            "width_um",
            "area_px",
            "color_r",
            "color_g",
            "color_b",
        ]
        with per_instance_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=instance_fields)
            writer.writeheader()
            writer.writerows(all_instance_rows)

    return {
        "num_masks": len(mask_files),
        "num_success": len(summary_rows),
        "num_failed": len(failed),
        "failed": failed,
        "output_csv": str(output_csv),
        "per_instance_csv": str(per_instance_csv) if per_instance_csv else None,
    }
