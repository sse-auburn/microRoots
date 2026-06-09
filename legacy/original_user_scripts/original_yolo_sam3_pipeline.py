# ============================================================
# FINAL YOLOv12+CBAM and SAM3 Pipeline for Root Hair Segmentation
# ============================================================

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import json
import colorsys
from tqdm import tqdm
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION - WSL COMPATIBLE PATHS
# ============================================================

class Config:
    """Central configuration for all paths and parameters - WSL Compatible"""
    
    # Main output directory
    OUTPUT_DIR = Path("/mnt/c/wsl_projects/yolosam3ft4paper")
    
    # Dataset paths
    YOLO_DATASET_DIR = Path("/mnt/c/wsl_projects/sam3_testing/YoloDataset_RH_Enhanced")
    COCO_DATASET_PATH = Path("/mnt/c/wsl_projects/Coco Dataset Polygone")
    
    # Model paths
    YOLO_MODEL_PATH = "/mnt/c/wsl_projects/Yolo detection final model Cbam with mosaic/last.pt"
    SAM3_ROOT = "/mnt/c/wsl_projects/SAM3FT_tiles_based/sam3"
    SAM3_CHECKPOINT = "/mnt/c/wsl_projects/Sam3 FT Models/FinalFT.pt"
    
    # YOLO hyperparameters
    YOLO_CONF = 0.80
    YOLO_IOU = 0.7
    YOLO_IMGSZ = 1024
    YOLO_BATCH = 8
    
    # SAM3 parameters
    SAM3_CONF_THRESHOLD = 0.9
    
    # Classes
    CLASSES = ['RH', 'MS']
    CLASS_COLORS = {
        'RH': (255, 0, 0),
        'MS':  (0, 255, 0)
    }

# ============================================================
# DIRECTORY STRUCTURE SETUP
# ============================================================

def setup_directories(config):
    """Create organized output directory structure"""
    
    dirs = {
        'root':  config.OUTPUT_DIR,
        'yolo': config.OUTPUT_DIR / "01_YOLO_Detection",
        'yolo_vis': config.OUTPUT_DIR / "01_YOLO_Detection" / "visualizations",
        'yolo_charts': config. OUTPUT_DIR / "01_YOLO_Detection" / "charts",
        'sam3': config.OUTPUT_DIR / "02_SAM3_Segmentation",
        'sam3_masks': config.OUTPUT_DIR / "02_SAM3_Segmentation" / "instance_masks",
        'sam3_overlays': config.OUTPUT_DIR / "02_SAM3_Segmentation" / "overlay_visualizations",
        'sam3_charts': config.OUTPUT_DIR / "02_SAM3_Segmentation" / "charts",
        'final':  config.OUTPUT_DIR / "03_Final_Results",
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        
    return dirs

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_filename(filename):
    """Convert filename to safe ASCII characters"""
    # Replace problematic characters
    safe_name = filename.replace(" ", "_")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_name)
    return safe_name


def safe_imwrite(filepath, image):
    """Safely write image with error handling"""
    try:
        filepath_str = str(filepath)
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath_str), exist_ok=True)
        
        # Try cv2.imwrite first
        success = cv2.imwrite(filepath_str, image)
        
        if not success: 
            # Fallback:  use PIL
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                Image.fromarray(image_rgb).save(filepath_str)
            else:
                Image.fromarray(image).save(filepath_str)
        return True
    except Exception as e: 
        print(f"      Warning: Could not save {filepath}: {str(e)[:50]}")
        return False


def get_distinct_colors(n):
    """Generate n visually distinct colors using golden ratio"""
    colors = []
    golden_ratio = 0.618033988749895
    for i in range(n):
        hue = (i * golden_ratio) % 1.0
        saturation = 0.75 + (i % 3) * 0.08
        value = 0.85 - (i % 2) * 0.1
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def calculate_dice(pred_mask, gt_mask):
    """Calculate Dice Similarity Coefficient (DSC)"""
    if pred_mask is None or gt_mask is None:
        return 0.0
    
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(
            pred_mask. astype(np. uint8),
            (gt_mask.shape[1], gt_mask. shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    pred_flat = pred_mask.flatten().astype(bool)
    gt_flat = gt_mask.flatten().astype(bool)
    
    intersection = np.sum(pred_flat & gt_flat)
    total = np.sum(pred_flat) + np.sum(gt_flat)
    
    dice = (2.0 * intersection) / (total + 1e-7)
    return float(dice)


def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union (IoU)"""
    if pred_mask is None or gt_mask is None: 
        return 0.0
    
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(
            pred_mask.astype(np.uint8),
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    pred_flat = pred_mask.flatten().astype(bool)
    gt_flat = gt_mask.flatten().astype(bool)
    
    intersection = np.sum(pred_flat & gt_flat)
    union = np.sum(pred_flat | gt_flat)
    
    iou = intersection / (union + 1e-7)
    return float(iou)


def load_gt_mask_from_coco(img_name, dataset_path):
    """Load ground truth mask from COCO polygon annotations"""
    
    if "Sharmin" in img_name: 
        set_path = dataset_path / "Batch1" / "Set1 Sharmin"
    elif "Waseem" in img_name:
        set_path = dataset_path / "Batch1" / "Set1 Waseem"
    else: 
        return None
    
    annotation_file = set_path / "annotations" / "instances_default.json"
    
    if not annotation_file.exists():
        return None
    
    try:
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
    except: 
        return None
    
    search_name = img_name. replace("Set1_Sharmin_", "").replace("Set1_Waseem_", "")
    
    img_info = None
    for img in coco_data['images']:
        original_stem = Path(img['file_name']).stem.replace(' ', '_').replace('/', '_')
        if original_stem == search_name or search_name in original_stem or original_stem in search_name:
            img_info = img
            break
    
    if img_info is None:
        return None
    
    img_height = img_info['height']
    img_width = img_info['width']
    
    combined_mask = np. zeros((img_height, img_width), dtype=np.uint8)
    
    for ann in coco_data['annotations']:
        if ann['image_id'] != img_info['id']:
            continue
        
        if 'segmentation' in ann and ann['segmentation']: 
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(combined_mask, [pts], 1)
    
    return combined_mask

# ============================================================
# YOLO DETECTION MODULE
# ============================================================

class YOLODetector: 
    """YOLOv12 Detection Module"""
    
    def __init__(self, config):
        self.config = config
        self. model = None
        
    def load_model(self):
        """Load YOLO model"""
        from ultralytics import YOLO
        print("\n📦 Loading YOLOv12 model...")
        self.model = YOLO(self.config.YOLO_MODEL_PATH)
        print(f"   ✅ Model loaded | Classes: {self.model.names}")
        return self.model
    
    def validate(self, dirs):
        """Run official YOLO validation"""
        print("\n🔄 Running YOLO validation on test set...")
        print("-" * 50)
        
        yaml_path = self.config.YOLO_DATASET_DIR / "dataset.yaml"
        
        results = self.model.val(
            data=str(yaml_path),
            split='test',
            imgsz=self. config.YOLO_IMGSZ,
            conf=self.config. YOLO_CONF,
            iou=self.config.YOLO_IOU,
            batch=self.config. YOLO_BATCH,
            save_json=True,
            verbose=True
        )
        
        metrics = {
            'mAP50': float(results. box. map50),
            'mAP50-95': float(results. box.map),
            'Precision':  float(results.box.mp),
            'Recall': float(results.box.mr),
            'F1':  float(2 * results.box.mp * results. box.mr / (results.box.mp + results.box. mr + 1e-7)),
            'AP50_per_class': {
                name: float(results. box.ap50[i])
                for i, name in enumerate(results.names. values())
                if i < len(results.box. ap50)
            }
        }
        
        print(f"\n✅ YOLO Validation Complete!")
        print(f"   mAP50:     {metrics['mAP50']:.4f}")
        print(f"   mAP50-95:  {metrics['mAP50-95']:.4f}")
        print(f"   Precision: {metrics['Precision']:.4f}")
        print(f"   Recall:    {metrics['Recall']:.4f}")
        print(f"   F1-Score:  {metrics['F1']:.4f}")
        
        with open(dirs['yolo'] / "yolo_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def detect_all(self, dirs):
        """Run detection on all test images and save visualizations"""
        print("\n🎨 Running detection and creating visualizations...")
        
        test_images_dir = self.config.YOLO_DATASET_DIR / "images" / "test"
        test_images = sorted(test_images_dir.glob("*.jpg"))
        
        all_detections = {}
        
        for img_path in tqdm(test_images, desc="   Detecting"):
            img_name = img_path.stem
            safe_name = safe_filename(img_name)
            
            try:
                image = cv2.imread(str(img_path))
                if image is None: 
                    print(f"\n   Warning: Could not read {img_path}")
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_h, img_w = image. shape[:2]
                
                results = self.model(
                    image_rgb,
                    conf=self.config. YOLO_CONF,
                    iou=self. config.YOLO_IOU,
                    imgsz=self. config.YOLO_IMGSZ,
                    verbose=False
                )
                
                pred_boxes = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = [float(v) for v in box. xyxy[0]. cpu().numpy()]
                        cls_id = int(box.cls[0].cpu().numpy())
                        pred_boxes.append({
                            'x1': x1, 'y1':  y1, 'x2': x2, 'y2': y2,
                            'conf': float(box. conf[0].cpu().numpy()),
                            'class': cls_id,
                            'class_name': self. model.names[cls_id]
                        })
                
                all_detections[img_name] = {
                    'image_path': str(img_path),
                    'boxes': pred_boxes,
                    'img_size': (img_h, img_w)
                }
                
                # Create visualization
                vis_image = image. copy()
                for box in pred_boxes:
                    x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
                    color = self.config.CLASS_COLORS.get(box['class_name'], (255, 255, 0))
                    color_bgr = (color[2], color[1], color[0])
                    
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_bgr, 2)
                    
                    label = f"{box['class_name']}: {box['conf']:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis_image, (x1, y1 - th - 10), (x1 + tw + 4, y1), color_bgr, -1)
                    cv2.putText(vis_image, label, (x1 + 2, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Use safe filename for saving
                save_path = dirs['yolo_vis'] / f"{safe_name}_yolo.jpg"
                safe_imwrite(save_path, vis_image)
                
            except Exception as e: 
                print(f"\n   Warning:  Error processing {img_name}: {str(e)[:50]}")
                continue
        
        with open(dirs['yolo'] / "all_detections.json", 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        print(f"   ✅ Saved {len(all_detections)} visualizations")
        
        return all_detections
    
    def create_charts(self, metrics, all_detections, dirs):
        """Create YOLO performance charts"""
        print("\n📊 Creating YOLO charts...")
        
        fig, axes = plt. subplots(1, 3, figsize=(18, 5))
        
        # Chart 1: Per-class AP50
        classes = list(metrics['AP50_per_class'].keys())
        ap_values = list(metrics['AP50_per_class'].values())
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars1 = axes[0].bar(classes, ap_values, color=colors[: len(classes)], edgecolor='black', linewidth=1.5)
        axes[0].set_title('Per-Class AP50', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('AP50', fontsize=12)
        axes[0].set_ylim(0, 1)
        for bar, val in zip(bars1, ap_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')
        
        # Chart 2: Overall Metrics
        metric_names = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1']
        metric_values = [metrics['mAP50'], metrics['mAP50-95'],
                        metrics['Precision'], metrics['Recall'], metrics['F1']]
        colors2 = ['#3498DB', '#9B59B6', '#2ECC71', '#F39C12', '#E74C3C']
        
        bars2 = axes[1].bar(metric_names, metric_values, color=colors2, edgecolor='black', linewidth=1.5)
        axes[1].set_title('YOLOv12 Detection Metrics', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)
        for bar, val in zip(bars2, metric_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
        
        # Chart 3: Detection Statistics
        total = sum(len(d['boxes']) for d in all_detections.values())
        rh = sum(sum(1 for b in d['boxes'] if b['class_name'] == 'RH') for d in all_detections.values())
        ms = sum(sum(1 for b in d['boxes'] if b['class_name'] == 'MS') for d in all_detections.values())
        
        stats_names = ['Total', 'RH', 'MS']
        stats_values = [total, rh, ms]
        colors3 = ['#34495E', '#FF6B6B', '#4ECDC4']
        
        bars3 = axes[2].bar(stats_names, stats_values, color=colors3, edgecolor='black', linewidth=1.5)
        axes[2].set_title('Detection Statistics', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Count', fontsize=12)
        for bar, val in zip(bars3, stats_values):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val}', ha='center', fontsize=12, fontweight='bold')
        
        plt.suptitle('YOLOv12 Detection Results', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(str(dirs['yolo_charts'] / "yolo_results.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Charts saved")
        
        return {'total': total, 'RH': rh, 'MS': ms}

# ============================================================
# SAM3 SEGMENTATION MODULE
# ============================================================

class SAM3Segmentor: 
    """SAM3 Instance Segmentation Module"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load SAM3 model"""
        print("\n📦 Loading SAM3 model...")
        
        if self.config.SAM3_ROOT not in sys.path:
            sys.path.insert(0, self. config.SAM3_ROOT)
        
        # Enable TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends. cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        
        from sam3.model_builder import build_sam3_image_model
        from sam3.model. sam3_image_processor import Sam3Processor
        
        self.model = build_sam3_image_model(
            checkpoint_path=self.config.SAM3_CHECKPOINT,
            device="cuda",
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=True,
            compile=False
        )
        
        self.processor = Sam3Processor(
            self.model,
            confidence_threshold=self.config.SAM3_CONF_THRESHOLD
        )
        
        print(f"   ✅ SAM3 loaded")
        return self. model
    
    def segment_boxes(self, image_pil, boxes):
        """Run SAM3 on YOLO detection boxes"""
        if len(boxes) == 0:
            return []
        
        try:
            inference_state = self.processor. set_image(image_pil)
        except Exception as e: 
            print(f"      Warning: Could not set image: {str(e)[:30]}")
            return []
        
        instances = []
        colors = get_distinct_colors(len(boxes) + 10)
        
        for idx, box in enumerate(boxes):
            try:
                input_box = np.array([box['x1'], box['y1'], box['x2'], box['y2']])
                
                masks, scores, _ = self.model.predict_inst(
                    inference_state,
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                mask = masks[0]
                if mask.dtype != bool:
                    mask = mask > 0.5
                mask = mask.astype(np.uint8)
                
                instances.append({
                    'mask': mask,
                    'score': float(scores[0]),
                    'color': colors[idx],
                    'class':  box['class_name'],
                    'box': box
                })
                
            except Exception as e: 
                continue
        
        return instances
    
    def create_instance_mask(self, instances, img_h, img_w):
        """Create pure color instance mask"""
        mask_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        
        for inst in instances: 
            mask = inst['mask']
            if len(mask. shape) == 3:
                mask = mask.squeeze()
            if mask.shape != (img_h, img_w):
                mask = cv2.resize(mask. astype(np. uint8), (img_w, img_h),
                                 interpolation=cv2.INTER_NEAREST)
            
            color = inst['color']
            for c in range(3):
                mask_image[: , :, c] = np.where(mask > 0, color[c], mask_image[:, : , c])
        
        return mask_image
    
    def create_overlay(self, image, instances, img_h, img_w, alpha=0.5):
        """Create overlay visualization"""
        overlay = image.copy()
        
        for inst in instances:
            mask = inst['mask']
            if len(mask.shape) == 3:
                mask = mask. squeeze()
            if mask.shape != (img_h, img_w):
                mask = cv2.resize(mask.astype(np.uint8), (img_w, img_h),
                                 interpolation=cv2.INTER_NEAREST)
            
            color = inst['color']
            colored_mask = np.zeros_like(overlay)
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c]
            
            mask_bool = mask. astype(bool)
            overlay[mask_bool] = cv2.addWeighted(
                overlay, 1 - alpha, colored_mask, alpha, 0
            )[mask_bool]
        
        return overlay
    
    def process_all(self, all_detections, dirs):
        """Process all images with SAM3"""
        print("\n🔄 Running SAM3 segmentation on all images...")
        
        results = []
        all_dice_scores = []
        all_iou_scores = []
        error_count = 0
        
        for img_name, det_data in tqdm(all_detections.items(), desc="   Segmenting"):
            safe_name = safe_filename(img_name)
            
            try:
                image = cv2.imread(det_data['image_path'])
                if image is None: 
                    error_count += 1
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                img_h, img_w = det_data['img_size']
                
                instances = self.segment_boxes(image_pil, det_data['boxes'])
                
                # Create and save instance mask
                instance_mask = self. create_instance_mask(instances, img_h, img_w)
                mask_save_path = dirs['sam3_masks'] / f"{safe_name}_mask.png"
                safe_imwrite(mask_save_path, cv2.cvtColor(instance_mask, cv2.COLOR_RGB2BGR))
                
                # Create and save overlay
                overlay = self.create_overlay(image_rgb, instances, img_h, img_w)
                overlay_save_path = dirs['sam3_overlays'] / f"{safe_name}_overlay.jpg"
                safe_imwrite(overlay_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                
                # Calculate metrics
                combined_pred = np.any(instance_mask > 0, axis=2).astype(np. uint8)
                gt_mask = load_gt_mask_from_coco(img_name, self.config.COCO_DATASET_PATH)
                
                dice = 0.0
                iou = 0.0
                if gt_mask is not None:
                    dice = calculate_dice(combined_pred, gt_mask)
                    iou = calculate_iou(combined_pred, gt_mask)
                    all_dice_scores. append(dice)
                    all_iou_scores.append(iou)
                
                results.append({
                    'image':  img_name,
                    'num_instances': len(instances),
                    'dice': dice,
                    'iou':  iou
                })
                
            except Exception as e:
                error_count += 1
                # Continue processing other images
                continue
        
        print(f"\n   ✅ Processed {len(results)} images successfully")
        if error_count > 0:
            print(f"   ⚠️  {error_count} images had errors (skipped)")
        
        return results, all_dice_scores, all_iou_scores
    
    def create_charts(self, results, dice_scores, iou_scores, dirs):
        """Create SAM3 performance charts - FIXED VERSION"""
        print("\n📊 Creating SAM3 charts...")
        
        # Pre-calculate all values
        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        std_dice = float(np.std(dice_scores)) if dice_scores else 0.0
        mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
        std_iou = float(np.std(iou_scores)) if iou_scores else 0.0
        min_dice = float(np.min(dice_scores)) if dice_scores else 0.0
        max_dice = float(np.max(dice_scores)) if dice_scores else 0.0
        total_instances = sum(r['num_instances'] for r in results)
        num_images = len(results)
        avg_inst = total_instances / num_images if num_images > 0 else 0.0
        
        fig, axes = plt. subplots(2, 2, figsize=(14, 12))
        
        # Chart 1: Dice Score Distribution
        if dice_scores: 
            axes[0, 0].hist(dice_scores, bins=25, color='#3498DB', alpha=0.8, edgecolor='black')
            axes[0, 0].axvline(mean_dice, color='#E74C3C', linestyle='--', linewidth=2.5,
                              label=f'Mean: {mean_dice:.4f}')
        axes[0, 0].set_title('Dice Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Dice Score', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0]. grid(True, alpha=0.3)
        
        # Chart 2: IoU Distribution
        if iou_scores:
            axes[0, 1].hist(iou_scores, bins=25, color='#9B59B6', alpha=0.8, edgecolor='black')
            axes[0, 1].axvline(mean_iou, color='#E74C3C', linestyle='--', linewidth=2.5,
                              label=f'Mean:  {mean_iou:.4f}')
        axes[0, 1].set_title('IoU Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('IoU Score', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1]. grid(True, alpha=0.3)
        
        # Chart 3: Overall Metrics
        metrics_names = ['Dice', 'IoU']
        metrics_values = [mean_dice, mean_iou]
        metrics_stds = [std_dice, std_iou]
        colors = ['#2ECC71', '#3498DB']
        
        bars = axes[1, 0].bar(metrics_names, metrics_values, color=colors,
                              edgecolor='black', linewidth=2, width=0.5)
        axes[1, 0]. errorbar(metrics_names, metrics_values, yerr=metrics_stds,
                           fmt='none', color='black', capsize=10, capthick=2)
        axes[1, 0]. set_title('SAM3 Segmentation Performance', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Score', fontsize=12)
        axes[1, 0].set_ylim(0, 1)
        for bar, val, std in zip(bars, metrics_values, metrics_stds):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.03,
                           f'{val:.4f}+/-{std:.4f}', ha='center', fontsize=11, fontweight='bold')
        
        # Chart 4: Summary Box
        axes[1, 1].axis('off')
        
        lines = [
            "=" * 48,
            "         SAM3 SEGMENTATION SUMMARY",
            "=" * 48,
            "",
            f"   Dice Score:        {mean_dice:.4f} +/- {std_dice:.4f}",
            f"   IoU Score:         {mean_iou:.4f} +/- {std_iou:.4f}",
            "",
            f"   Min Dice:          {min_dice:.4f}",
            f"   Max Dice:         {max_dice:.4f}",
            "",
            f"   Total Images:     {num_images}",
            f"   Total Instances:  {total_instances}",
            f"   Avg Inst/Image:   {avg_inst:.1f}",
            "",
            "=" * 48,
        ]
        summary_text = "\n".join(lines)
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, fontfamily='monospace',
                       verticalalignment='center', transform=axes[1, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.suptitle('SAM3 Instance Segmentation Results', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(str(dirs['sam3_charts'] / "sam3_results.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics
        metrics = {
            'dice_mean': mean_dice,
            'dice_std': std_dice,
            'dice_min': min_dice,
            'dice_max': max_dice,
            'iou_mean': mean_iou,
            'iou_std': std_iou,
            'total_images': num_images,
            'total_instances': total_instances,
            'per_image_results': results
        }
        
        with open(dirs['sam3'] / "sam3_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   ✅ Charts saved")
        
        return metrics

# ============================================================
# FINAL RESULTS GENERATOR
# ============================================================

def create_final_results(yolo_metrics, yolo_stats, sam3_metrics, dirs):
    """Create final combined results"""
    print("\n📊 Creating final combined results...")
    
    fig, axes = plt. subplots(1, 2, figsize=(16, 6))
    
    # YOLO Results
    axes[0].bar(['mAP50'], [yolo_metrics['mAP50']],
               color='#3498DB', edgecolor='black', linewidth=2, width=0.4)
    axes[0].set_title('YOLOv12 Detection\nmAP50', fontsize=16, fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Score', fontsize=14)
    axes[0].text(0, yolo_metrics['mAP50'] + 0.03, f"{yolo_metrics['mAP50']:.4f}",
                ha='center', fontsize=18, fontweight='bold', color='#2C3E50')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # SAM3 Results
    dice_mean = sam3_metrics['dice_mean']
    dice_std = sam3_metrics['dice_std']
    
    axes[1]. bar(['Dice Score'], [dice_mean],
               color='#2ECC71', edgecolor='black', linewidth=2, width=0.4)
    axes[1].errorbar(['Dice Score'], [dice_mean],
                    yerr=[dice_std], fmt='none',
                    color='black', capsize=15, capthick=2)
    axes[1].set_title('SAM3 Segmentation\nDice Score', fontsize=16, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Score', fontsize=14)
    axes[1]. text(0, dice_mean + dice_std + 0.05,
                f"{dice_mean:.4f} +/- {dice_std:.4f}",
                ha='center', fontsize=16, fontweight='bold', color='#2C3E50')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('YOLOv12 + SAM3 Pipeline:  Final Results', fontsize=20, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(str(dirs['final'] / "FINAL_RESULTS.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Final summary JSON
    final_summary = {
        'pipeline':  'YOLOv12 + SAM3',
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'yolo_detection':  {
            'model':  'YOLOv12 with CBAM',
            'mAP50': yolo_metrics['mAP50'],
            'mAP50-95': yolo_metrics['mAP50-95'],
            'Precision': yolo_metrics['Precision'],
            'Recall':  yolo_metrics['Recall'],
            'F1': yolo_metrics['F1'],
            'AP50_per_class': yolo_metrics['AP50_per_class'],
            'total_detections': yolo_stats['total'],
            'RH_detections': yolo_stats['RH'],
            'MS_detections': yolo_stats['MS']
        },
        'sam3_segmentation': {
            'model': 'SAM3 Fine-tuned',
            'dice_mean': sam3_metrics['dice_mean'],
            'dice_std': sam3_metrics['dice_std'],
            'iou_mean': sam3_metrics['iou_mean'],
            'iou_std': sam3_metrics['iou_std'],
            'total_instances': sam3_metrics['total_instances'],
            'total_images': sam3_metrics['total_images']
        }
    }
    
    with open(dirs['final'] / "FINAL_SUMMARY.json", 'w') as f:
        json. dump(final_summary, f, indent=2)
    
    # Paper-ready table
    rh_ap = yolo_metrics['AP50_per_class']. get('RH', 0)
    ms_ap = yolo_metrics['AP50_per_class'].get('MS', 0)
    
    table_lines = [
        "=" * 80,
        "                    RESULTS FOR PAPER - YOLOv12 + SAM3 PIPELINE",
        "=" * 80,
        "",
        "Table 1: YOLOv12 Detection Performance",
        "-" * 80,
        "Metric          | Value",
        "----------------|" + "-" * 63,
        f"mAP@50          | {yolo_metrics['mAP50']:.4f}",
        f"mAP@50-95       | {yolo_metrics['mAP50-95']:.4f}",
        f"Precision       | {yolo_metrics['Precision']:.4f}",
        f"Recall          | {yolo_metrics['Recall']:.4f}",
        f"F1-Score        | {yolo_metrics['F1']:.4f}",
        "-" * 80,
        "Per-Class AP@50:",
        f"  - RH (Root Hair):       {rh_ap:.4f}",
        f"  - MS (Meristematic):    {ms_ap:.4f}",
        "-" * 80,
        "",
        "Table 2: SAM3 Segmentation Performance",
        "-" * 80,
        "Metric          | Value",
        "----------------|" + "-" * 63,
        f"Dice Score      | {sam3_metrics['dice_mean']:.4f} +/- {sam3_metrics['dice_std']:.4f}",
        f"IoU Score       | {sam3_metrics['iou_mean']:.4f} +/- {sam3_metrics['iou_std']:.4f}",
        "-" * 80,
        "",
        "Statistics:",
        f"  - Total Test Images:     {sam3_metrics['total_images']}",
        f"  - Total Detections:     {yolo_stats['total']}",
        f"  - Total Instances:      {sam3_metrics['total_instances']}",
        "=" * 80,
    ]
    
    table_text = "\n".join(table_lines)
    
    with open(dirs['final'] / "PAPER_RESULTS.txt", 'w') as f:
        f.write(table_text)
    
    print(table_text)
    print(f"\n   ✅ Final results saved to {dirs['final']}")

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """Run complete YOLOv12 + SAM3 pipeline"""
    
    print("=" * 70)
    print("   YOLOv12 + SAM3 PIPELINE FOR ROOT HAIR SEGMENTATION")
    print("   Final Version for Paper (WSL Compatible) - V2")
    print("=" * 70)
    
    config = Config()
    
    print("\n📁 Setting up output directories...")
    dirs = setup_directories(config)
    print(f"   Output:  {config.OUTPUT_DIR}")
    
    # ============================================================
    # PHASE 1: YOLO DETECTION
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: YOLOv12 DETECTION")
    print("=" * 70)
    
    yolo = YOLODetector(config)
    yolo.load_model()
    yolo_metrics = yolo. validate(dirs)
    all_detections = yolo.detect_all(dirs)
    yolo_stats = yolo.create_charts(yolo_metrics, all_detections, dirs)
    
    # ============================================================
    # PHASE 2: SAM3 SEGMENTATION
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: SAM3 SEGMENTATION")
    print("=" * 70)
    
    sam3 = SAM3Segmentor(config)
    sam3.load_model()
    sam3_results, dice_scores, iou_scores = sam3.process_all(all_detections, dirs)
    sam3_metrics = sam3.create_charts(sam3_results, dice_scores, iou_scores, dirs)
    
    # ============================================================
    # PHASE 3: FINAL RESULTS
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 3: GENERATING FINAL RESULTS")
    print("=" * 70)
    
    create_final_results(yolo_metrics, yolo_stats, sam3_metrics, dirs)
    
    # ============================================================
    # COMPLETE
    # ============================================================
    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 70)
    
    num_detections = len(all_detections)
    num_sam3 = len(sam3_results)
    
    print(f"""
   Output Structure:
   {config.OUTPUT_DIR}/
   |-- 01_YOLO_Detection/
   |   |-- visualizations/     ({num_detections} images)
   |   |-- charts/
   |   |-- yolo_metrics.json
   |   |-- all_detections.json
   |-- 02_SAM3_Segmentation/
   |   |-- instance_masks/     ({num_sam3} masks)
   |   |-- overlay_visualizations/
   |   |-- charts/
   |   |-- sam3_metrics.json
   |-- 03_Final_Results/
       |-- FINAL_RESULTS.png
       |-- FINAL_SUMMARY.json
       |-- PAPER_RESULTS.txt
    """)

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__": 
    main()
