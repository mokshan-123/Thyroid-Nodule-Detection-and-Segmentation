"""
pipeline.py  –  All model + processing code from the notebook.
Imported by app.py (Streamlit UI).
"""

import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn

from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as TF
from torchvision.ops import box_iou

# ──────────────────────────────────────────────────────────────────────────────
# ① UPDATE THESE PATHS to point to your .pth weight files
# ──────────────────────────────────────────────────────────────────────────────
RCNN_WEIGHTS_PATH = "E:/Github/Thyroid nodule detection/RCNN Model data/Best_wei/best_fasterrcnn.pth"
DENSENET_WEIGHTS_PATHS = [
    "E:/Github/Thyroid nodule detection/CNN Model data/Best models from 3Folds/best_densenet_fold1.pth",
    "E:/Github/Thyroid nodule detection/CNN Model data/Best models from 3Folds/best_densenet_fold2.pth",
    "E:/Github/Thyroid nodule detection/CNN Model data/Best models from 3Folds/best_densenet_fold3.pth",
]
RESNET_WEIGHTS_PATHS = [
    "E:/Github/Thyroid nodule detection/Resnet Model/Best Models/best_resnet50_fold1.pth",
    "E:/Github/Thyroid nodule detection/Resnet Model/Best Models/best_resnet50_fold2.pth",
    "E:/Github/Thyroid nodule detection/Resnet Model/Best Models/best_resnet50_fold3.pth",
]

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLAHE_CLIP       = 2.0
CLAHE_GRID       = (8, 8)
CROP_MARGIN      = 30
RESIZE_SIZE      = 224
CLASS_NAMES      = {0: "benign", 1: "malignant"}
RCNN_LABEL_NAMES = {1: "benign", 2: "malignant"}
RCNN_SCORE_THRESH  = 0.4
NMS_IOU_THRESH     = 0.3
MALIGNANT_THRESH   = 0.32
BOX_COLORS         = {"benign": (0, 220, 0), "malignant": (220, 0, 0)}
CONTOUR_COLORS     = {"benign": (0, 255, 180), "malignant": (255, 80, 0)}

# ── Preprocessing ─────────────────────────────────────────────────────────────
def apply_clahe(img_bgr):
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l_clahe = clahe.apply(l)
    img_clahe = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)
    blurred   = cv2.GaussianBlur(img_clahe, (0, 0), sigmaX=2)
    return cv2.addWeighted(img_clahe, 1.3, blurred, -0.3, 0)

def load_and_preprocess(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    img_bgr = apply_clahe(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, Image.fromarray(img_rgb)

# ── Model builders ────────────────────────────────────────────────────────────
def build_rcnn(num_classes=3):
    return fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=num_classes)

def build_densenet():
    model = models.densenet121(weights=None)
    nf = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(nf, 2))
    return model

def build_resnet50():
    model = models.resnet50(weights=None)
    nf = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(nf, 512), nn.ReLU(inplace=True),
        nn.Dropout(p=0.3), nn.Linear(512, 2)
    )
    return model

# ── Model loaders ─────────────────────────────────────────────────────────────
def load_rcnn(weights_path):
    model = build_rcnn()
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_densenet_ensemble(paths):
    ensemble = []
    for p in paths:
        m = build_densenet()
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.to(DEVICE).eval()
        ensemble.append(m)
    return ensemble

def load_resnet_ensemble(paths):
    ensemble = []
    for p in paths:
        m = build_resnet50()
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.to(DEVICE).eval()
        ensemble.append(m)
    return ensemble

# ── Transforms ────────────────────────────────────────────────────────────────
val_transform = T.Compose([
    T.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
tta_transform = T.Compose([
    T.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    T.FiveCrop(int(RESIZE_SIZE * 0.9)),
    T.Lambda(lambda crops: torch.stack([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(T.ToTensor()(c))
        for c in crops
    ]))
])

# ── Detection (with adaptive thresholding) ────────────────────────────────────
def detect_nodules(rcnn_model, img_pil, min_thresh=0.20):
    tensor = TF.to_tensor(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        preds = rcnn_model(tensor)[0]

    boxes  = preds["boxes"].cpu()
    labels = preds["labels"].cpu()
    scores = preds["scores"].cpu()

    thresh = RCNN_SCORE_THRESH  # Start at 0.4

    while thresh >= min_thresh:
        keep = scores >= thresh
        if keep.any():
            return boxes[keep], labels[keep], scores[keep], thresh
        thresh = round(thresh - 0.05, 2)

    # Nothing found even at min_thresh — return whatever exists there
    keep = scores >= min_thresh
    return boxes[keep], labels[keep], scores[keep], min_thresh

# ── NMS ───────────────────────────────────────────────────────────────────────
def apply_nodule_nms(boxes, labels, scores):
    if len(boxes) == 0:
        return []
    order  = scores.argsort(descending=True)
    boxes, labels, scores = boxes[order], labels[order], scores[order]
    kept   = []
    active = [True] * len(boxes)
    for i in range(len(boxes)):
        if not active[i]:
            continue
        kept.append((boxes[i], labels[i], scores[i]))
        for j in range(i + 1, len(boxes)):
            if active[j] and box_iou(boxes[i].unsqueeze(0), boxes[j].unsqueeze(0)).item() >= NMS_IOU_THRESH:
                active[j] = False
    return kept

def crop_nodule(img_pil, box):
    W, H = img_pil.size
    x1, y1, x2, y2 = map(int, box.tolist())
    x1 = max(0, x1 - CROP_MARGIN);  y1 = max(0, y1 - CROP_MARGIN)
    x2 = min(W, x2 + CROP_MARGIN);  y2 = min(H, y2 + CROP_MARGIN)
    return img_pil.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)

def classify_crop(crop_pil, densenet_models, resnet_models):
    crop_rgb   = crop_pil.convert("RGB")
    probs_list = []
    tensor_dn  = val_transform(crop_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        for m in densenet_models:
            probs_list.append(torch.softmax(m(tensor_dn), dim=1)[0].cpu().numpy())
    tensor_rn = tta_transform(crop_rgb).to(DEVICE)
    with torch.no_grad():
        for m in resnet_models:
            probs_list.append(torch.softmax(m(tensor_rn), dim=1).mean(dim=0).cpu().numpy())
    avg_probs = np.mean(probs_list, axis=0)
    mal_prob  = float(avg_probs[1])
    return ("malignant" if mal_prob >= MALIGNANT_THRESH else "benign"), mal_prob, probs_list

def fuse_predictions(cnn_label, mal_prob, rcnn_label_int):
    rcnn_label = RCNN_LABEL_NAMES.get(rcnn_label_int, "unknown")
    if cnn_label == rcnn_label:
        return cnn_label, mal_prob, f"CNN+RCNN agree → {cnn_label}"
    if 0.35 <= mal_prob <= 0.50:
        return rcnn_label, mal_prob, f"Marginal CNN ({mal_prob:.2f}), RCNN overrides → {rcnn_label}"
    return cnn_label, mal_prob, f"CNN confident ({mal_prob:.2f}), overrides RCNN → {cnn_label}"

# ── Enhancement & edges ───────────────────────────────────────────────────────
def enhance_us_roi(roi_bgr):
    gray      = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe     = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    enhanced  = clahe.apply(bilateral)
    return cv2.medianBlur(enhanced, ksize=3)

def detect_edges_canny(enhanced_gray, low_thresh=30, high_thresh=80):
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), sigmaX=1.5)
    return cv2.Canny(blurred, low_thresh, high_thresh, apertureSize=3, L2gradient=True)

# ── Greedy snake ──────────────────────────────────────────────────────────────
def _image_energy(enhanced_gray):
    gx  = cv2.Sobel(enhanced_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy  = cv2.Sobel(enhanced_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag = cv2.GaussianBlur(mag, (21, 21), sigmaX=5)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag

def _internal_energy(contour):
    N     = len(contour)
    prev  = contour[np.arange(N) - 1]
    next_ = contour[(np.arange(N) + 1) % N]
    d_prev = contour - prev
    avg_dist = np.mean(np.linalg.norm(d_prev, axis=1))
    e_elastic   = (np.linalg.norm(d_prev, axis=1) - avg_dist) ** 2
    e_curvature = np.linalg.norm(next_ - 2 * contour + prev, axis=1) ** 2
    return e_elastic, e_curvature

def greedy_snake(enhanced_gray, init_contour, alpha=0.5, beta=0.2, gamma=2.0,
                 n_iters=300, search_radius=3, convergence_thresh=0.1):
    H, W       = enhanced_gray.shape
    ext_energy = _image_energy(enhanced_gray)
    contour    = init_contour.astype(np.float64).copy()
    r          = search_radius
    for _ in range(n_iters):
        e_elastic, e_curv = _internal_energy(contour)
        e_el_max = e_elastic.max() + 1e-9
        e_cu_max = e_curv.max()    + 1e-9
        total_move = 0.0
        for i in range(len(contour)):
            cx, cy   = contour[i]
            best_e   = np.inf
            best_pos = contour[i].copy()
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = int(round(cx + dx)), int(round(cy + dy))
                    if not (0 <= nx < W and 0 <= ny < H):
                        continue
                    old = contour[i].copy()
                    contour[i] = [nx, ny]
                    e_el_new, e_cu_new = _internal_energy(contour)
                    contour[i] = old
                    e_total = (alpha * e_el_new[i] / e_el_max
                             + beta  * e_cu_new[i] / e_cu_max
                             - gamma * ext_energy[ny, nx])
                    if e_total < best_e:
                        best_e   = e_total
                        best_pos = np.array([nx, ny], dtype=np.float64)
            total_move += np.linalg.norm(best_pos - contour[i])
            contour[i]  = best_pos
        if total_move / len(contour) < convergence_thresh:
            break
    return contour

def init_ellipse_contour(roi_w, roi_h, n_points=120, shrink=0.90):
    cx, cy  = roi_w / 2, roi_h / 2
    rx, ry  = (roi_w / 2) * shrink, (roi_h / 2) * shrink
    angles  = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)], axis=1)

def extract_nodule_contour(img_bgr, box_coords,
                            snake_alpha=0.5, snake_beta=0.2, snake_gamma=2.0,
                            snake_iters=300, snake_radius=3,
                            canny_low=30, canny_high=80, n_points=120):
    x1, y1, x2, y2 = box_coords
    roi_bgr = img_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        raise ValueError(f"Empty ROI for box {box_coords}")
    roi_h, roi_w = roi_bgr.shape[:2]
    enhanced     = enhance_us_roi(roi_bgr)
    edges        = detect_edges_canny(enhanced, canny_low, canny_high)
    init_contour = init_ellipse_contour(roi_w, roi_h, n_points)
    local_contour = greedy_snake(enhanced, init_contour,
                                  alpha=snake_alpha, beta=snake_beta, gamma=snake_gamma,
                                  n_iters=snake_iters, search_radius=snake_radius)
    contour_global = (local_contour + np.array([x1, y1])).astype(np.int32)
    return contour_global, edges, enhanced

# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_results_with_contour(img_bgr, nodule_results):
    out = img_bgr.copy()
    for res in nodule_results:
        x1, y1, x2, y2 = res["box"]
        label   = res["final_label"]
        prob    = res["mal_prob"]
        color   = BOX_COLORS.get(label, (200, 200, 0))
        c_color = CONTOUR_COLORS.get(label, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        contour = res.get("contour")
        if contour is not None and len(contour) > 2:
            cv2.polylines(out, [contour.reshape((-1, 1, 2))],
                          isClosed=True, color=c_color, thickness=2, lineType=cv2.LINE_AA)
        conf = prob if label == "malignant" else 1 - prob
        text = f"{label.upper()}  {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        ty = max(y1 - 8, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), (0, 0, 0), -1)
        cv2.putText(out, text, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return out

# ── Main pipeline (returns dict, no matplotlib) ───────────────────────────────
def run_pipeline_with_contour(
        image_path, rcnn_model, densenet_models, resnet_models,
        snake_alpha=0.5, snake_beta=0.2, snake_gamma=2.0,
        snake_iters=300, snake_radius=3,
        canny_low=30, canny_high=80, n_points=120,
        min_detection_thresh=0.20,
        progress_callback=None):
    """
    Returns a dict:
      {
        "nodules":               list of per-nodule dicts,
        "annotated_bgr":         full image with boxes + snake (BGR ndarray),
        "original_bgr":          CLAHE-enhanced original (BGR ndarray),
        "detection_thresh_used": the threshold that actually found boxes (float),
      }
    Each nodule dict has keys:
      box, final_label, mal_prob, note, contour, edges, enhanced
    """

    def report(step, total, message):
        if progress_callback is not None:
            progress_callback(step, total, message)

    total_base_steps = 4

    report(1, total_base_steps, "Loading and preprocessing image...")
    img_bgr, img_pil = load_and_preprocess(image_path)

    report(2, total_base_steps, "Detecting nodules with Faster R-CNN...")
    boxes, labels, scores, used_thresh = detect_nodules(
        rcnn_model, img_pil, min_thresh=min_detection_thresh
    )

    if len(boxes) == 0:
        report(4, total_base_steps, "No nodules detected.")
        return {
            "nodules": [],
            "annotated_bgr": img_bgr,
            "original_bgr": img_bgr,
            "detection_thresh_used": used_thresh,
        }

    report(3, total_base_steps, "Applying non-maximum suppression...")
    kept = apply_nodule_nms(boxes, labels, scores)

    n_kept = len(kept)
    total_steps = total_base_steps + (2 * n_kept) + 1
    current_step = 4

    nodule_results = []
    for idx, (box, rcnn_label_t, score_t) in enumerate(kept, start=1):
        report(current_step, total_steps, f"Classifying nodule {idx}/{n_kept}...")
        crop_pil, crop_coords = crop_nodule(img_pil, box)
        cnn_label, mal_prob, per_model = classify_crop(crop_pil, densenet_models, resnet_models)
        final_label, mal_prob, note = fuse_predictions(cnn_label, mal_prob, rcnn_label_t.item())
        current_step += 1

        report(current_step, total_steps, f"Extracting contour for nodule {idx}/{n_kept}...")
        contour_global = enhanced = edges = None
        try:
            contour_global, edges, enhanced = extract_nodule_contour(
                img_bgr, crop_coords,
                snake_alpha=snake_alpha, snake_beta=snake_beta,
                snake_gamma=snake_gamma, snake_iters=snake_iters,
                snake_radius=snake_radius,
                canny_low=canny_low, canny_high=canny_high, n_points=n_points
            )
        except Exception as e:
            print(f"Contour extraction failed for nodule {idx}: {e}")
        current_step += 1

        nodule_results.append({
            "box": crop_coords,
            "final_label": final_label,
            "mal_prob": mal_prob,
            "note": note,
            "contour": contour_global,
            "edges": edges,
            "enhanced": enhanced,
        })

    report(current_step, total_steps, "Drawing final annotated result...")
    annotated = draw_results_with_contour(img_bgr, nodule_results)

    report(total_steps, total_steps, "Finished.")

    return {
        "nodules": nodule_results,
        "annotated_bgr": annotated,
        "original_bgr": img_bgr,
        "detection_thresh_used": used_thresh,
    }
