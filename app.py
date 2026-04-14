"""
app.py  –  Streamlit UI for Thyroid Nodule Detection
Run with:  streamlit run app.py
"""

import streamlit as st
import cv2
import tempfile
import os

from pipeline import (
    load_rcnn, load_densenet_ensemble, load_resnet_ensemble,
    run_pipeline_with_contour,
    RCNN_SCORE_THRESH,
    RCNN_WEIGHTS_PATH, DENSENET_WEIGHTS_PATHS, RESNET_WEIGHTS_PATHS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thyroid Nodule Analyser",
    page_icon="🔬",
    layout="wide",
)

st.title("Thyroid Nodule Analyser")
st.caption("Upload a thyroid ultrasound image — the pipeline will detect, segment, and classify nodules.")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Detection settings")
    min_detection_thresh = st.slider(
        "Minimum detection threshold",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
        help=(
            "If no nodules are found at the default threshold (0.40), the pipeline "
            "automatically lowers it in 0.05 steps down to this value. "
            "Lower = more sensitive but may produce more false positives."
        )
    )

# ── Load models once (cached so they don't reload on every interaction) ───────
@st.cache_resource(show_spinner="Loading models — this takes a moment the first time...")
def load_models():
    rcnn = load_rcnn(RCNN_WEIGHTS_PATH)
    densenet = load_densenet_ensemble(DENSENET_WEIGHTS_PATHS)
    resnet = load_resnet_ensemble(RESNET_WEIGHTS_PATHS)
    return rcnn, densenet, resnet

try:
    rcnn_model, densenet_models, resnet_models = load_models()
    st.success("Models loaded successfully")
except Exception as e:
    st.error(f"Could not load models: {e}\n\nCheck the weight paths in pipeline.py")
    st.stop()

st.divider()

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload ultrasound image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Supported formats: JPG, PNG, BMP"
)

if uploaded is None:
    st.info("👆 Upload an image above to get started.")
    st.stop()

# ── Save upload to a temp file (pipeline needs a file path) ──────────────────
suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

# ── Progress UI ───────────────────────────────────────────────────────────────
status_box = st.empty()
progress_bar = st.progress(0, text="Processing...")

def update_progress(current_step, total_steps, message):
    pct = int((current_step / total_steps) * 100)
    progress_bar.progress(pct, text=message)
    status_box.info(f"Step {current_step}/{total_steps}: {message}")

# ── Run pipeline ──────────────────────────────────────────────────────────────
try:
    results = run_pipeline_with_contour(
        image_path=tmp_path,
        rcnn_model=rcnn_model,
        densenet_models=densenet_models,
        resnet_models=resnet_models,
        min_detection_thresh=min_detection_thresh,
        progress_callback=update_progress,
    )
    progress_bar.progress(100, text="Processing complete")
    status_box.success("Processing finished successfully")
except Exception as e:
    st.error(f"Pipeline error: {e}")
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
    st.stop()

if os.path.exists(tmp_path):
    os.unlink(tmp_path)

# ── Adaptive threshold warning ────────────────────────────────────────────────
used_thresh = results.get("detection_thresh_used", RCNN_SCORE_THRESH)
if used_thresh < RCNN_SCORE_THRESH:
    st.warning(
        f"⚠️ No nodules found at the default threshold (0.40). "
        f"Threshold was automatically reduced to **{used_thresh:.2f}** to find detections. "
        f"Results may include lower-confidence regions — review carefully."
    )

nodules = results["nodules"]

# ── No nodules case ───────────────────────────────────────────────────────────
if not nodules:
    st.warning("No nodules were detected above the confidence threshold.")
    st.image(
        cv2.cvtColor(results["original_bgr"], cv2.COLOR_BGR2RGB),
        caption="Input image (CLAHE enhanced)",
        use_container_width=True,
    )
    st.stop()

# ── Per-nodule results ────────────────────────────────────────────────────────
for i, nod in enumerate(nodules):
    label = nod["final_label"]
    prob = nod["mal_prob"]
    conf = prob if label == "malignant" else 1 - prob
    pct = prob * 100

    st.subheader(f"Nodule {i + 1}")

    # Classification badge + metrics
    col_a, col_b, col_c = st.columns(3)

    if label == "malignant":
        col_a.error(f"🔴 MALIGNANT  ({conf:.0%} confidence)")
    else:
        col_a.success(f"🟢 BENIGN  ({conf:.0%} confidence)")

    col_b.metric("Malignancy probability", f"{pct:.1f}%")
    col_c.metric("Fusion note", nod["note"], help="How the CNN and RCNN predictions were combined")

    # Per-nodule probability bar
    st.progress(int(round(pct)), text=f"Malignancy probability: {pct:.1f}%")

    # Four image panels
    c1, c2, c3, c4 = st.columns(4)

    # Panel 1 – annotated full image
    annotated_rgb = cv2.cvtColor(results["annotated_bgr"], cv2.COLOR_BGR2RGB)
    c1.image(annotated_rgb, caption="Bounding box + snake contour", use_container_width=True)

    # Panel 2 – CLAHE-enhanced full image
    original_rgb = cv2.cvtColor(results["original_bgr"], cv2.COLOR_BGR2RGB)
    c2.image(original_rgb, caption="CLAHE enhanced input", use_container_width=True)

    # Panel 3 – Canny edges (ROI)
    edges = nod.get("edges")
    if edges is not None:
        c3.image(edges, caption="Canny edges (ROI)", use_container_width=True, clamp=True)
    else:
        c3.write("Canny edges not available")

    # Panel 4 – Enhanced grayscale ROI
    enhanced = nod.get("enhanced")
    if enhanced is not None:
        c4.image(enhanced, caption="Enhanced ROI (grayscale)", use_container_width=True, clamp=True)
    else:
        c4.write("Enhanced ROI not available")

    st.divider()

# ── Download annotated image ──────────────────────────────────────────────────
annotated_bgr = results["annotated_bgr"]
success, buf = cv2.imencode(".jpg", annotated_bgr)
if success:
    st.download_button(
        label="Download annotated image",
        data=buf.tobytes(),
        file_name="thyroid_result.jpg",
        mime="image/jpeg",
    )
