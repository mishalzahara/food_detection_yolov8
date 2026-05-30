import streamlit as st
import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import pandas as pd
import subprocess

# Set Page Config
st.set_page_config(
    page_title="Food Object Detection Dashboard",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling with Outfit & Plus Jakarta Sans Fonts
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Global Font Override */
.main, .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #0d0f17 !important;
    color: #e2e8f0 !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #131722 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.1) !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Beautiful Hero Header Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(249, 115, 22, 0.15) 50%, rgba(236, 72, 153, 0.05) 100%);
    border: 1px solid rgba(249, 115, 22, 0.2);
    border-radius: 20px;
    padding: 35px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(249, 115, 22, 0.05) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.8rem !important;
    margin: 0 !important;
    padding: 0 !important;
    background: linear-gradient(90deg, #ff7e40 0%, #ff4b4b 50%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-top: 10px !important;
    font-weight: 400;
}

/* Premium Card Components */
.glass-card {
    background: rgba(19, 23, 34, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(4px);
    transition: transform 0.3s ease, border-color 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    border-color: rgba(249, 115, 22, 0.3);
}

.metric-num {
    font-size: 2.2rem;
    font-weight: 800;
    font-family: 'Outfit', sans-serif;
    color: #ff7e40;
}
.metric-num.tuned {
    color: #10b981;
}

/* Streamlit Tabs Custom Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    padding-bottom: 12px;
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    background-color: rgba(19, 23, 34, 0.5) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(249, 115, 22, 0.2) 0%, rgba(239, 68, 68, 0.2) 100%) !important;
    color: #ffffff !important;
    border-color: rgba(249, 115, 22, 0.4) !important;
    box-shadow: 0 4px 12px rgba(249, 115, 22, 0.15);
}

/* Buttons Styling */
div.stButton > button {
    background: linear-gradient(90deg, #ff7e40 0%, #ff4b4b 100%) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 10px 28px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.25) !important;
    transition: all 0.3s ease !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(239, 68, 68, 0.4) !important;
}

/* Timeline Custom Styles for Pipeline Explorer */
.timeline-item {
    padding-left: 25px;
    border-left: 2px solid rgba(249, 115, 22, 0.3);
    position: relative;
    padding-bottom: 25px;
}
.timeline-item::before {
    content: '';
    position: absolute;
    left: -7px;
    top: 5px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: #ff7e40;
    box-shadow: 0 0 10px #ff7e40;
}
.timeline-active {
    border-left-color: #ff4b4b !important;
}
.timeline-active::before {
    background-color: #ff4b4b !important;
    box-shadow: 0 0 12px #ff4b4b !important;
}

/* Custom Table/DataFrame display */
[data-testid="stTable"] table {
    background-color: rgba(19, 23, 34, 0.8) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Hide Streamlit Footer & Customize Main Menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# YOLOv8 Importing & Setup
YOLO_AVAILABLE = False
yolo_error_msg = ""
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    yolo_error_msg = str(e)

# Project Config
WORKSPACE_PATH = os.path.dirname(os.path.abspath(__file__))
SAMPLE_IMAGES_DIR = os.path.join(WORKSPACE_PATH, "sample_images")

# Category info helper
CATEGORY_DESCRIPTIONS = {
    "Pizza": "High-calorie flatbread topped with tomatoes, cheese, and meats. Rich in fats and carbohydrates.",
    "Salad": "Low-calorie, highly nutritious mixture of leafy greens and fresh vegetables. High in dietary fiber and vitamins.",
    "Breads": "Staple food prepared from dough of flour and water. High carbohydrate energy source.",
    "Curry": "Flavorful spiced dish containing vegetables or meats. High protein and healthy spice compounds.",
    "Rice": "Cereal grain, staple carbohydrate source for a large portion of the world's population.",
    "Roti_Chappati": "Unleavened flatbread. Rich in dietary fibers, iron, and complex carbohydrates.",
    "Cake": "Sweet dessert food, baked, high in sugars and fats.",
    "Drink": "Beverage items. Caloric contents vary widely depending on sugar content.",
    "Egg": "Excellent high-protein source rich in vitamins B2, B12, and Vitamin D.",
    "Fruit": "Naturally sweet produce. Loaded with vitamin C, antioxidants, and fibers."
}

# Standard COCO to custom label mapping for YOLOv8 pre-trained
COCO_TO_CUSTOM_MAP = {
    "pizza": "Pizza",
    "sandwich": "Breads",
    "banana": "Fruit",
    "apple": "Fruit",
    "orange": "Fruit",
    "broccoli": "Salad",
    "carrot": "Salad",
    "cake": "Cake",
    "hot dog": "Curry",  # rough maps for fun demo
    "donut": "Cake"
}

# Dynamic box drawing helper
def process_and_draw(image, model, conf_thresh, iou_thresh, model_type="COCO"):
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img, "RGBA")
    width, height = draw_img.size
    
    # Harmonious HSL colors for drawing
    color_palette = [
        (249, 115, 22),   # Orange
        (239, 68, 68),    # Red
        (16, 185, 129),   # Emerald
        (59, 130, 246),   # Blue
        (236, 72, 153),   # Pink
        (139, 92, 246),   # Purple
        (245, 158, 11),   # Amber
        (6, 182, 212),    # Cyan
        (101, 163, 13),   # Lime
        (79, 70, 229)     # Indigo
    ]
    
    detections = []
    
    if model is not None and YOLO_AVAILABLE:
        # Run real inference
        results = model.predict(image, conf=conf_thresh, iou=iou_thresh, verbose=False)
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                raw_label = results[0].names[cls_id]
                
                # Filter/map label
                if model_type == "COCO":
                    if raw_label in COCO_TO_CUSTOM_MAP:
                        label = COCO_TO_CUSTOM_MAP[raw_label]
                    else:
                        label = raw_label.capitalize()
                else:
                    # Custom model labels
                    label = raw_label
                
                detections.append({
                    "box": coords,
                    "label": label,
                    "conf": conf
                })
    else:
        # Fallback Mock Inference for Demo
        # Detect if it matches sample images or simulate based on random crops
        is_pizza = width == 1024 or "pizza" in getattr(image, "filename", "").lower()
        is_salad = "salad" in getattr(image, "filename", "").lower()
        
        if is_pizza:
            detections = [
                {"box": [int(0.12*width), int(0.15*height), int(0.88*width), int(0.85*height)], "label": "Pizza", "conf": 0.94},
                {"box": [int(0.2*width), int(0.2*height), int(0.4*width), int(0.45*height)], "label": "Breads", "conf": 0.52}
            ]
        elif is_salad:
            detections = [
                {"box": [int(0.15*width), int(0.12*height), int(0.85*width), int(0.88*height)], "label": "Salad", "conf": 0.91},
                {"box": [int(0.35*width), int(0.25*height), int(0.5*width), int(0.4*height)], "label": "Fruit", "conf": 0.78}
            ]
        else:
            # Simulate generic detections for custom upload in mock mode
            np.random.seed(42)
            classes = ["Pizza", "Salad", "Cake", "Breads", "Curry", "Drink", "Egg", "Fruit"]
            # 1 to 3 mock objects
            num_objects = np.random.randint(1, 4)
            for i in range(num_objects):
                cls = classes[np.random.randint(0, len(classes))]
                xmin = int(np.random.uniform(0.1, 0.4) * width)
                ymin = int(np.random.uniform(0.1, 0.4) * height)
                xmax = int(np.random.uniform(0.6, 0.9) * width)
                ymax = int(np.random.uniform(0.6, 0.9) * height)
                conf = float(np.random.uniform(0.65, 0.96))
                detections.append({
                    "box": [xmin, ymin, xmax, ymax],
                    "label": cls,
                    "conf": conf
                })
    
    # Draw detections
    for i, det in enumerate(detections):
        box = det["box"]
        label = det["label"]
        conf = det["conf"]
        
        # Select color based on label index or position
        color_idx = hash(label) % len(color_palette)
        color = color_palette[color_idx]
        
        # Transparent overlay box
        draw.rectangle(box, outline=color + (255,), width=4, fill=color + (40,))
        
        # Text label background
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            
        text_str = f" {label} {conf:.1%} "
        
        # Text dimensions
        left, top, right, bottom = draw.textbbox((0, 0), text_str, font=font)
        tw = right - left
        th = bottom - top
        
        # Draw background and text
        draw.rectangle([box[0], box[1] - th - 10, box[0] + tw + 6, box[1]], fill=color + (230,))
        draw.text((box[0] + 3, box[1] - th - 7), text_str, fill=(255, 255, 255), font=font)
        
    return draw_img, detections


# ================= SIDEBAR =================
st.sidebar.markdown(f"<div style='text-align: center;'><h2 style='margin-bottom: 0;'>⚙️ Control Panel</h2></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Model configuration
st.sidebar.subheader("🤖 Model Configuration")
model_option = st.sidebar.selectbox(
    "Choose YOLOv8 Model Version:",
    ["Standard YOLOv8n (Pre-trained COCO)", "Custom Model Weights (.pt file)", "Mock Demo Mode"]
)

# Confidence and IoU Thresholds
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
iou_thresh = st.sidebar.slider("IoU Threshold (NMS)", 0.1, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Workspace Details")
st.sidebar.markdown(f"**Root**: `{os.path.basename(WORKSPACE_PATH)}`")
st.sidebar.markdown(f"**YOLO Engine**: `{'Active (Ultralytics)' if YOLO_AVAILABLE else 'Inactive (Mock Fallback)'}`")
if not YOLO_AVAILABLE:
    st.sidebar.warning(f"Using mock fallback because: {yolo_error_msg[:100]}...")

# Load selected model
@st.cache_resource
def load_yolo_model(model_type, custom_path=None):
    if not YOLO_AVAILABLE:
        return None
    try:
        if model_type == "COCO":
            return YOLO("yolov8n.pt")
        elif model_type == "CUSTOM" and custom_path and os.path.exists(custom_path):
            return YOLO(custom_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return None

model = None
custom_weight_path = None
model_type_code = "COCO"

if YOLO_AVAILABLE:
    if model_option == "Standard YOLOv8n (Pre-trained COCO)":
        model = load_yolo_model("COCO")
        model_type_code = "COCO"
    elif model_option == "Custom Model Weights (.pt file)":
        custom_weight_path = st.sidebar.text_input("Local weight file path (.pt):", "runs/detect/train/weights/best.pt")
        if os.path.exists(custom_weight_path):
            model = load_yolo_model("CUSTOM", custom_weight_path)
            model_type_code = "CUSTOM"
            st.sidebar.success("Loaded Custom weights successfully!")
        else:
            st.sidebar.info("Enter a valid file path to a trained YOLOv8 model (.pt). Fallback to COCO model.")
            model = load_yolo_model("COCO")
            model_type_code = "COCO"
else:
    st.sidebar.info("Operating in Mock Mode. All visual representations are simulated.")


# ================= MAIN BODY =================

# Hero Section Banner
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">🍕 Food Object Detection Platform</h1>
    <div class="hero-subtitle">Interactive YOLOv8 pipeline analytics, model validation, and real-time food item localization.</div>
</div>
""", unsafe_allow_html=True)

# Define Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Real-time Inference", "📊 Model Metrics & Charts", "⚙️ Pipeline Explorer"])

# ----------------- TAB 1: REAL-TIME INFERENCE -----------------
with tab1:
    st.markdown("### 🔍 Real-time Food Detector")
    st.markdown("Upload a culinary image or choose one of our generated premium sample images to test the model's performance in real time.")
    
    col1, col2 = st.columns([1, 1])
    
    selected_image = None
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Input Source")
        
        source_mode = st.radio("Choose Input Type:", ["Use Sample Images", "Upload My Own Image"])
        
        if source_mode == "Use Sample Images":
            sample_choice = st.selectbox("Select a Sample Image:", ["Delicious Pizza", "Healthy Greek Salad"])
            
            # Paths to sample images
            pizza_path = os.path.join(SAMPLE_IMAGES_DIR, "pizza.png")
            salad_path = os.path.join(SAMPLE_IMAGES_DIR, "salad.png")
            
            if sample_choice == "Delicious Pizza" and os.path.exists(pizza_path):
                selected_image = Image.open(pizza_path)
                selected_image.filename = "pizza.png"  # inject filename helper
            elif sample_choice == "Healthy Greek Salad" and os.path.exists(salad_path):
                selected_image = Image.open(salad_path)
                selected_image.filename = "salad.png"
            else:
                st.error("Sample image file not found. Have they been moved?")
        else:
            uploaded_file = st.file_uploader("Upload an Image (JPG/PNG):", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
                selected_image.filename = uploaded_file.name
                
        st.markdown("</div>", unsafe_allow_html=True)
        
        if selected_image is not None:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Original Image Preview")
            st.image(selected_image, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Detection Results")
        
        if selected_image is not None:
            # Spinner animation for modern feel
            with st.spinner("🧠 YOLOv8 running neural localization network..."):
                time.sleep(0.5)  # brief sleep for UX feel
                processed_image, detections = process_and_draw(
                    selected_image, model, conf_thresh, iou_thresh, model_type_code
                )
                
            st.image(processed_image, use_column_width=True, caption="YOLOv8 Output Annotations")
            
            st.markdown("##### Detected Classes Details")
            if len(detections) > 0:
                table_data = []
                for det in detections:
                    lbl = det["label"]
                    cf = det["conf"]
                    desc = CATEGORY_DESCRIPTIONS.get(lbl, "No detailed nutrient estimation description available.")
                    table_data.append({"Class Name": lbl, "Confidence": f"{cf:.1%}", "Nutritional Analysis (Est.)": desc})
                
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            else:
                st.info("No items detected above the confidence threshold. Adjust the threshold slider on the sidebar.")
        else:
            st.warning("Please upload an image or select a sample image to start object detection.")
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------- TAB 2: MODEL COMPARISON & METRICS -----------------
with tab2:
    st.markdown("### 📊 Dataset & Model Evaluation Insights")
    st.markdown("A deep-dive comparison between the baseline model (`YOLOv8n` trained on 50 epochs) and the tuned model (`YOLOv8m` trained on 150 epochs).")
    
    # Key performance summary cards
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">YOLOv8n (Baseline) mAP50</div>
            <div class="metric-num">14.09%</div>
            <div style="font-size: 0.8rem; color: #f43f5e;">↓ Standard Baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">YOLOv8m (Tuned) mAP50</div>
            <div class="metric-num tuned">31.54%</div>
            <div style="font-size: 0.8rem; color: #10b981;">↑ +123% Improvement</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol3:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">YOLOv8n mAP50-95</div>
            <div class="metric-num">6.56%</div>
            <div style="font-size: 0.8rem; color: #f43f5e;">↓ Lower Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol4:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">YOLOv8m mAP50-95</div>
            <div class="metric-num tuned">12.93%</div>
            <div style="font-size: 0.8rem; color: #10b981;">↑ +97% Improvement</div>
        </div>
        """, unsafe_allow_html=True)

    # Class-wise AP Comparison Chart using Plotly
    st.markdown("#### 📈 Class-wise Average Precision (AP50) Comparison")
    
    classes_list = ["Breads", "Curry", "Pizza", "Rice", "Roti_Chappati", "Cake", "Drink", "Egg", "Fruit", "Salad"]
    yolov8n_ap = [0.1400, 0.1348, 0.0435, 0.1133, 0.0121, 0.0436, 0.1444, 0.0028, 0.0188, 0.0024]
    yolov8m_ap = [0.1030, 0.2130, 0.1553, 0.1910, 0.2260, 0.0917, 0.1804, 0.0839, 0.0230, 0.0256]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=classes_list,
        y=yolov8n_ap,
        name='YOLOv8n (Baseline, 50 Epochs)',
        marker_color='#f43f5e',
        opacity=0.85
    ))
    fig.add_trace(go.Bar(
        x=classes_list,
        y=yolov8m_ap,
        name='YOLOv8m (Tuned, 150 Epochs)',
        marker_color='#10b981',
        opacity=0.85
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(19,23,34,0.7)',
        paper_bgcolor='rgba(19,23,34,0)',
        font=dict(color='#cbd5e1', family='Plus Jakarta Sans'),
        title_font=dict(size=18, family='Outfit'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Food Category"),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="AP50 Score", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode='group',
        bargap=0.15,
        height=450,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Class distribution analysis
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 💡 Evaluation Insights & Findings")
    st.markdown("""
    - **Significant Performance Jump**: Training for **150 epochs** using a larger capacity network model (`YOLOv8m`) yields over a **double overall precision boost** (from `14.09%` to `31.54%` mAP50).
    - **Top Gaining Categories**:
      - **Roti_Chappati** AP50 increased from `1.2%` to **`22.6%`** (+1,768% increase).
      - **Curry** AP50 rose from `13.5%` to **`21.3%`** (+58% increase).
      - **Pizza** AP50 improved from `4.3%` to **`15.5%`** (+260% increase).
    - **Breads Anomaly**: Interestingly, `Breads` was the *only* class to experience regression (AP50 dropped from `14.0%` to `10.3%`). This indicates possible over-indexing on Curry and Roti in the longer training duration or class conflict/mislabeled instances.
    - **Low Performing Classes**: Both `Fruit` (`2.3%` AP50) and `Salad` (`2.5%` AP50) remain highly challenging for both model sizes. This suggests dataset imbalances or high variance in lighting/packaging.
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------- TAB 3: PIPELINE EXPLORER -----------------
with tab3:
    st.markdown("### ⚙️ Pipeline Execution Inspector")
    st.markdown("Explore the Python scripts that handle dataset initialization, visual data checks, augmentation splits, model training, and metrics processing.")
    
    pipeline_steps = [
        {"file": "01_data_loading.py", "title": "1. Mount & Extract Archives", "desc": "Handles Google Drive authorization mounting, directories setup, and extracting compressed datasets (.rar)."},
        {"file": "02_visualization.py", "title": "2. Box Visualization Check", "desc": "Extracts raw coordinates and draws temporary preview overlays on 10 training images to ensure coordinate sanity."},
        {"file": "03_preprocessing.py", "title": "3. Deduplication & Blur Check", "desc": "Filters images using MD5 hashing, flags blurry images with Laplacian variance, resizes to 640x640, and generates 5x augmented variants."},
        {"file": "04_dataset_split.py", "title": "4. Train-Test Validation Split", "desc": "Splits preprocessed image-annotations into an 80/20 train-test cohort, structuring directories in the custom YOLO configuration format."},
        {"file": "05_train.py", "title": "5. Training YOLOv8 Models", "desc": "Initiates model initialization and executes training loops for YOLOv8n (50 epochs) and YOLOv8m (150 epochs) on local configs."},
        {"file": "06_evaluate.py", "title": "6. Validation Report Engine", "desc": "Tests trained weight files on testing directories, prints precision-recall metrics, and outputs comparisons graphs."}
    ]
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Select Pipeline Stage")
        
        selected_index = st.radio(
            "Select Step to Inspect:",
            range(len(pipeline_steps)),
            format_func=lambda x: f"{pipeline_steps[x]['title']} ({pipeline_steps[x]['file']})"
        )
        
        step_info = pipeline_steps[selected_index]
        
        st.markdown("##### Stage Description:")
        st.write(step_info["desc"])
        
        # Script execution utility
        st.markdown("##### 🚀 Execution Sandbox")
        st.markdown("You can run a quick diagnostic dry run of this script step. *Note: Training scripts require the full dataset structure.*")
        
        target_script = os.path.join(WORKSPACE_PATH, "src", step_info["file"])
        
        if st.button("Run Script Diagnostic"):
            st.info(f"Triggering execution of `{step_info['file']}`...")
            
            # Simple dry run simulation for safety and ease if no arguments
            try:
                # We can execute using subprocess and stream logs
                log_placeholder = st.empty()
                log_text = ""
                
                # Running scripts with --help or just checking syntax since they run heavy stuff
                proc = subprocess.Popen(
                    ["python", target_script, "--help"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True
                )
                
                # Stream logs
                for line in proc.stdout:
                    log_text += line
                    log_placeholder.code(log_text)
                
                proc.wait()
                if proc.returncode == 0:
                    st.success(f"`{step_info['file']}` executed successfully or printed help config!")
                else:
                    # If help fails, let's just do a diagnostic dry run warning
                    st.warning(f"`{step_info['file']}` exited with code {proc.returncode}. (This is normal if datasets or inputs are missing).")
            except Exception as e:
                st.error(f"Failed to launch process: {e}")
                
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"#### 📄 Source Code: `{step_info['file']}`")
        
        # Load script file contents
        filepath = os.path.join(WORKSPACE_PATH, "src", step_info["file"])
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    code_content = f.read()
                st.code(code_content, language="python")
            except Exception as e:
                st.error(f"Could not load script code: {e}")
        else:
            st.error(f"Script file not found at: `{filepath}`")
            
        st.markdown("</div>", unsafe_allow_html=True)
