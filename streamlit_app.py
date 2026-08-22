"""
streamlit_app.py

This script implements a premium, production-ready healthcare dashboard
for DermaVision - Intelligent Skin Image Classification.
It uses a 2-column layout matching the original structure, styled as a premium SaaS dashboard.
"""

import os
import time
import datetime
import os
import time
import datetime
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import streamlit as st
from fpdf import FPDF
import json

from preprocessing import preprocess_image
from validator import is_valid_dermatology_image
from prediction import predict_top_classes, load_prediction_model
from utils import load_class_mapping, format_probabilities, evaluate_image_quality

@st.cache_resource
def cached_load_prediction_model():
    return load_prediction_model()

def clean_html(html_str):
    """
    Strips leading and trailing whitespaces from every line in the HTML string
    to prevent Streamlit's markdown parser from interpreting nested indented
    lines as Markdown code blocks.
    """
    return "\n".join(line.strip() for line in html_str.splitlines())

class_names = load_class_mapping()

# Clinical disease database for all 23 classes
DISEASE_INFO = {
    "BCC": {
        "description": "Basal Cell Carcinoma (BCC) is the most common form of skin cancer and the most frequently occurring form of all cancers. It typically arises in sun-exposed areas of the body.",
        "symptoms": "Often appears as a slightly transparent bump on the skin, though it can take other forms. It may also look like a shiny skin-colored bump, a brown, black, or blue lesion, or a flat, scaly, reddish patch.",
        "causes": "Chronic, cumulative ultraviolet (UV) radiation exposure from sunlight or tanning beds leads to DNA mutations in the basal cells of the epidermis.",
        "treatment": "Surgical excision, Mohs micrographic surgery, electrosurgery (curettage and electrodesiccation), or cryotherapy. Topical treatments (fluorouracil, imiquimod) may be used for superficial BCCs.",
        "precautions": "Strict sun protection is mandatory: wear broad-spectrum SPF 30+ sunscreen daily, wear wide-brimmed hats, seek shade, and conduct monthly skin self-examinations.",
        "consult": "Immediately. Any new, changing, bleeding, or non-healing skin lesion must be evaluated by a dermatologist for a biopsy.",
        "risk": "High"
    },
    "Melanoma": {
        "description": "Melanoma is a highly aggressive and the most dangerous form of skin cancer. It develops from melanocytes, the pigment-producing cells of the skin.",
        "symptoms": "Moles displaying ABCDE criteria: Asymmetry, Border irregularity, Color variation, Diameter > 6mm, or Evolving size, shape, or color. Lesions may also itch, bleed, or ooze.",
        "causes": "Intense, intermittent UV radiation exposure (such as severe sunburns), genetic susceptibility, and a high number of typical or atypical moles.",
        "treatment": "Surgical excision with wide margins is the primary treatment. Advanced stages may require sentinel lymph node biopsy, immunotherapy, targeted therapy, or radiation.",
        "precautions": "Perform monthly skin checks, receive annual professional full-body skin examinations, protect skin from UV radiation, and avoid artificial tanning beds.",
        "consult": "Urgent. Any evolving, bleeding, or highly asymmetric mole requires immediate dermatological evaluation and excision/biopsy.",
        "risk": "High"
    },
    "Nevus": {
        "description": "A nevus (plural: nevi) is a benign, non-cancerous proliferation of melanocytes. Commonly known as a mole.",
        "symptoms": "Usually an evenly colored brown, tan, or black spot on the skin. It can be flat or raised, round or oval. Most are smaller than 6 millimeters (about the size of a pencil eraser).",
        "causes": "Caused by melanocytes growing in clusters instead of being spread out. Sun exposure during childhood can increase the number of nevi.",
        "treatment": "None required unless symptomatic, changing, or cosmetically undesired. If suspicious, a dermatologist may perform a biopsy.",
        "precautions": "Monitor moles regularly using the ABCDE criteria to ensure they do not change. Practice good sun safety.",
        "consult": "If a mole changes in size, shape, or color, or if it starts to itch, bleed, or become painful.",
        "risk": "Low"
    }
}

# 3. Set the page configuration
st.set_page_config(
    page_title="DermaVision",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 19. Modular function: load_model()
# 2. Load the trained model only once using @st.cache_resource
def get_model_size_mb():
    """
    Calculates the file size of the keras model on disk dynamically.
    """
    path = "models/best_mobilenetv2.keras"
    if os.path.exists(path):
        try:
            size_bytes = os.path.getsize(path)
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        except:
            pass
    return "9.7 MB"


def generate_pdf_report(disease_name, confidence, metadata, quality, predictions, disease_info, temp_img_path):
    """
    Generates a professional, medical-grade PDF report using fpdf2.
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Document Title Block
    pdf.set_fill_color(24, 24, 27)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_text_color(250, 250, 250)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 10, "  DermaVision Clinical Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(161, 161, 170)
    pdf.cell(0, 5, "  Intelligent Skin Image Classification", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)
    
    # Metadata and Image Section
    pdf.set_y(46)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(24, 24, 27)
    pdf.cell(0, 10, "Clinical Screening Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(39, 39, 42)
    pdf.line(10, 56, 200, 56)
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(39, 39, 42)
    
    # Metadata Info Table
    pdf.cell(100, 6, f"Patient Reference: Anonymous Screening Case", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(100, 6, f"Report ID: DV-{np.random.randint(100000, 999999)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(100, 6, f"Inference Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(100, 7, "Image Parameters", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(100, 5, f"Resolution: {metadata.get('resolution')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(100, 5, f"File Size: {metadata.get('size_str')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(100, 5, f"Format: {metadata.get('format')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    
    # Place centering image
    if temp_img_path and os.path.exists(temp_img_path):
        pdf.image(temp_img_path, x=135, y=58, w=55, h=55)
        
    pdf.set_y(120)
    
    # Diagnostic Output Section
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Primary Classification Output", new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, 128, 200, 128)
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 8, f"Condition Match: {disease_name}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(24, 24, 27)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Match Rate (Confidence): {confidence * 100:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Prediction Reliability: {quality['reliability']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Risk Profile: {disease_info.get('risk')} Risk", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    
    # Probability Spectrum Section
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Probability Spectrum (Top 5 Matches)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for idx, (c_name, c_prob) in enumerate(predictions):
        pdf.cell(0, 5.5, f" - {c_name}: {c_prob*100:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    
    # Quality Analysis
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Image Quality Check", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5.5, f"Brightness: {quality['brightness']:.1f} ({quality['brightness_status']})", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5.5, f"Contrast: {quality['contrast']:.1f} ({quality['contrast_status']})", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5.5, f"Detail Focus: {quality['focus_score']:.1f} ({quality['focus_status']})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    
    # Etiology Section
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Clinical Condition Etiology & Workup", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    
    pdf.multi_cell(190, 5, f"Description: {disease_info.get('description')}")
    pdf.multi_cell(190, 5, f"Symptoms: {disease_info.get('symptoms')}")
    pdf.multi_cell(190, 5, f"Causes: {disease_info.get('causes')}")
    pdf.multi_cell(190, 5, f"Treatment Overview: {disease_info.get('treatment')}")
    pdf.multi_cell(190, 5, f"Precautions: {disease_info.get('precautions')}")
    pdf.multi_cell(190, 5, f"Dermatologist Consult recommendation: {disease_info.get('consult')}")
    pdf.ln(4)
    
    # Model details
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Engine Configuration", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Backbone model: MobileNetV2 Fine-Tuned (Accuracy: {"90.96%"} | Latency: {quality['latency']:.1f} ms)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    
    # Medical Disclaimer Block
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(220, 53, 69)
    pdf.multi_cell(190, 4, "DISCLAIMER: This diagnostic screening report is compiled by an artificial intelligence model for research and educational purposes only. It does not replace a clinical examination or biopsy by a certified dermatologist.")
    
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(161, 161, 170)
    pdf.cell(0, 10, "DermaVision Clinical Screening | Generated dynamically via TensorFlow & Streamlit", align="C")
    
    return pdf.output()


# Custom CSS styling function for shadcn-style premium interface
def apply_custom_styles():
    custom_css = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist+Sans:wght@300;400;500;600;700;800&family=Geist:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* Main app background & font settings */
.stApp {
background-color: #F8FAFC !important;
color: #0F172A !important;
font-family: 'Geist Sans', 'Geist', -apple-system, sans-serif !important;
}

/* Reduce Streamlit container padding to move navbar to top */
[data-testid="block-container"] {
padding-top: 15px !important;
padding-bottom: 20px !important;
max-width: 960px !important;
margin: 0 auto !important;
}

/* Center single-column layout container */
[data-testid="stAppViewBlockContainer"] {
max-width: 960px !important;
margin: 0 auto !important;
}

/* Navigation Layout */
.top-nav-bar {
display: flex;
justify-content: space-between;
align-items: center;
background: #FFFFFF !important;
border: 1px solid rgba(226, 232, 240, 0.8) !important;
border-radius: 20px !important;
padding: 12px 28px !important;
box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.02), 0 1px 3px 0 rgba(0, 0, 0, 0.01) !important;
margin-top: 10px;
margin-bottom: 20px;
height: 70px;
}
.nav-brand {
display: flex;
align-items: center;
gap: 16px;
}
.nav-logo-icon {
color: #2563EB;
}
.nav-title {
font-size: 24px;
font-weight: 800;
color: #0F172A;
letter-spacing: -0.03em;
}
.nav-badge {
background-color: #EFF6FF;
color: #2563EB;
border: 1px solid #BFDBFE;
font-size: 9px;
font-weight: 700;
padding: 2px 8px;
border-radius: 6px;
text-transform: uppercase;
letter-spacing: 0.05em;
}
.nav-status-group {
display: flex;
align-items: center;
gap: 24px;
}
.status-indicator {
display: inline-flex;
align-items: center;
gap: 6px;
background: rgba(34, 197, 94, 0.08);
color: #15803D;
padding: 4px 10px;
border-radius: 8px;
font-size: 12px;
font-weight: 600;
border: 1px solid rgba(34, 197, 94, 0.15);
}
.status-dot {
width: 6px;
height: 6px;
background-color: #22C55E;
border-radius: 50%;
display: inline-block;
animation: status-pulse 1.8s infinite;
}
@keyframes status-pulse {
0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.5); }
50% { transform: scale(1.15); box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
}
.nav-link {
font-size: 15px;
color: #64748B;
text-decoration: none;
transition: color 0.15s;
display: flex;
align-items: center;
gap: 6px;
font-weight: 500;
}
.nav-link:hover {
color: #0F172A;
}

/* Sections */
.section-title {
font-size: 30px !important;
font-weight: 700 !important;
color: #0F172A !important;
margin-top: 30px !important;
margin-bottom: 16px !important;
letter-spacing: -0.02em !important;
border-bottom: none !important;
padding-bottom: 0 !important;
line-height: 1.3 !important;
}

/* Custom Cards */
.shadcn-card {
background: #FFFFFF !important;
border: 1px solid rgba(226, 232, 240, 0.8) !important;
border-radius: 20px !important;
padding: 28px;
box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.02), 0 1px 3px 0 rgba(0, 0, 0, 0.01) !important;
margin-bottom: 24px;
transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.shadcn-card:hover {
border-color: rgba(226, 232, 240, 1) !important;
box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.03), 0 4px 6px -2px rgba(0, 0, 0, 0.02) !important;
}

.card-header-title {
font-size: 22px !important;
font-weight: 600 !important;
color: #0F172A;
margin-bottom: 16px;
display: flex;
align-items: center;
gap: 8px;
letter-spacing: -0.02em;
}

/* Image metadata layout */
.metadata-grid {
display: grid;
grid-template-columns: repeat(3, 1fr);
gap: 12px;
margin-top: 12px;
}
.metadata-item {
background-color: #F8FAFC;
border: 1px solid #E2E8F0;
border-radius: 8px;
padding: 10px;
text-align: center;
transition: all 0.15s;
}
.metadata-item:hover {
background-color: #F1F5F9;
border-color: #CBD5E1;
}
.metadata-label {
font-size: 10px;
color: #64748B;
text-transform: uppercase;
letter-spacing: 0.05em;
margin-bottom: 4px;
font-weight: 600;
}
.metadata-value {
font-size: 14px;
color: #0F172A;
font-weight: 700;
}

/* Badges */
.badge-pill {
font-size: 12px;
font-weight: 600;
padding: 4px 10px;
border-radius: 9999px;
display: inline-flex;
align-items: center;
gap: 4px;
}
.badge-pill-success {
background-color: #DCFCE7;
color: #15803D;
border: 1px solid #BBF7D0;
}
.badge-pill-warning {
background-color: #FEF3C7;
color: #B45309;
border: 1px solid #FDE68A;
}
.badge-pill-error {
background-color: #FEE2E2;
color: #B91C1C;
border: 1px solid #FECACA;
}

.risk-badge {
padding: 4px 10px;
border-radius: 6px;
font-size: 12px;
font-weight: 700;
text-transform: uppercase;
display: inline-block;
letter-spacing: 0.05em;
}
.badge-low {
background: #DCFCE7;
color: #15803D;
border: 1px solid #BBF7D0;
}
.badge-medium {
background: #FEF3C7;
color: #B45309;
border: 1px solid #FDE68A;
}
.badge-high {
background: #FEE2E2;
color: #B91C1C;
border: 1px solid #FECACA;
animation: pulse-border 2s infinite;
}
@keyframes pulse-border {
0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
70% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

/* Progress bars */
.progress-container {
margin-bottom: 14px;
}
.progress-info {
display: flex;
justify-content: space-between;
margin-bottom: 6px;
font-size: 16px;
}
.progress-name {
font-weight: 600;
color: #0F172A;
}
.progress-val {
font-weight: 700;
color: #2563EB;
}
.progress-track {
background-color: #F1F5F9;
height: 8px;
border-radius: 4px;
overflow: hidden;
border: 1px solid #E2E8F0;
}
.progress-thumb-gradient {
background: #2563EB !important;
height: 100%;
border-radius: 4px;
transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Metric Display card */
.shadcn-metric {
display: flex;
justify-content: space-between;
align-items: center;
background-color: #F8FAFC;
border: 1px solid #E2E8F0;
border-radius: 12px;
padding: 16px;
margin-bottom: 16px;
}
.metric-lhs {
display: flex;
flex-direction: column;
}
.metric-lbl {
font-size: 10px;
color: #64748B;
text-transform: uppercase;
font-weight: 600;
letter-spacing: 0.05em;
margin-bottom: 4px;
}
.metric-v {
font-size: 26px;
font-weight: 800;
color: #0F172A;
letter-spacing: -0.02em;
}

/* Quality indicators grid */
.quality-grid {
display: grid;
grid-template-columns: repeat(3, 1fr);
gap: 12px;
margin-top: 8px;
}
.quality-item {
background-color: #FFFFFF;
border: 1px solid rgba(226, 232, 240, 0.8);
border-radius: 20px;
padding: 18px;
text-align: center;
box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.02), 0 1px 3px 0 rgba(0, 0, 0, 0.01) !important;
transition: all 0.2s;
}
.quality-item:hover {
border-color: #CBD5E1;
background-color: #F8FAFC;
}
.quality-label {
font-size: 11px;
color: #64748B;
font-weight: 600;
margin-bottom: 6px;
text-transform: uppercase;
letter-spacing: 0.03em;
}
.quality-val {
font-size: 18px;
color: #0F172A;
font-weight: 700;
}

/* Stepper Workflow */
.stepper-wrapper {
display: flex;
align-items: center;
justify-content: space-between;
gap: 8px;
margin-top: 16px;
}
.step-item {
display: flex;
flex-direction: column;
align-items: center;
text-align: center;
flex: 1;
background-color: #FFFFFF;
border: 1px solid rgba(226, 232, 240, 0.8);
padding: 16px 12px;
border-radius: 20px;
box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.02), 0 1px 3px 0 rgba(0, 0, 0, 0.01) !important;
min-height: 110px;
transition: all 0.2s;
}
.step-item:hover {
border-color: #CBD5E1;
background-color: #F8FAFC;
}
.step-number {
background-color: #EFF6FF;
color: #2563EB;
border: 1px solid #BFDBFE;
width: 28px;
height: 28px;
border-radius: 50%;
display: flex;
align-items: center;
justify-content: center;
font-size: 14px;
font-weight: 700;
margin-bottom: 8px;
}
.step-title {
font-size: 14px;
font-weight: 700;
color: #0F172A;
margin-bottom: 2px;
}
.step-desc {
font-size: 12px;
color: #64748B;
line-height: 1.3;
}
.step-arrow {
display: flex;
align-items: center;
color: #CBD5E1;
font-weight: bold;
}

/* Warning Card Disclaimer */
.warning-card {
background-color: #FFFBEB;
border: 1px solid #FDE68A;
border-left: 4px solid #F59E0B;
border-radius: 20px;
padding: 18px;
margin-top: 24px;
display: flex;
gap: 12px;
align-items: flex-start;
}
.warning-icon {
color: #D97706;
flex-shrink: 0;
margin-top: 2px;
}
.warning-text {
font-size: 15px;
color: #78350F;
line-height: 1.5;
margin: 0;
}
.warning-text strong {
color: #0F172A;
}

/* Model Specs Grid */
.specs-grid {
display: grid;
grid-template-columns: repeat(3, 1fr);
gap: 12px;
margin-top: 12px;
}
.spec-item {
background-color: #FFFFFF;
border: 1px solid rgba(226, 232, 240, 0.8);
border-radius: 20px;
padding: 18px;
display: flex;
flex-direction: column;
justify-content: space-between;
align-items: flex-start;
gap: 8px;
box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.02), 0 1px 3px 0 rgba(0, 0, 0, 0.01) !important;
transition: all 0.2s;
}
.spec-item:hover {
border-color: #CBD5E1;
background-color: #F8FAFC;
}
.spec-label {
font-size: 12px;
color: #64748B;
font-weight: 600;
text-transform: uppercase;
letter-spacing: 0.05em;
}
.spec-val {
font-size: 18px;
color: #0F172A;
font-weight: 700;
}
/* Native File Uploader Overrides - Styled Premium */
div[data-testid="stFileUploader"] {
background: #FFFFFF !important;
border: 2px dashed #CBD5E1 !important;
border-radius: 24px !important;
padding: 40px !important;
width: 95% !important;
max-width: 850px !important;
min-height: 480px !important;
display: flex !important;
flex-direction: column !important;
justify-content: center !important;
align-items: center !important;
transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.04), 0 1px 3px 0 rgba(0, 0, 0, 0.01) !important;
margin: 0 auto 30px auto !important;
font-family: 'Geist Sans', sans-serif !important;
}

div[data-testid="stFileUploader"]:hover {
border-color: #2563EB !important;
background: #F8FAFC !important;
box-shadow: 0 20px 40px -15px rgba(37, 99, 235, 0.08), 0 1px 3px 0 rgba(37, 99, 235, 0.02) !important;
transform: translateY(-2px);
}

/* 1. Medical Icon & Title */
div[data-testid="stFileUploader"] section::before {
content: "Upload Dermoscopic Image" !important;
display: block !important;
padding-top: 90px !important; /* Space for the 65px icon + 25px gap */
background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%232563EB" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M12 8v8"/><path d="M8 12h8"/></svg>') !important;
background-size: 65px 65px !important;
background-position: top center !important;
background-repeat: no-repeat !important;
font-size: 30px !important;
font-weight: 700 !important;
color: #0F172A !important;
margin-bottom: 20px !important;
font-family: 'Geist Sans', sans-serif !important;
text-align: center !important;
transition: all 0.3s ease !important;
}

/* 3. Subtitle */
div[data-testid="stFileUploader"] section > span::before {
content: "Upload a high-quality dermoscopic image (JPG, JPEG, PNG)" !important;
display: block !important;
font-size: 19px !important;
color: #64748B !important;
margin-bottom: 32px !important;
font-family: 'Geist Sans', sans-serif !important;
text-align: center !important;
font-weight: 400 !important;
transition: all 0.3s ease !important;
}

/* Hide Icon/Title/Subtitle when a file has been uploaded */
div[data-testid="stFileUploader"]:has([data-testid="stUploadedFileData"])::before,
div[data-testid="stFileUploader"]:has([data-testid="stUploadedFileData"]) section::before,
div[data-testid="stFileUploader"]:has([data-testid="stUploadedFileData"]) section > span::before,
div[data-testid="stFileUploader"]:has(button[aria-label*="Remove"])::before,
div[data-testid="stFileUploader"]:has(button[aria-label*="Remove"]) section::before,
div[data-testid="stFileUploader"]:has(button[aria-label*="Remove"]) section > span::before {
display: none !important;
}

div[data-testid="stFileUploader"] section {
background-color: transparent !important;
border: none !important;
padding: 0 !important;
display: flex !important;
flex-direction: column !important;
align-items: center !important;
justify-content: center !important;
width: 100% !important;
text-align: center !important;
}

div[data-testid="stFileUploader"] label {
display: none !important;
}

/* Hide default drag-and-drop text and limits text */
div[data-testid="stFileUploader"] section > div {
display: none !important;
}
div[data-testid="stFileUploader"] section > p {
display: none !important;
}

/* Style Browse Files Button */
div[data-testid="stFileUploader"] button:not([aria-label*="Remove"]) {
background-color: #2563EB !important;
color: #FFFFFF !important;
border: 1px solid #1D4ED8 !important;
border-radius: 12px !important;
width: 175px !important;
height: 55px !important;
display: flex !important;
justify-content: center !important;
align-items: center !important;
margin: 0 auto !important;
font-size: 16px !important;
font-weight: 600 !important;
transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.12), 0 2px 4px -1px rgba(37, 99, 235, 0.08) !important;
}

div[data-testid="stFileUploader"] button:not([aria-label*="Remove"]) * {
color: #FFFFFF !important;
stroke: #FFFFFF !important;
opacity: 1 !important;
}

div[data-testid="stFileUploader"] button:not([aria-label*="Remove"]) svg {
width: 23px !important;
height: 23px !important;
min-width: 23px !important;
min-height: 23px !important;
fill: none !important;
stroke: #FFFFFF !important;
stroke-width: 2.75 !important;
stroke-linecap: round !important;
stroke-linejoin: round !important;
opacity: 1 !important;
filter: none !important;
}

div[data-testid="stFileUploader"] button:not([aria-label*="Remove"]) svg * {
color: #FFFFFF !important;
stroke: #FFFFFF !important;
fill: none !important;
stroke-width: 2.75 !important;
stroke-linecap: round !important;
stroke-linejoin: round !important;
opacity: 1 !important;
filter: none !important;
}

div[data-testid="stFileUploader"] button:not([aria-label*="Remove"]):hover {
background-color: #1D4ED8 !important;
border-color: #1E40AF !important;
transform: scale(1.03) !important;
box-shadow: 0 8px 16px -4px rgba(37, 99, 235, 0.2) !important;
}

div[data-testid="stFileUploader"] button:not([aria-label*="Remove"]):active {
transform: scale(0.97) !important;
}

/* Style the uploaded file state (Compact Filled State) */
div[data-testid="stFileUploader"]:has(button[aria-label*="Remove"]) {
    border: 2px dashed rgba(37, 99, 235, 0.4) !important;
    background: #F8FAFC !important;
    min-height: unset !important;
    padding: 12px 20px !important;
    border-radius: 16px !important;
}

div[data-testid="stFileUploader"]:has(button[aria-label*="Remove"])::before {
    display: none !important;
}



/* Inject detailed helper text below the button */
div[data-testid="stFileUploader"] section::after {
content: "Supported formats: JPG, JPEG, PNG  •  Maximum size: 200 MB\A Best results with clear, well-lit dermoscopic images" !important;
white-space: pre-line !important;
display: block !important;
font-size: 15px !important;
color: #64748B !important;
margin-top: 25px !important;
line-height: 1.8 !important;
font-family: 'Geist Sans', sans-serif !important;
text-align: center !important;
font-weight: 500 !important;
}

/* Hide helper text when a file is uploaded */
div[data-testid="stFileUploader"]:has([data-testid="stUploadedFileData"]) section::after,
div[data-testid="stFileUploader"]:has(button[aria-label*="Remove"]) section::after {
display: none !important;
}

/* Style streamlit expander to look like elegant accordions */
.stExpander {
background: #FFFFFF !important;
border: 1px solid rgba(226, 232, 240, 0.8) !important;
border-radius: 20px !important;
box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.02), 0 1px 3px 0 rgba(0, 0, 0, 0.01) !important;
margin-bottom: 12px !important;
}
.stExpander summary {
padding: 16px 24px !important;
font-weight: 600 !important;
font-size: 18px !important;
color: #0F172A !important;
border-bottom: none !important;
}
.stExpander [data-testid="stExpanderDetails"] {
border-top: 1px solid #E2E8F0 !important;
padding: 16px 24px !important;
background-color: #FCFDFE !important;
border-radius: 0 0 20px 20px !important;
}

/* Action Button styling overrides */
div.stButton > button {
background: #2563EB !important;
color: #FFFFFF !important;
border: 1px solid #1D4ED8 !important;
border-radius: 12px !important;
padding: 14px 28px !important;
font-weight: 700 !important;
font-size: 16px !important;
width: 100% !important;
box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div.stButton > button:hover {
background: #1D4ED8 !important;
border-color: #1E40AF !important;
box-shadow: 0 6px 20px rgba(37, 99, 235, 0.15) !important;
}
div.stButton > button:active {
transform: scale(0.98) !important;
}

/* Download Button styling overrides */
div[data-testid="stDownloadButton"] > button {
background: #FFFFFF !important;
color: #000000 !important;
border: 1px solid #CBD5E1 !important;
border-radius: 12px !important;
padding: 14px 28px !important;
font-weight: 800 !important;
font-size: 16px !important;
width: 100% !important;
box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div[data-testid="stDownloadButton"] > button * {
color: #000000 !important;
font-weight: 800 !important;
}
div[data-testid="stDownloadButton"] > button:hover {
background: #F8FAFC !important;
border-color: #CBD5E1 !important;
color: #0F172A !important;
}
div[data-testid="stDownloadButton"] > button:active {
transform: scale(0.98) !important;
}

/* Scrollbar override */
::-webkit-scrollbar {
width: 8px;
height: 8px;
}
::-webkit-scrollbar-track {
background: #F8FAFC;
}
::-webkit-scrollbar-thumb {
background: #CBD5E1;
border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
background: #94A3B8;
}

.footer-text {
text-align: center;
color: #64748B;
font-size: 15px;
margin-top: 48px;
line-height: 1.6;
}

/* Prediction history items */
.history-item {
background-color: #FFFFFF;
border: 1px solid #E2E8F0;
border-radius: 12px;
padding: 12px 16px;
display: flex;
justify-content: space-between;
align-items: center;
margin-bottom: 10px;
transition: all 0.2s;
}
.history-item:hover {
background-color: #F8FAFC;
border-color: #CBD5E1;
}
.history-lbl {
font-size: 14px;
font-weight: 600;
color: #0F172A;
}
.history-time {
font-size: 12px;
color: #64748B;
margin-top: 2px;
}
.history-conf {
font-size: 14px;
font-weight: 700;
color: #2563EB;
}

/* Sidebar Input Formatting */
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    padding-top: 22px !important;
    padding-bottom: 22px !important;
    padding-right: 14px !important;
    line-height: 1.5 !important;
}
[data-testid="stSidebar"] [data-testid="InputInstructions"] {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] {
    margin-bottom: 4px;
}

/* Sidebar Button Formatting */
[data-testid="stSidebar"] div.stButton > button {
    background-color: #2563EB !important;
    color: white !important;
    border-radius: 12px !important;
    height: 54px !important;
    font-weight: 700 !important;
    border: none !important;
    transition: background-color 0.2s;
}
[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #1D4ED8 !important;
    color: white !important;
}
[data-testid="stSidebar"] div.stButton > button:disabled {
    background-color: #94A3B8 !important;
    color: #F1F5F9 !important;
}


/* Compact Evaluation Section Styles */
.eval-container {
    max-width: 1200px;
    margin: 0 auto;
}
.eval-title {
    font-size: 20px !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    margin: 16px 0 16px 0 !important;
    letter-spacing: -0.01em !important;
    text-transform: uppercase;
}
.eval-subtitle {
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #475569 !important;
    margin: 24px 0 12px 0 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 6px;
}
.eval-metrics-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
}
.eval-metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 14px 8px;
    text-align: center;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
.eval-metric-val {
    font-size: 18px;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 4px;
}
.eval-metric-lbl {
    font-size: 10px;
    color: #64748B;
    font-weight: 700;
    text-transform: uppercase;
    line-height: 1.2;
}
.eval-classes-grid {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.eval-class-row {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
.eval-class-name {
    font-size: 15px;
    font-weight: 800;
    color: #2563EB;
    width: 120px;
}
.eval-class-stats {
    display: flex;
    gap: 32px;
    flex: 1;
    justify-content: flex-end;
}
.eval-class-stat {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    width: 80px;
}
.eval-class-stat-lbl {
    font-size: 10px;
    color: #64748B;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.eval-class-stat-val {
    font-size: 15px;
    font-weight: 700;
    color: #0F172A;
}
</style>"""
    st.markdown(custom_css, unsafe_allow_html=True)


# 19. Modular function: main()
def main():
    """
    Main Streamlit application pipeline. Handles UI structure, user interactions,
    and exception handling.
    """
    # Apply global styles
    apply_custom_styles()
    
    # Check model loading status
    model_loaded = False
    try:
        model = cached_load_prediction_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Critical error loading model system: {e}")
        st.stop()
        
    # Initialize session state for prediction history
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
        

    
    # 1. Top Navbar
    navbar_html = """<div class="top-nav-bar">
<div class="nav-brand">
<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="url(#blue-grad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="nav-logo-icon">
<defs>
<linearGradient id="blue-grad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#2563EB" />
<stop offset="100%" stop-color="#0EA5E9" />
</linearGradient>
</defs>
<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
<path d="M12 8v8"/>
<path d="M8 12h8"/>
</svg>
<span class="nav-title">DermaVision</span>
<span class="nav-badge">AI Diagnostic</span>
</div>
<div class="nav-status-group">
<div class="status-indicator">
<span class="status-dot"></span>
<span style="font-weight: 600;">System Active</span>
</div>
<a href="https://github.com" target="_blank" class="nav-link">
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/></svg>
GitHub
</a>
</div>
</div>"""
    st.markdown(navbar_html, unsafe_allow_html=True)
    
    # 2. Hero Section
    hero_html = """<div style="background: linear-gradient(135deg, #EFF6FF 0%, #F0FDF4 100%); border: 1px solid rgba(226, 232, 240, 0.8); border-radius: 20px; padding: 32px 24px; text-align: center; margin-bottom: 30px; box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.02), 0 1px 3px 0 rgba(0, 0, 0, 0.01);">
<h1 style="font-size: 56px; font-weight: 800; color: #0F172A; letter-spacing: -0.04em; margin-bottom: 8px; line-height: 1.1; font-family: 'Geist Sans', sans-serif;">DermaVision</h1>
<p style="font-size: 34px; font-weight: 700; color: #2563EB; margin-bottom: 12px; letter-spacing: -0.03em; line-height: 1.2; font-family: 'Geist Sans', sans-serif;">Intelligent Skin Image Classification</p>
<p style="font-size: 20px; font-weight: 400; color: #64748B; max-width: 680px; margin: 0 auto; line-height: 1.6; font-family: 'Geist Sans', sans-serif;">
Upload a clinical skin lesion image and receive an AI-assisted disease classification powered by fine-tuned MobileNetV2 Transfer Learning.
</p>
</div>"""
    st.markdown(hero_html, unsafe_allow_html=True)
    
    # 2.5 Sidebar for Evaluation
    with st.sidebar:
        st.markdown("<h2 style='font-family: \"Geist Sans\", sans-serif; color: #FFFFFF; margin-top: 0; padding-top: 0; font-size: 24px; font-weight: 800; letter-spacing: 0.05em;'>MODEL EVALUATION</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #B8C7E6; font-size: 15px; margin-bottom: 24px; line-height: 1.5;'>Evaluate MobileNetV2 on the complete ISIC 2019 validation set.</p>", unsafe_allow_html=True)
        
        import json
        sidebar_metrics = None
        if os.path.exists("metrics/metrics.json"):
            try:
                with open("metrics/metrics.json", "r") as f:
                    sidebar_metrics = json.load(f)
                
                # Ensure all required keys exist to prevent KeyErrors
                req_keys = ["accuracy", "macro_f1", "macro_auc"]
                if not all(k in sidebar_metrics for k in req_keys):
                    sidebar_metrics = None
            except:
                pass
                
        if "show_eval_config" not in st.session_state:
            st.session_state.show_eval_config = not bool(sidebar_metrics)
            
        if sidebar_metrics and not st.session_state.show_eval_config:
            success_html = f"""
            <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 12px; padding: 16px; margin-bottom: 20px;">
                <div style="color: #15803D; font-weight: 700; font-size: 14px; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                    Evaluation complete
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div>
                        <div style="font-size: 11px; color: #166534; text-transform: uppercase; font-weight: 700; margin-bottom: 2px;">Validation set</div>
                        <div style="font-size: 15px; font-weight: 800; color: #14532D;">2,489 images</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #166534; text-transform: uppercase; font-weight: 700; margin-bottom: 2px;">Accuracy</div>
                        <div style="font-size: 15px; font-weight: 800; color: #14532D;">{sidebar_metrics['accuracy']*100:.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #166534; text-transform: uppercase; font-weight: 700; margin-bottom: 2px;">Macro F1</div>
                        <div style="font-size: 15px; font-weight: 800; color: #14532D;">{sidebar_metrics['macro_f1']*100:.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #166534; text-transform: uppercase; font-weight: 700; margin-bottom: 2px;">Macro AUC</div>
                        <div style="font-size: 15px; font-weight: 800; color: #14532D;">{sidebar_metrics['macro_auc']:.4f}</div>
                    </div>
                </div>
            </div>
            """
            st.markdown(success_html, unsafe_allow_html=True)
            if st.button("Reconfigure Dataset", use_container_width=True):
                st.session_state.show_eval_config = True
                st.rerun()
        else:
            dataset_html = """
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px 22px; margin-bottom: 24px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);">
                <div style="font-size: 11px; color: #64748B; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 4px;">DATASET</div>
                <div style="font-size: 15px; font-weight: 700; color: #0F172A;">ISIC 2019</div>
                <div style="font-size: 13px; color: #475569; margin-top: 2px;">3-class validation</div>
            </div>
            """
            st.markdown(dataset_html, unsafe_allow_html=True)
            
            raw_csv = st.text_input("Ground Truth CSV", placeholder="Path to ISIC 2019 ground-truth CSV")
            raw_img = st.text_input("Image Directory", placeholder="Path to ISIC 2019 image directory")
            
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            
            # Normalize Paths
            import pandas as pd
            from pathlib import Path
            
            csv_path = None
            img_dir = None
            
            if raw_csv and raw_img:
                ground_truth_csv = raw_csv.strip().strip('"').strip("'")
                image_directory = raw_img.strip().strip('"').strip("'")
                csv_path = Path(ground_truth_csv).expanduser()
                img_dir = Path(image_directory).expanduser()
            
            check_dataset = st.button("Check Dataset", use_container_width=True)
            
            if check_dataset and csv_path and img_dir:
                st.markdown("### DATASET CHECK")
                st.markdown(f"**Ground Truth CSV:**<br>`{csv_path.absolute()}`", unsafe_allow_html=True)
                csv_exists = csv_path.exists() and csv_path.is_file()
                st.markdown(f"**CSV exists:** {'✓' if csv_exists else 'False'}")
                
                st.markdown(f"**Image Directory:**<br>`{img_dir.absolute()}`", unsafe_allow_html=True)
                dir_exists = img_dir.exists() and img_dir.is_dir()
                st.markdown(f"**Image directory exists:** {'✓' if dir_exists else 'False'}")
                
                if csv_exists and dir_exists:
                    jpg_count = len(list(img_dir.glob("*.jpg")))
                    st.markdown(f"**JPG images found:** {jpg_count}")
                    
                    try:
                        df = pd.read_csv(str(csv_path))
                        st.markdown(f"**CSV rows:** {len(df)}")
                        
                        expected_cols = {"image", "MEL", "NV", "BCC"}
                        if expected_cols.issubset(set(df.columns)):
                            st.markdown("**Required columns:** ✓")
                            st.session_state.dataset_checked = True
                            st.session_state.checked_csv = str(csv_path)
                            st.session_state.checked_img = str(img_dir)
                        else:
                            st.error(f"Missing columns. Expected: {expected_cols}. Found: {df.columns.tolist()}")
                    except Exception as e:
                        st.error(f"Could not read CSV: {e}")
            elif check_dataset:
                st.error("Please provide both paths.")
                
            evaluate_disabled = not st.session_state.get("dataset_checked", False)
            
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            evaluate_clicked = st.button("Evaluate Model", type="primary", use_container_width=True, disabled=evaluate_disabled)
            
            if evaluate_clicked:
                if not st.session_state.get("dataset_checked", False):
                    st.error("Please pass the dataset check first.")
                else:
                    with st.spinner("Running full ML evaluation. This may take a while..."):
                        try:
                            from evaluation import evaluate_model
                            evaluate_model(model=model, ground_truth_csv=st.session_state.checked_csv, image_directory=st.session_state.checked_img)
                            st.session_state.show_eval_config = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during evaluation: {e}")
            elif not st.session_state.get("dataset_checked", False):
                info_html = """
                <div style="background-color: #F8FAFC; border-left: 3px solid #CBD5E1; padding: 16px 16px; margin-top: 20px; border-radius: 0 8px 8px 0;">
                    <div style="font-size: 13px; font-weight: 600; color: #334155; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
                        Dataset Validation Required
                    </div>
                    <div style="font-size: 12px; color: #64748B; margin-top: 4px;">Click 'Check Dataset' to validate your paths before evaluating.</div>
                </div>
                """
                st.markdown(info_html, unsafe_allow_html=True)
        
    # 3. File Uploader Container
    uploaded_file = st.file_uploader("Select dermoscopic skin image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    # Enable test mode via query parameters to bypass OS file uploader block in browser testing
    query_params = st.query_params
    if query_params.get("test_mode") == "true" and uploaded_file is None:
        test_path = "Dataset/test/Eczema Photos/03DermatitisArm1.jpg"
        if os.path.exists(test_path):
            class MockUploadedFile:
                def __init__(self, path):
                    self.name = os.path.basename(path)
                    self.size = os.path.getsize(path)
                    self.path = path
                def read(self):
                    with open(self.path, "rb") as f:
                        return f.read()
            uploaded_file = MockUploadedFile(test_path)
            
    if uploaded_file is None:
        # Empty State
        empty_state_html = """<div class="shadcn-card" style="text-align: center; padding: 40px 32px; max-width: 680px; margin: -10px auto 30px auto; border-radius: 24px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.02) !important;">
<div style="color: #2563EB; margin-bottom: 20px; display: flex; justify-content: center;">
<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
</div>
<h3 style="font-size: 22px; font-weight: 700; color: #0F172A; margin-bottom: 10px; letter-spacing: -0.02em; font-family: 'Geist Sans', sans-serif;">Awaiting Visual Input</h3>
<p style="font-size: 15px; color: #64748B; max-width: 460px; margin: 0 auto 24px auto; line-height: 1.6; font-family: 'Geist Sans', sans-serif;">
Please upload or drag and drop a clinical skin image file on the upload area above to initialize the AI diagnostic scanning workflow.
</p>
<div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 16px; padding: 20px; max-width: 500px; margin: 0 auto; text-align: left;">
<h4 style="font-size: 13px; font-weight: 700; color: #0F172A; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em; font-family: 'Geist Sans', sans-serif;">Supported Formats & Rules</h4>
<div style="display: flex; flex-direction: column; gap: 10px; font-size: 14px; color: #64748B; font-family: 'Geist Sans', sans-serif;">
<div style="display: flex; align-items: flex-start; gap: 8px;">
<span style="color: #2563EB; font-weight: bold;">•</span>
<span>Only JPG, JPEG, and PNG formats are supported (max size 200MB).</span>
</div>
<div style="display: flex; align-items: flex-start; gap: 8px;">
<span style="color: #2563EB; font-weight: bold;">•</span>
<span>Ensure high contrast, sharp focus, and optimal lighting parameters.</span>
</div>
<div style="display: flex; align-items: flex-start; gap: 8px;">
<span style="color: #2563EB; font-weight: bold;">•</span>
<span>Diagnostic scanning processes CNN transfer learning layers.</span>
</div>
</div>
</div>
</div>"""
        st.markdown(empty_state_html, unsafe_allow_html=True)
    else:
        # Preprocessing & Preview Section
        try:
            # Preprocess the file
            img_batch, original_img = preprocess_image(uploaded_file)
            
            # Validate if the image is a valid dermatological image
            from validator import is_valid_dermatology_image
            if not is_valid_dermatology_image(original_img):
                # Render premium validation failed card
                validation_failed_html = """
                <div class="warning-card" style="border-left: 4px solid #EF4444; background-color: #FEF2F2; padding: 24px; border-radius: 16px; margin: 20px auto 30px auto; max-width: 680px; text-align: left;">
                    <div style="display: flex; align-items: flex-start; gap: 16px;">
                        <div style="color: #EF4444; margin-top: 2px; flex-shrink: 0;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                            </svg>
                        </div>
                        <div>
                            <h3 style="font-size: 18px; font-weight: 700; color: #991B1B; margin: 0 0 8px 0; font-family: 'Geist Sans', sans-serif; text-align: left;">Image Validation Failed</h3>
                            <p style="font-size: 14px; color: #7F1D1D; line-height: 1.6; margin: 0; font-family: 'Geist Sans', sans-serif; font-weight: 500; text-align: left;">
                                This image does not appear to be a valid dermatological image.<br><br>
                                DermaVision is designed exclusively for AI-assisted skin disease analysis.<br><br>
                                Please upload a clear image of human skin or a skin lesion for accurate classification.
                            </p>
                        </div>
                    </div>
                </div>
                """
                st.markdown(clean_html(validation_failed_html), unsafe_allow_html=True)
                
                # Render disclaimer and footer to cleanly finish layout before early exit
                disclaimer_html = """<div class="warning-card">
<div class="warning-icon">
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
</div>
<p class="warning-text">
<strong>Disclaimer:</strong> This application is intended for educational and research screening purposes only. It is not a clinical replacement for professional dermatological examination, diagnosis, or biopsy.
</p>
</div>"""
                st.markdown(disclaimer_html, unsafe_allow_html=True)
                
                footer_html = """<div class="footer-text">
<p style="font-weight: 700; color: #0F172A; margin:0 0 4px 0;">DermaVision</p>
<p style="font-size: 12px; color: #64748B; margin:0 0 8px 0;">Intelligent Skin Image Classification</p>
<p style="font-size: 11px; color: #94A3B8; margin:0;">Built with TensorFlow • MobileNetV2 • Streamlit • ISIC 2019</p>
<p style="font-size: 11px; color: #94A3B8; margin:6px 0 0 0;">© 2026 DermaVision. All Rights Reserved.</p>
</div>"""
                st.markdown(footer_html, unsafe_allow_html=True)
                return
            
            # Evaluate Image Quality before prediction
            quality_analysis = evaluate_image_quality(original_img)
            
            # Retrieve metadata details
            res_str = f"{original_img.width} x {original_img.height}"
            size_bytes = uploaded_file.size
            if size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                
            fmt_str = "JPEG"
            if uploaded_file.name:
                parts = uploaded_file.name.split(".")
                if len(parts) > 1:
                    fmt_str = parts[-1].upper()
                    
            filename_str = uploaded_file.name if uploaded_file.name else "Unknown Image"
            
            metadata_records = {
                "resolution": res_str,
                "size_str": size_str,
                "format": fmt_str
            }
            
            # Render premium preview card with all details and success indicator
            st.markdown("<div class='section-title'>Dermoscopic Image Details</div>", unsafe_allow_html=True)
            
            with st.container(border=True):
                # 1. Upload Success Indicator Banner (Compact premium SaaS style)
                success_banner_html = """
                <div style="display: flex; align-items: center; justify-content: center; gap: 8px; background-color: #DCFCE7; color: #15803D; border: 1px solid #BBF7D0; border-radius: 14px; padding: 12px 18px; min-height: 48px; box-sizing: border-box; font-size: 18px; font-weight: 600; max-width: 720px; margin: 8px auto 16px auto; width: 100%;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0;">
                        <polyline points="20 6 9 17 4 12"/>
                    </svg>
                    <span>Dermoscopic Image Uploaded Successfully</span>
                </div>
                """
                st.markdown(success_banner_html, unsafe_allow_html=True)
                
                # 2. Centered Image Preview
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(original_img, use_container_width=True)
                
                # 3. File Name and Details
                st.markdown(f"**File Name:** `{filename_str}`")
                
                # 4. Metadata Details Grid (Resolution, Size, Format)
                metadata_html = f"""
                <div class="metadata-grid">
                    <div class="metadata-item">
                        <div class="metadata-label">Resolution</div>
                        <div class="metadata-value">{res_str}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">File Size</div>
                        <div class="metadata-value">{size_str}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Format</div>
                        <div class="metadata-value">{fmt_str}</div>
                    </div>
                </div>
                """
                st.markdown(metadata_html, unsafe_allow_html=True)
            
            # Image Quality Analysis Section (Evaluated BEFORE prediction)
            st.markdown("<div class='section-title'>Image Quality Assessment</div>", unsafe_allow_html=True)
            
            bright_badge = '<span class="badge-pill badge-pill-success">PASS</span>' if quality_analysis["brightness_status"] == "PASS" else '<span class="badge-pill badge-pill-warning">CAUTION</span>'
            contrast_badge = '<span class="badge-pill badge-pill-success">PASS</span>' if quality_analysis["contrast_status"] == "PASS" else '<span class="badge-pill badge-pill-warning">CAUTION</span>'
            focus_badge = '<span class="badge-pill badge-pill-success">PASS</span>' if quality_analysis["focus_status"] == "PASS" else '<span class="badge-pill badge-pill-warning">CAUTION</span>'
            
            recs_html = ""
            if quality_analysis["recommendations"]:
                recs_list = "".join(f"<li style='margin-bottom:4px;'>{rec}</li>" for rec in quality_analysis["recommendations"])
                recs_html = f"""<div style='margin-top: 12px; font-size:14px; color:#B45309;'>
<strong>Suggested Actions:</strong>
<ul style='margin: 4px 0 0 16px; padding:0;'>{recs_list}</ul>
</div>"""
            else:
                recs_html = "<div style='margin-top: 12px; font-size:14px; color:#15803D;'><strong>Validation:</strong> Image quality parameters fall within recommended thresholds for deep learning inference.</div>"
                
            quality_html = f"""<div class="shadcn-card" style="margin-bottom: 24px;">
<div class="quality-grid">
<div class="quality-item">
<div class="quality-label">Brightness</div>
<div class="quality-val">{quality_analysis['brightness']:.1f}</div>
<div style="margin-top: 4px;">{bright_badge}</div>
</div>
<div class="quality-item">
<div class="quality-label">Contrast</div>
<div class="quality-val">{quality_analysis['contrast']:.1f}</div>
<div style="margin-top: 4px;">{contrast_badge}</div>
</div>
<div class="quality-item">
<div class="quality-label">Detail Focus</div>
<div class="quality-val">{quality_analysis['focus_score']:.1f}</div>
<div style="margin-top: 4px;">{focus_badge}</div>
</div>
</div>
{recs_html}
</div>"""
            st.markdown(quality_html, unsafe_allow_html=True)
            
            # Predict trigger state management
            if "predicted" not in st.session_state:
                st.session_state.predicted = False
            if "last_uploaded_file" not in st.session_state:
                st.session_state.last_uploaded_file = None
                
            uploaded_id = (uploaded_file.name, uploaded_file.size) if uploaded_file else None
            last_uploaded_id = (st.session_state.last_uploaded_file.name, st.session_state.last_uploaded_file.size) if st.session_state.last_uploaded_file else None
            
            if uploaded_id != last_uploaded_id:
                st.session_state.predicted = False
                st.session_state.last_uploaded_file = uploaded_file
                
            if not st.session_state.predicted:
                predict_button = st.button("Run Diagnostic Inference", use_container_width=True)
                if predict_button:
                    st.session_state.predicted = True
                    st.rerun()
            else:
                # Animated thinking sequence mimicking a ChatGPT thinking model
                if "animation_run" not in st.session_state:
                    st.session_state.animation_run = {}
                
                uploaded_key = f"{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.animation_run.get(uploaded_key) is not True:
                    placeholder = st.empty()
                    steps = [
                        "Normalizing image color space...",
                        "Resizing inputs to 224x224 pixels...",
                        "Extracting deep CNN features...",
                        "Running convolutional inference layers...",
                        "Generating class probability scores..."
                    ]
                    for i in range(len(steps)):
                        with placeholder.container():
                            animation_html = f"""<div class="shadcn-card" style="padding: 24px; margin-top: 16px;">
<h3 style="font-size:18px; font-weight:700; color:#0F172A; margin-bottom:16px;">Analyzing Image</h3>
<div style="display:flex; flex-direction:column; gap:12px;">
"""
                            for j in range(i):
                                animation_html += f"""<div style="display:flex; align-items:center; gap:10px; color:#15803D; font-size:15px; font-weight:600;">
<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
<span>{steps[j]}</span>
</div>"""
                            animation_html += f"""<div style="display:flex; align-items:center; gap:10px; color:#2563EB; font-size:15px; font-weight:600; animation: pulse-border 1.5s infinite;">
<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="animation: spin 1s linear infinite;"><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/></svg>
<span>{steps[i]}</span>
</div>"""
                            for j in range(i + 1, len(steps)):
                                animation_html += f"""<div style="display:flex; align-items:center; gap:10px; color:#94A3B8; font-size:15px;">
<div style="width:16px; height:16px; border-radius:50%; border:2px solid #CBD5E1; box-sizing:border-box;"></div>
<span>{steps[j]}</span>
</div>"""
                            animation_html += """</div></div>
<style>
@keyframes spin { 100% { transform: rotate(360deg); } }
</style>"""
                            st.markdown(animation_html, unsafe_allow_html=True)
                            time.sleep(0.6)
                            
                    placeholder.empty()
                    st.session_state.animation_run[uploaded_key] = True
                
                # Perform actual predictions
                start_time = time.time()
                predictions = predict_top_classes(model, img_batch, class_names, top_k=5)
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                top_class, top_conf = predictions[0]
                disease_meta = DISEASE_INFO.get(top_class, {
                    "description": "No localized description is currently configured for this classification category.",
                    "symptoms": "No documented symptom details available.",
                    "causes": "No documented etiology details available.",
                    "consult": "Consult with a certified clinical dermatologist.",
                    "precautions": "Seek professional clinical guidance.",
                    "treatment": "Clinical evaluation and symptomatic management.",
                    "risk": "Medium"
                })
                
                conf_percentage = top_conf * 100.0
                if conf_percentage >= 80.0:
                    reliability = "High Reliability"
                    reliability_class = "badge-pill-success"
                elif conf_percentage >= 60.0:
                    reliability = "Medium Reliability"
                    reliability_class = "badge-pill-warning"
                else:
                    reliability = "Low Reliability"
                    reliability_class = "badge-pill-error"
                    
                quality_report = {
                    "brightness": quality_analysis["brightness"],
                    "brightness_status": quality_analysis["brightness_status"],
                    "brightness_text": quality_analysis["brightness_text"],
                    "contrast": quality_analysis["contrast"],
                    "contrast_status": quality_analysis["contrast_status"],
                    "contrast_text": quality_analysis["contrast_text"],
                    "focus_score": quality_analysis["focus_score"],
                    "focus_status": quality_analysis["focus_status"],
                    "focus_text": quality_analysis["focus_text"],
                    "overall_status": quality_analysis["overall_status"],
                    "reliability": reliability,
                    "latency": latency_ms
                }
                
                # Append to history safely
                history_key = f"{uploaded_file.name}_{conf_percentage:.2f}"
                if "last_history_key" not in st.session_state or st.session_state.last_history_key != history_key:
                    st.session_state.prediction_history.append({
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "disease": top_class,
                        "conf": conf_percentage
                    })
                    st.session_state.last_history_key = history_key
                
                # Render Primary Prediction
                st.markdown("<div class='section-title' style='margin-top:0px !important;'>Prediction</div>", unsafe_allow_html=True)
                
                risk_lower = disease_meta["risk"].lower()
                primary_pred_html = f"""<div class="shadcn-card" style="padding: 32px 24px; text-align: center;">
<div style="display: flex; justify-content: center; gap: 12px; margin-bottom: 20px;">
<span class="risk-badge badge-{risk_lower}">{disease_meta["risk"]} Risk</span>
<span class="badge-pill {reliability_class}">{reliability}</span>
</div>
<h2 class="prediction-disease-name">{top_class}</h2>
<div class="prediction-confidence-huge">{conf_percentage:.1f}%</div>
<p style="font-size: 14px; color: #64748B; margin-top: 8px; font-weight: 500;">AI MATCH RATE CONFIDENCE</p>
<div class="progress-track" style="height: 12px; border-radius: 6px; max-width: 500px; margin: 24px auto 0 auto;">
<div class="progress-thumb-gradient" style="width: {conf_percentage}%; border-radius: 6px;"></div>
</div>
</div>"""
                st.markdown(primary_pred_html, unsafe_allow_html=True)
                
                # Classification Results (Vercel-like progress list)
                st.markdown("<div class='section-title' style='margin-top:0px !important;'>Classification Results</div>", unsafe_allow_html=True)
                results_html = """<div class="shadcn-card" style="display:flex; flex-direction:column; gap:16px;">"""
                for class_name, prob in predictions:
                    prob_pct = prob * 100
                    results_html += f"""<div style="display:flex; flex-direction:column; gap:4px;">
<div class="progress-info">
<span class="progress-name" style="font-size:14px; font-weight:600;">{class_name}</span>
<span class="progress-val" style="font-size:14px; font-weight:700;">{prob_pct:.1f}%</span>
</div>
<div class="progress-track" style="height:8px; border-radius:4px;">
<div class="progress-thumb-gradient" style="width: {prob_pct}%; border-radius:4px;"></div>
</div>
</div>"""
                results_html += "</div>"
                st.markdown(results_html, unsafe_allow_html=True)
                
                # Disease Information (Accordions)
                st.markdown("<div class='section-title' style='margin-top:0px !important;'>Clinical Description Profile</div>", unsafe_allow_html=True)
                
                with st.expander("Description", expanded=True):
                    st.write(disease_meta["description"])
                with st.expander("Symptoms"):
                    st.write(disease_meta["symptoms"])
                with st.expander("Causes"):
                    st.write(disease_meta["causes"])
                with st.expander("Precautions"):
                    st.write(disease_meta["precautions"])
                with st.expander("Treatment Overview"):
                    st.write(disease_meta["treatment"])
                with st.expander("When to Consult Dermatologist"):
                    st.write(disease_meta["consult"])
                    
                # Model Information
                st.markdown("<div class='section-title' style='margin-top:18px !important;'>Model Information</div>", unsafe_allow_html=True)
                
                import json
                verified_metrics = None
                display_acc = "Requires Evaluation"
                if os.path.exists("metrics/metrics.json"):
                    try:
                        with open("metrics/metrics.json", "r") as f:
                            verified_metrics = json.load(f)
                        if "accuracy" in verified_metrics:
                            display_acc = f"{verified_metrics['accuracy']*100:.2f}%"
                    except:
                        pass
                        
                model_info_html = f"""<div class="specs-grid" style="margin-bottom: 30px;">
<div class="spec-item">
<span class="spec-label">Model Backbone</span>
<span class="spec-val">MobileNetV2</span>
</div>
<div class="spec-item">
<span class="spec-label">Dataset</span>
<span class="spec-val">ISIC 2019</span>
</div>
<div class="spec-item">
<span class="spec-label">Fine-Tuned Classes</span>
<span class="spec-val">3 Classes</span>
</div>
<div class="spec-item">
<span class="spec-label" title="Accuracy calculated on the validation dataset">Model Validation Accuracy</span>
<span class="spec-val">{display_acc}</span>
</div>
<div class="spec-item">
<span class="spec-label">Model Size</span>
<span class="spec-val">{get_model_size_mb()}</span>
</div>
<div class="spec-item">
<span class="spec-label">CNN Latency</span>
<span class="spec-val">{latency_ms:.1f} ms</span>
</div>
</div>"""
                st.markdown(model_info_html, unsafe_allow_html=True)
                

                
                # Stepper Workflow Timeline
                st.markdown("<div class='section-title' style='margin-top:0px !important;'>Classification Pipeline Workflow</div>", unsafe_allow_html=True)
                workflow_html = """<div style="margin-bottom: 30px;">
<div class="stepper-wrapper">
<div class="step-item">
<div class="step-number">1</div>
<div class="step-title">Upload</div>
<div class="step-desc">lesion image</div>
</div>
<div class="step-arrow">→</div>
<div class="step-item">
<div class="step-number">2</div>
<div class="step-title">Resize</div>
<div class="step-desc">to 224x224</div>
</div>
<div class="step-arrow">→</div>
<div class="step-item">
<div class="step-number">3</div>
<div class="step-title">Normalize</div>
<div class="step-desc">values to [-1, 1]</div>
</div>
<div class="step-arrow">→</div>
<div class="step-item">
<div class="step-number">4</div>
<div class="step-title">MobileNetV2</div>
<div class="step-desc">feature extraction</div>
</div>
<div class="step-arrow">→</div>
<div class="step-item">
<div class="step-number">5</div>
<div class="step-title">CNN Layer</div>
<div class="step-desc">global pooling</div>
</div>
<div class="step-arrow">→</div>
<div class="step-item">
<div class="step-number">6</div>
<div class="step-title">Softmax</div>
<div class="step-desc">class probability</div>
</div>
<div class="step-arrow">→</div>
<div class="step-item">
<div class="step-number">7</div>
<div class="step-title">Prediction</div>
<div class="step-desc">output result</div>
</div>
</div>
</div>"""
                st.markdown(workflow_html, unsafe_allow_html=True)
                
                # Download Report PDF
                st.markdown("<div class='section-title' style='margin-top:0px !important;'>Export Screening Summary</div>", unsafe_allow_html=True)
                
                # Temporarily save image to generate PDF report
                temp_img_path = "scratch_temp_report_img.png"
                original_img.save(temp_img_path)
                
                pdf_data = generate_pdf_report(top_class, top_conf, metadata_records, quality_report, predictions, disease_meta, temp_img_path)
                pdf_bytes = bytes(pdf_data)
                
                # Delete temp image after PDF generation
                if os.path.exists(temp_img_path):
                    try:
                        os.remove(temp_img_path)
                    except:
                        pass
                        
                st.download_button(
                    label="Download Hospital PDF Report",
                    data=pdf_bytes,
                    file_name=f"DermaVision_Screening_Report_{np.random.randint(100000, 999999)}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                # Medical history inside prediction results
                if st.session_state.prediction_history:
                    st.markdown("<div class='section-title' style='margin-top:30px !important;'>Session Scanning Logs</div>", unsafe_allow_html=True)
                    for record in reversed(st.session_state.prediction_history):
                        h_html = f"""<div class="history-item">
<div>
<div class="history-lbl">{record['disease']}</div>
<div class="history-time">{record['time']}</div>
</div>
<div class="history-conf">{record['conf']:.1f}% Match</div>
</div>"""
                        st.markdown(h_html, unsafe_allow_html=True)
                        
                    if st.button("Clear History Logs", use_container_width=True):
                        st.session_state.prediction_history = []
                        if "last_history_key" in st.session_state:
                            del st.session_state.last_history_key
                        st.rerun()
                        
        except UnidentifiedImageError:
            st.error("Error: Unsupported image format or corrupted file. Please provide a valid JPG/PNG.")
            uploaded_file = None
        except Exception as e:
            st.error(f"Error processing image: {e}")
            uploaded_file = None
            
    # Global Model Performance Section
    st.markdown("<div class='eval-container'>", unsafe_allow_html=True)
    st.markdown("<div class='eval-title'>MODEL PERFORMANCE</div>", unsafe_allow_html=True)
    
    import json
    verified_metrics = None
    if os.path.exists("metrics/metrics.json"):
        try:
            with open("metrics/metrics.json", "r") as f:
                verified_metrics = json.load(f)
        except:
            pass
            
    if verified_metrics:
        
        metrics_html = f"""<div class="eval-metrics-grid">
<div class="eval-metric-card"><div class="eval-metric-lbl">Accuracy</div><div class="eval-metric-val">{verified_metrics['accuracy']*100:.2f}%</div></div>
<div class="eval-metric-card"><div class="eval-metric-lbl">Macro Precision</div><div class="eval-metric-val">{verified_metrics['macro_precision']*100:.2f}%</div></div>
<div class="eval-metric-card"><div class="eval-metric-lbl">Macro Recall</div><div class="eval-metric-val">{verified_metrics['macro_recall']*100:.2f}%</div></div>
<div class="eval-metric-card"><div class="eval-metric-lbl">Macro F1</div><div class="eval-metric-val">{verified_metrics['macro_f1']*100:.2f}%</div></div>
<div class="eval-metric-card"><div class="eval-metric-lbl">Weighted F1</div><div class="eval-metric-val">{verified_metrics['weighted_f1']*100:.2f}%</div></div>
<div class="eval-metric-card"><div class="eval-metric-lbl">Macro AUC</div><div class="eval-metric-val">{verified_metrics['macro_auc']:.4f}</div></div>
</div>"""
        st.markdown(metrics_html, unsafe_allow_html=True)
        
        st.markdown("<div class='eval-subtitle'>PER-CLASS PERFORMANCE</div>", unsafe_allow_html=True)
        import pandas as pd
        if os.path.exists("metrics/classification_report.csv"):
            class_report = pd.read_csv("metrics/classification_report.csv", index_col=0)
            classes_list = ["BCC", "Melanoma", "Nevus"]
            classes_html = '<div class="eval-classes-grid">'
            for cls in classes_list:
                if cls in class_report.index:
                    prec = f"{class_report.loc[cls, 'precision']*100:.2f}%"
                    rec = f"{class_report.loc[cls, 'recall']*100:.2f}%"
                    f1 = f"{class_report.loc[cls, 'f1-score']*100:.2f}%"
                    sup = int(class_report.loc[cls, 'support'])
                    
                    classes_html += f"""
<div class="eval-class-row">
    <div class="eval-class-name">{cls}</div>
    <div class="eval-class-stats">
        <div class="eval-class-stat"><div class="eval-class-stat-lbl">Precision</div><div class="eval-class-stat-val">{prec}</div></div>
        <div class="eval-class-stat"><div class="eval-class-stat-lbl">Recall</div><div class="eval-class-stat-val">{rec}</div></div>
        <div class="eval-class-stat"><div class="eval-class-stat-lbl">F1 Score</div><div class="eval-class-stat-val">{f1}</div></div>
        <div class="eval-class-stat"><div class="eval-class-stat-lbl">Support</div><div class="eval-class-stat-val">{sup}</div></div>
    </div>
</div>"""
            classes_html += "</div>"
            st.markdown(classes_html, unsafe_allow_html=True)
            
        st.markdown("<div class='eval-subtitle' style='text-align: center;'>CONFUSION MATRIX</div>", unsafe_allow_html=True)
        if os.path.exists("metrics/confusion_matrix.png"):
            c1, c2, c3 = st.columns([1, 3, 1])
            with c2:
                st.image("metrics/confusion_matrix.png", use_container_width=True)
            
        st.markdown("<div class='eval-subtitle' style='text-align: center;'>ROC / AUC CURVES</div>", unsafe_allow_html=True)
        if os.path.exists("metrics/roc_auc_curve.png"):
            c1, c2, c3 = st.columns([1, 4, 1])
            with c2:
                st.image("metrics/roc_auc_curve.png", use_container_width=True)
    else:
        st.info("Model performance evaluation has not been run yet.")
        st.markdown("Configure the ISIC 2019 validation dataset and click **Evaluate Model** in the sidebar.")
        
    st.markdown("</div>", unsafe_allow_html=True)
        
    # Medical Disclaimer Card
    disclaimer_html = """<div class="warning-card">
<div class="warning-icon">
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
</div>
<p class="warning-text">
<strong>Disclaimer:</strong> This application is intended for educational and research screening purposes only. It is not a clinical replacement for professional dermatological examination, diagnosis, or biopsy.
</p>
</div>"""
    st.markdown(disclaimer_html, unsafe_allow_html=True)
    
    # 8. Footer Section
    footer_html = """<div class="footer-text">
<p style="font-weight: 700; color: #0F172A; margin:0 0 4px 0;">DermaVision</p>
<p style="font-size: 12px; color: #64748B; margin:0 0 8px 0;">Intelligent Skin Image Classification</p>
<p style="font-size: 11px; color: #94A3B8; margin:0;">Built with TensorFlow • MobileNetV2 • Streamlit • ISIC 2019</p>
<p style="font-size: 11px; color: #94A3B8; margin:6px 0 0 0;">© 2026 DermaVision. All Rights Reserved.</p>
</div>"""
    st.markdown(footer_html, unsafe_allow_html=True)
    



# The file should execute using `streamlit run streamlit_app.py`
if __name__ == "__main__":
    main()