# AI-Powered Radiology Report Generation System

Automated chest X-ray analysis and radiology report generation using BioViT, BioBERT, and LSTM Decoder — deployed as a Flask web application with PDF report generation.

---


**Course:** Deep Learning / Artificial Intelligence
**Academic Year:** 2025-26

---

## Project Overview

This project implements an end-to-end AI pipeline for automated radiology report generation from chest X-ray images. Given a chest X-ray and optional clinical history, the system:

1. Classifies 14 chest pathologies using BioViT (Vision Transformer) — the best performing model
2. Generates a structured radiology report (FINDINGS + IMPRESSION) using BioBERT + LSTM decoder
3. Produces GradCAM saliency overlays for model interpretability
4. Generates downloadable professional PDF reports via a web interface

---

## System Architecture

```
Chest X-Ray (224x224)        Clinical History (text)
       |                              |
  BioViT Encoder               BioBERT Encoder
  (ViT-Base/16, frozen)        (frozen, 768-dim)
  768-dim features             768-dim features
       |                              |
       +----------concat-------------+
                    |
              1536-dim vector
                    |
             LSTM Decoder
          (2 layers, 512 hidden)
                    |
          Generated Report Text
          FINDINGS + IMPRESSION
                    |
           PDF Report Output
```

---

## Dataset

- **Name:** Indiana University Chest X-Ray Collection (IU X-Ray)
- **Size:** 3,955 radiology studies — 7,470 images (frontal + lateral)
- **Labels:** 14 pathology classes (Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices)
- **Usable samples after filtering:** 3,827
- **Split:** 80% Train / 10% Val / 10% Test (seed=42)
- **Source:** [https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university]

---

## Models

| Model | Purpose | Params | Weights |
|-------|---------|--------|---------|
| CheXNet (DenseNet-121) | 14-label classification | 7.98M | Fine-tuned on IU X-Ray |
| BioViT (ViT-Base/16) | Classification + Image Encoder — BEST MODEL | 85.8M | Fine-tuned on IU X-Ray |
| EfficientNet-B4 | 14-label classification | 17.6M | Fine-tuned on IU X-Ray |
| BioBERT | Clinical history text encoder | 110M | Frozen — HuggingFace pretrained |
| LSTM Decoder | Report generation | 4.8M | Trained from scratch |

**Best Model: BioViT (ViT-Base/16)**

BioViT is the best performing model across all classification metrics. It is used as the image encoder in the report generation pipeline, for GradCAM saliency visualization, and for disease confidence scoring in the deployed application. Its transformer-based self-attention mechanism captures global anatomical relationships across the entire X-ray, which CNN-based models cannot achieve with local convolution filters.

**Trained model's saved weights (Google Drive):** [https://drive.google.com/drive/folders/1Z20ZyT5hRecOSb3AuxDS1o4kSGPisKOu?usp=sharing]

---


## Project Structure

```
ScratchRadiology/
|
|-- 01_Data_Preprocessing.ipynb       # Data cleaning, EDA, preprocessing
|-- 02_CNN_Scratch_Fixed.ipynb        # Custom CNN trained from scratch (6-channel)
|-- 03_Pre-train_Model_Training.ipynb # CheXNet, BioViT, EfficientNet training
|-- NB4_GradCAM.ipynb                 # GradCAM saliency visualization
|-- NB5_ReportGeneration.ipynb        # BioBERT + LSTM report generation training
|-- NB6_Evaluation.ipynb              # BLEU, ROUGE evaluation metrics
|-- NB7_App.ipynb                     # Flask app and frontend generation
|
|-- app.py                            # Flask REST API backend
|-- index.html                        # Frontend web interface
|
|-- ScratchCnnModels/                 # Saved model checkpoints
|   |-- BioViT.pth                    # Best model
|   |-- CheXNet.pth
|   `-- EfficientNet.pth
|
|-- report_gen/                       # Report generation outputs
|   |-- best_decoder.pth              # Best LSTM decoder weights
|   |-- vocab.json                    # Vocabulary (word2idx)
|   |-- training_curve.json           # Training loss history
|   |-- training_curve.png            # Loss curve plot
|   `-- evaluation/
|       |-- evaluation_results.json   # BLEU + ROUGE scores
|       |-- sample_results.json       # Generated vs reference reports
|       `-- evaluation_scores.png     # Metrics bar chart
|
|-- preprocessed_df.csv               # Preprocessed dataset CSV
`-- requirements.txt                  # Python dependencies
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone[ ](https://github.com/samadhanmane/RadiologyReportGeneration.git]
cd ScratchRadiology
```

### 2. Create Virtual Environment

```bash
python -m venv gpu_env

# Windows
gpu_env\Scripts\Activate.ps1

# Linux / Mac
source gpu_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Model Weights

Download the model weights from Google Drive and place them in the correct folders.

**Google Drive Link:** [https://drive.google.com/drive/folders/1Z20ZyT5hRecOSb3AuxDS1o4kSGPisKOu?usp=sharing]

```
Download and place files as follows:
  BioViT.pth       ->  ScratchRadiology/ScratchCnnModels/BioViT.pth
  CheXNet.pth      ->  ScratchRadiology/ScratchCnnModels/CheXNet.pth
  EfficientNet.pth ->  ScratchRadiology/ScratchCnnModels/EfficientNet.pth
  best_decoder.pth ->  ScratchRadiology/report_gen/best_decoder.pth
  vocab.json       ->  ScratchRadiology/report_gen/vocab.json
```

### 5. Download Dataset

Download the IU X-Ray dataset and update the image paths in preprocessed_df.csv to match your local directory.

**Dataset Source:** https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university

### 6. Run the Application

```bash
python app.py
```

Open browser and go to:

```
http://localhost:5000
```

---

## How to Use the App

1. Open http://localhost:5000 in your browser
2. Upload a chest X-ray image (PNG or JPG)
3. Fill in patient details (Name, Age, Sex, PID, Referring Doctor)
4. Enter clinical history (example: shortness of breath, chest pain)
5. Click Analyze X-Ray and Generate Report
6. View disease confidence scores and GradCAM saliency overlay
7. Click Download PDF Report to get the professional radiology report

---

## Running the Notebooks

Run notebooks strictly in this order:

```
01_Data_Preprocessing.ipynb       ->  Preprocess dataset
02_CNN_Scratch_Fixed.ipynb        ->  Train custom CNN from scratch
03_Pre-train_Model_Training.ipynb ->  Train BioViT, CheXNet, EfficientNet
NB4_GradCAM.ipynb                 ->  Generate GradCAM visualizations
NB5_ReportGeneration.ipynb        ->  Train LSTM report generation model
NB6_Evaluation.ipynb              ->  Evaluate BLEU and ROUGE metrics
NB7_App.ipynb                     ->  Generate app.py and index.html
```

Note: Every notebook is self-contained and re-runnable independently without depending on variables from other notebooks.

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
transformers>=4.30.0
flask>=2.3.0
flask-cors>=4.0.0
albumentations>=1.3.0
opencv-python>=4.8.0
Pillow>=9.5.0
reportlab>=4.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
nltk>=3.8.0
rouge-score>=0.1.2
tqdm>=4.65.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Key Technical Contributions

1. BioViT as best model — ViT-Base/16 fine-tuned on IU X-Ray achieves macro F1 of 0.74 and ROC-AUC of 0.85, outperforming CheXNet and EfficientNet-B4
2. Multi-modal fusion — BioViT image features (768-dim) and BioBERT text features (768-dim) concatenated to a 1536-dim conditioning vector for the LSTM decoder
3. Feature precomputation — Both encoders run only once before training, reducing training time from 30+ hours to 6 minutes (5-8x speedup)
4. GradCAM on Vision Transformer — Saliency visualization applied to the last transformer block for radiologist-interpretable attention maps
5. Custom CNN from scratch — 6-channel CNN (frontal + lateral images concatenated) trained without any pretrained weights (NB2)
6. End-to-end deployment — Flask REST API with HTML frontend and ReportLab PDF generation in a single deployable system

---

## Deployment

### Local Deployment

```bash
python app.py
```

Open: http://localhost:5000

### HuggingFace Spaces Deployment

This project is deployed on HuggingFace Spaces.

**HuggingFace Spaces Link:** 

```
https://huggingface.co/spaces/PreetiK2172/ScratchRadiology
```

To access the deployed app:
1. Open the HuggingFace Spaces link above
2. Wait for the Space to load — first load may take 1 to 2 minutes as models are initialized
3. Upload a chest X-ray image and fill in patient details
4. Click Analyze to generate the report and download the PDF

Note: The HuggingFace Spaces deployment runs on CPU. For faster inference, use the local deployment with a GPU.



