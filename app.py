
import os, sys, json, re, io, base64, warnings, datetime
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable, Image as RLImage)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

app  = Flask(__name__)
CORS(app)
@app.route("/")
def index():
    return send_file("index.html")

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = 224
NUM_CLASSES = 14
MAX_SEQ_LEN = 150
EMBED_DIM   = 256
HIDDEN_DIM  = 512
NUM_LAYERS  = 2
SAVE_DIR    = Path("report_gen")
PRETRAIN    = Path("ScratchCnnModels")
VOCAB_PATH  = SAVE_DIR / "vocab.json"          # ← FIX: was missing

DISEASE_LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
    "Lung Opacity","No Finding","Pleural Effusion",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]

class BioViT(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.vit      = timm.create_model("vit_base_patch16_224", pretrained=False)
        in_feats      = self.vit.head.in_features
        self.vit.head = nn.Identity()
        self.head     = nn.Sequential(nn.LayerNorm(in_feats),
                                      nn.Linear(in_feats, num_classes))
    def forward(self, x): return self.head(self.vit(x))

# class ImageEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         biovit = BioViT(num_classes=NUM_CLASSES)
#         ckpt   = torch.load(PRETRAIN / "BioViT.pth", map_location=DEVICE, weights_only=False)
#         biovit.load_state_dict(ckpt)
#         self.vit      = biovit.vit
#         self.feat_dim = 768
#         for p in self.vit.parameters(): p.requires_grad = False
#     def forward(self, x): return self.vit(x)
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        biovit = BioViT(num_classes=NUM_CLASSES)

        ckpt = torch.load(
            PRETRAIN / "BioViT.pth",
            map_location=DEVICE,
            weights_only=False
        )

        # Support both checkpoint formats
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            biovit.load_state_dict(ckpt["model_state_dict"])
        else:
            biovit.load_state_dict(ckpt)

        self.vit = biovit.vit
        self.feat_dim = 768

        for p in self.vit.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.vit(x)

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model_name     = "dmis-lab/biobert-base-cased-v1.2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        try:
            self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                                   use_safetensors=True)
        except:
            self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for p in self.bert.parameters(): p.requires_grad = False
        self.feat_dim = 768
    def forward(self, texts):
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=64, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): out = self.bert(**enc)
        return out.last_hidden_state[:, 0, :]

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()
        encoder_dim    = 1536
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=pad_idx)
        self.fc_init_h = nn.Linear(encoder_dim, HIDDEN_DIM)
        self.fc_init_c = nn.Linear(encoder_dim, HIDDEN_DIM)
        self.lstm      = nn.LSTM(EMBED_DIM, HIDDEN_DIM, NUM_LAYERS,
                                 batch_first=True, dropout=0.3)
        self.fc_out    = nn.Linear(HIDDEN_DIM, vocab_size)
        self.dropout   = nn.Dropout(0.3)
    def forward(self, encoder_out, input_ids):
        embeds = self.dropout(self.embedding(input_ids))
        h = self.fc_init_h(encoder_out).unsqueeze(0).repeat(NUM_LAYERS, 1, 1)
        c = self.fc_init_c(encoder_out).unsqueeze(0).repeat(NUM_LAYERS, 1, 1)
        out, _ = self.lstm(embeds, (h, c))
        return self.fc_out(out)

class ViTGradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer = model.vit.blocks[-1].norm1
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)
    def _save_act(self, m, i, o):  self.activations = o.detach()
    def _save_grad(self, m, gi, go): self.gradients = go[0].detach()
    def generate(self, img_tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(img_tensor)
        logits[0, class_idx].backward()
        grads = self.gradients[:, 1:, :]
        acts  = self.activations[:, 1:, :]
        weights = grads.mean(dim=-1, keepdim=True)
        cam = (weights * acts).sum(dim=-1)
        cam = cam.squeeze().reshape(14, 14).cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0: cam = cam / cam.max()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam

# ── Load models at startup ────────────────────────────────────
print("Loading models...")

with open(VOCAB_PATH, encoding="utf-8") as f:
    vd = json.load(f)

if "word2idx" in vd:
    word2idx = vd["word2idx"]
    idx2word = {int(k): v for k, v in vd["idx2word"].items()}
else:
    word2idx = vd
    idx2word = {v: k for k, v in vd.items()}

VOCAB_SIZE = len(word2idx)
PAD_IDX    = word2idx["<PAD>"]
SOS_IDX    = word2idx["<SOS>"]
EOS_IDX    = word2idx["<EOS>"]

img_encoder  = ImageEncoder().to(DEVICE).eval()
text_encoder = TextEncoder().to(DEVICE).eval()
decoder      = LSTMDecoder(VOCAB_SIZE, PAD_IDX).to(DEVICE)
ckpt = torch.load(SAVE_DIR / "best_decoder.pth", map_location=DEVICE, weights_only=False)
decoder.load_state_dict(ckpt["decoder_state"])
decoder.eval()

disease_model = BioViT(NUM_CLASSES).to(DEVICE)
ckpt2 = torch.load(PRETRAIN / "BioViT.pth", map_location=DEVICE, weights_only=False)
disease_model.load_state_dict(ckpt2["model_state_dict"])
disease_model.eval()

gradcam = ViTGradCAM(disease_model)
print(f"All models loaded on {DEVICE}")

# ── Helpers ───────────────────────────────────────────────────
def preprocess_image(img_bytes):
    arr   = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])
    return transform(image=img)["image"].unsqueeze(0).to(DEVICE), img


def run_pipeline(img_tensor, raw_img, clinical_history):
    with torch.no_grad():
        img_feats   = img_encoder(img_tensor)
        text_feats  = text_encoder([clinical_history])
        encoder_out = torch.cat([img_feats, text_feats], dim=-1)

        logits = disease_model(img_tensor)
        probs  = torch.sigmoid(logits)[0].cpu().numpy()

        input_ids = torch.tensor([[SOS_IDX]], dtype=torch.long).to(DEVICE)
        generated = []
        h = decoder.fc_init_h(encoder_out).unsqueeze(0).repeat(NUM_LAYERS, 1, 1)
        c = decoder.fc_init_c(encoder_out).unsqueeze(0).repeat(NUM_LAYERS, 1, 1)
        for _ in range(MAX_SEQ_LEN):
            emb        = decoder.embedding(input_ids)
            out, (h,c) = decoder.lstm(emb, (h, c))
            logit      = decoder.fc_out(out[:, -1, :])
            next_tok   = logit.argmax(dim=-1).item()
            if next_tok == EOS_IDX: break
            generated.append(next_tok)
            input_ids = torch.tensor([[next_tok]], dtype=torch.long).to(DEVICE)

    words  = [idx2word.get(i,"") for i in generated
               if i not in [PAD_IDX, SOS_IDX, EOS_IDX]]
    report = " ".join(words)

    findings   = ""
    impression = ""
    if "impression:" in report.lower():
        parts      = re.split(r"impression:", report, flags=re.IGNORECASE, maxsplit=1)
        findings   = parts[0].replace("FINDINGS:","").replace("findings:","").strip()
        impression = parts[1].strip()
    else:
        findings   = report.strip()
        impression = "A narrative report."

    top_disease_idx = int(np.argmax(probs))
    disease_model.zero_grad()
    cam = gradcam.generate(img_tensor, top_disease_idx)
    heatmap     = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    raw_resized = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))
    raw_bgr     = cv2.cvtColor(raw_resized, cv2.COLOR_RGB2BGR)
    overlay     = cv2.addWeighted(raw_bgr, 0.55, heatmap, 0.45, 0)
    overlay_b64 = base64.b64encode(
        cv2.imencode(".png", overlay)[1].tobytes()).decode()
    orig_b64    = base64.b64encode(
        cv2.imencode(".png", raw_bgr)[1].tobytes()).decode()

    diseases = [{"name": DISEASE_LABELS[i], "prob": float(probs[i])}
                for i in np.argsort(probs)[::-1]]

    return {
        "findings"   : findings,
        "impression" : impression,
        "diseases"   : diseases,
        "overlay_b64": overlay_b64,
        "orig_b64"   : orig_b64,
        "top_disease": DISEASE_LABELS[top_disease_idx],
    }


def generate_pdf(patient_name, age, sex, pid, ref_doctor,
                 clinical_history, findings, impression,
                 diseases, orig_b64, overlay_b64):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.5*cm, bottomMargin=2*cm)
    styles    = getSampleStyleSheet()
    story     = []
    W, H      = A4
    content_w = W - 4*cm

    BLUE  = colors.HexColor("#1a4f8a")
    LBLUE = colors.HexColor("#e8f0fb")
    GRAY  = colors.HexColor("#555555")
    LGRAY = colors.HexColor("#f5f5f5")
    GREEN = colors.HexColor("#1a7a4a")
    ORANGE= colors.HexColor("#c45c00")
    RED   = colors.HexColor("#b91c1c")

    def ps(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=styles[parent], **kw)

    hosp_style  = ps("Hosp",  fontSize=22, textColor=BLUE, fontName="Helvetica-Bold",
                     alignment=TA_CENTER, spaceAfter=2)
    url_style   = ps("URL",   fontSize=9,  textColor=GRAY, alignment=TA_CENTER, spaceAfter=6)
    title_style = ps("Title", fontSize=13, textColor=BLUE, fontName="Helvetica-Bold",
                     alignment=TA_CENTER, spaceBefore=10, spaceAfter=6)
    sec_style   = ps("Sec",   fontSize=10, textColor=BLUE, fontName="Helvetica-Bold",
                     spaceBefore=10, spaceAfter=4)
    body_style  = ps("Body",  fontSize=9.5, textColor=colors.black,
                     leading=15, alignment=TA_JUSTIFY, spaceAfter=4)
    label_style = ps("Lbl",   fontSize=9,  textColor=GRAY, fontName="Helvetica-Bold")
    val_style   = ps("Val",   fontSize=9,  textColor=colors.black)

    story.append(Paragraph("SMART IMAGING CENTER", hosp_style))
    story.append(Paragraph("www.smartimaging.com", url_style))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=8))

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    info_data = [
        [Paragraph(f"<b>{patient_name.upper()}</b>", ps("PN", fontSize=11, fontName="Helvetica-Bold")),
         Paragraph(f"PID : {pid}", label_style),
         Paragraph("Reported on:", label_style)],
        [Paragraph(f"Age : {age} Years", val_style),
         Paragraph(f"Ref : {ref_doctor}", val_style),
         Paragraph(now, val_style)],
        [Paragraph(f"Sex : {sex}", val_style), "", ""],
    ]
    info_table = Table(info_data, colWidths=[content_w*0.38, content_w*0.35, content_w*0.27])
    info_table.setStyle(TableStyle([
        ("VALIGN",      (0,0),(-1,-1),"TOP"),
        ("TOPPADDING",  (0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    story.append(info_table)
    story.append(HRFlowable(width="100%", thickness=0.8, color=GRAY, spaceBefore=6, spaceAfter=10))
    story.append(Paragraph("CHEST RADIOGRAPH (PA VIEW)", title_style))
    story.append(HRFlowable(width="40%", thickness=1, color=BLUE,
                             hAlign="CENTER", spaceAfter=12))

    try:
        orig_bytes    = base64.b64decode(orig_b64)
        overlay_bytes = base64.b64decode(overlay_b64)
        orig_pil    = Image.open(io.BytesIO(orig_bytes)).convert("RGB")
        overlay_pil = Image.open(io.BytesIO(overlay_bytes)).convert("RGB")
        orig_io    = io.BytesIO(); orig_pil.save(orig_io,    "PNG"); orig_io.seek(0)
        overlay_io = io.BytesIO(); overlay_pil.save(overlay_io,"PNG"); overlay_io.seek(0)
        img_w = content_w * 0.47
        img_h = img_w
        img_table = Table(
            [[RLImage(orig_io, width=img_w, height=img_h),
              RLImage(overlay_io, width=img_w, height=img_h)]],
            colWidths=[content_w*0.5, content_w*0.5]
        )
        img_table.setStyle(TableStyle([
            ("ALIGN",         (0,0),(-1,-1),"CENTER"),
            ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
            ("LEFTPADDING",   (0,0),(-1,-1),4),
            ("RIGHTPADDING",  (0,0),(-1,-1),4),
        ]))
        story.append(img_table)
        caption_data = [[
            Paragraph("Original X-Ray",
                      ps("C1",fontSize=8,textColor=GRAY,alignment=TA_CENTER)),
            Paragraph("Saliency Activation Map",
                      ps("C2",fontSize=8,textColor=GRAY,alignment=TA_CENTER)),
        ]]
        caption_table = Table(caption_data, colWidths=[content_w*0.5, content_w*0.5])
        caption_table.setStyle(TableStyle([
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("TOPPADDING",(0,0),(-1,-1),2)
        ]))
        story.append(caption_table)
        story.append(Spacer(1, 10))
    except Exception as e:
        story.append(Paragraph(f"[Images unavailable: {e}]", body_style))

    story.append(Paragraph("Clinical History", sec_style))
    story.append(Paragraph(clinical_history or "Not provided.", body_style))
    story.append(Paragraph("Technique", sec_style))
    story.append(Paragraph(
        "Standard frontal projection of the chest was obtained. "
        "AI-assisted analysis was performed using BioViT deep learning model.", body_style))
    story.append(Paragraph("Comparison", sec_style))
    story.append(Paragraph("No prior examinations are available for comparison.", body_style))
    story.append(Paragraph("Findings", sec_style))
    story.append(Paragraph(findings or "No significant findings.", body_style))
    story.append(Paragraph("Impression", sec_style))
    story.append(Paragraph(impression or "A narrative report.", body_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("AI Disease Confidence Scores", sec_style))
    top5 = sorted(diseases, key=lambda x: x["prob"], reverse=True)[:5]
    dis_rows = [[
        Paragraph("Disease",    ps("DH",  fontSize=9, fontName="Helvetica-Bold", textColor=colors.white)),
        Paragraph("Confidence", ps("DH2", fontSize=9, fontName="Helvetica-Bold", textColor=colors.white)),
        Paragraph("Level",      ps("DH3", fontSize=9, fontName="Helvetica-Bold", textColor=colors.white)),
    ]]
    for d in top5:
        pct = d["prob"] * 100
        lvl = "HIGH" if pct >= 60 else "MODERATE" if pct >= 35 else "LOW"
        clr = RED if pct >= 60 else ORANGE if pct >= 35 else GREEN
        dis_rows.append([
            Paragraph(d["name"], ps("DN", fontSize=9)),
            Paragraph(f"{pct:.1f}%", ps("DP", fontSize=9, alignment=TA_CENTER)),
            Paragraph(lvl, ps("DL", fontSize=9, textColor=clr,
                              fontName="Helvetica-Bold", alignment=TA_CENTER)),
        ])
    dis_table = Table(dis_rows, colWidths=[content_w*0.55, content_w*0.22, content_w*0.23])
    dis_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  BLUE),
        ("BACKGROUND",    (0,1),(-1,1),  LBLUE),
        ("BACKGROUND",    (0,2),(-1,2),  LGRAY),
        ("BACKGROUND",    (0,3),(-1,3),  LBLUE),
        ("BACKGROUND",    (0,4),(-1,4),  LGRAY),
        ("BACKGROUND",    (0,5),(-1,5),  LBLUE),
        ("ALIGN",         (1,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.white),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ]))
    story.append(dis_table)
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.8, color=GRAY))
    story.append(Paragraph(
        "Thanks for Reference &nbsp;&nbsp;&nbsp; ****End of Report****",
        ps("End", fontSize=9, textColor=GRAY, alignment=TA_CENTER, spaceBefore=8)))
    story.append(Paragraph(
        "This report is AI-assisted. Final diagnosis must be confirmed by a qualified radiologist.",
        ps("Disc", fontSize=7.5, textColor=GRAY, alignment=TA_CENTER, spaceBefore=4)))

    doc.build(story)
    buf.seek(0)
    return buf


# ── API Routes ────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        img_bytes = request.files["image"].read()
        history   = request.form.get("history", "")
        img_tensor, raw_img = preprocess_image(img_bytes)
        result = run_pipeline(img_tensor, raw_img, history)
        return jsonify({"status": "ok", "data": result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/download_pdf", methods=["POST"])
def download_pdf():
    try:
        body = request.get_json()
        pdf_buf = generate_pdf(
            patient_name     = body.get("patient_name",    "Patient"),
            age              = body.get("age",             "N/A"),
            sex              = body.get("sex",             "N/A"),
            pid              = body.get("pid",             "N/A"),
            ref_doctor       = body.get("ref_doctor",      "Self-Referral"),
            clinical_history = body.get("clinical_history",""),
            findings         = body.get("findings",        ""),
            impression       = body.get("impression",      ""),
            diseases         = body.get("diseases",        []),
            orig_b64         = body.get("orig_b64",        ""),
            overlay_b64      = body.get("overlay_b64",     ""),
        )
        return send_file(pdf_buf, mimetype="application/pdf",
                         as_attachment=True,
                         download_name="Radiology_Report.pdf")
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
