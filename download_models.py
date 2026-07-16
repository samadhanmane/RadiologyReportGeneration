import gdown, os
from pathlib import Path

def download_models():
    Path("ScratchCnnModels").mkdir(exist_ok=True)
    Path("report_gen").mkdir(exist_ok=True)

    models = {
    "ScratchCnnModels/BioViT.pth": "1jdnimRLhm5u73S8J-NKA97FU-JwsUFOC",
    "report_gen/best_decoder.pth": "1jxsCa3glD71dnoiXe8ZVuyrkl5q45RjS",
    "report_gen/vocab.json": "1KcQsIbFTmgUZZb1W_UFo79FjRRyzfLB3",
}

    for path, file_id in models.items():
        if not os.path.exists(path):
            print(f"Downloading {path}...")
            gdown.download(
                f"https://drive.google.com/uc?export=download&id={file_id}",
                path,
                quiet=False
            )
            print(f"Done → {path}")
        else:
            print(f"Already exists → {path}")

if __name__ == "__main__":
    download_models()