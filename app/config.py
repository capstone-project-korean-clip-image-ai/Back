import torch
import os
from dotenv import load_dotenv

load_dotenv()

# GPU 개수 자동 탐지
GPU_COUNT = torch.cuda.device_count()

# AWS S3
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# 모델 경로 설정
MODEL_PATHS = {
    "txt2img": {
        "base": "Bingsu/my-korean-stable-diffusion-v1-5",
        "DreamShaper": "Lykon/DreamShaper",
    },
    "img2img": {
        "base": "runwayml/stable-diffusion-inpainting",
    },
}

# LoRA 경로 설정
LORA_PATHS = {
    "hanbok": "daeunn/hanbok-LoRA",
    "hanok": "parrel777/hanok-LoRA-ver1",
}

DEVICE = "cuda"
