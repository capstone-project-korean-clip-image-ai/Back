import torch
import os
from dotenv import load_dotenv

load_dotenv()

# GPU 개수 자동 탐지
GPU_COUNT = torch.cuda.device_count()

# AWS S3
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# AWS CloudFront
CLOUDFRONT_DOMAIN = os.getenv("CLOUDFRONT_DOMAIN")

# 모델 경로 설정
MODEL_PATHS = {
    "txt2img": {
        "base": "Bingsu/my-korean-stable-diffusion-v1-5",
        "DreamShaper": "Lykon/DreamShaper",
        "toonyou": "frankjoshua/toonyou_beta6"
    },
    "inpaint": {
        "base": "runwayml/stable-diffusion-inpainting",
        "DreamShaper": "Lykon/dreamshaper-8-inpainting",
    },
}

# LoRA 경로 설정
LORA_PATHS = {
    "none": None,
    "hanbok": "daeunn/hanbok-LoRA",
    "hanok": "parrel777/hanok-LoRA-ver1",
}

DEVICE = "cuda"
