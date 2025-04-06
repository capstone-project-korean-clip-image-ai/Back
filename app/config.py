import torch

# GPU 개수 자동 탐지
GPU_COUNT = torch.cuda.device_count()

# 모델 경로 설정
MODEL_PATHS = {
    "txt2img": {
        "base": "Bingsu/my-korean-stable-diffusion-v1-5",
        "DreamShaper": "Lykon/DreamShaper",
    },
    "img2img": {
        "base": "Bingsu/my-korean-sd1.5-img2img",
    },
}

# LoRA 경로 설정
LORA_PATHS = {
    "hanbok": "daeunn/hanbok-LoRA",
    "hanok": "parrel777/hanok-LoRA-ver1",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
