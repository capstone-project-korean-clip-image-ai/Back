import torch

GPU_COUNT = 7
# 추후 DB에 정리해서 사용
MODEL_PATHS = {
    "base": "Bingsu/my-korean-stable-diffusion-v1-5",
    "DreamShaper": "Lykon/DreamShaper",
}
LORA_PATHS = {
    "hanbok": "daeunn/hanbok-LoRA",
    "hanok": "parrel777/hanok-LoRA-ver1",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
