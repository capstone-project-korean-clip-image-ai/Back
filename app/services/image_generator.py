from app.config import MODEL_PATHS, LORA_PATHS, DEVICE, BUCKET_NAME
from app.models.request_models import GenerateRequest
from app.utils.s3 import upload_image_to_s3
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from fastapi.responses import JSONResponse
import torch
import boto3
import uuid
from io import BytesIO
from datetime import datetime
from PIL import JpegImagePlugin, TiffImagePlugin

# S3 설정
s3 = boto3.client("s3")

# 모델 해제 함수
def unload_model(pipe):
    try:
        pipe.to("cpu")
        del pipe
        torch.cuda.empty_cache()
        print("모델 메모리 해제 완료")
    except Exception as e:
        print(f"모델 메모리 해제 중 오류 발생: {e}")

# 메타데이터 구성 (WebUI 스타일)
def build_metadata(request: GenerateRequest, seed: int = None) -> str:
    return f"""Prompt: {request.prompt}
Negative prompt: {request.negative_prompt}
Steps: {request.inference_steps}, CFG scale: {request.guidance_scale}
Clip skip: {request.clip_skip}, Size: 512x512
Model: {request.model}, LoRA: {request.lora}
Created Date: {datetime.utcnow().isoformat()}Z
Seed: {seed if seed else 'N/A'}
"""

# 메인 이미지 생성 함수
def generate_image(request: GenerateRequest):
    model_path = MODEL_PATHS.get("txt2img", {}).get(request.model)
    lora_path = LORA_PATHS.get(request.lora)

    # 파이프라인 로딩
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        safety_checker=None
    )

    # 한국어 CLIP 적용
    koCLIP = "Bingsu/clip-vit-large-patch14-ko"
    pipe.text_encoder = CLIPTextModel.from_pretrained(koCLIP, torch_dtype=torch.float32)
    pipe.tokenizer = CLIPTokenizer.from_pretrained(koCLIP)

    if lora_path:
        pipe.unet.load_attn_procs(lora_path)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.use_karras_sigmas = True
    pipe.to(DEVICE)

    # 한 번에 이미지 4장 생성
    images = pipe(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.inference_steps,
        height=512,
        width=512,
        guidance_scale=request.guidance_scale,
        clip_skip=request.clip_skip,
        num_images_per_prompt=4
    ).images

    presigned_urls = []

    for image in images:
        url = upload_image_to_s3(image, bucket_name=BUCKET_NAME, folder="generated_images")
        presigned_urls.append(url)

    unload_model(pipe)

    return JSONResponse(content={"urls": presigned_urls})
