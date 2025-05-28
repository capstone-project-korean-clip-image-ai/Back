from app.config import MODEL_PATHS, LORA_PATHS, DEVICE
from app.models.request_models import GenerateRequest
from app.utils.s3 import upload_image_to_s3
from fastapi import APIRouter, UploadFile, File
from diffusers import StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, ControlNetModel, UniPCMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from controlnet_aux import OpenposeDetector
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2
import torch
import boto3

s3 = boto3.client("s3")

def unload_model(pipe):
    try:
        pipe.to("cpu")
        del pipe
        torch.cuda.empty_cache()
        print("모델 메모리 해제 완료")
    except Exception as e:
        print(f"모델 메모리 해제 중 오류 발생: {e}")

def edge_copy(
        request: GenerateRequest,
        image: UploadFile = File(...),
):
    # 모델, LoRA 경로 설정
    if request.model not in MODEL_PATHS.get("txt2img", {}):
        return JSONResponse(content={"error": "지원하지 않는 모델입니다."}, status_code=400)
    model_path = MODEL_PATHS.get("txt2img", {}).get(request.model)

    if request.lora and request.lora not in LORA_PATHS:
        return JSONResponse(content={"error": "지원하지 않는 LoRA입니다."}, status_code=400)
    lora_path = LORA_PATHS.get(request.lora)

    # ControlNet(Canny) 모델 로드
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float32
    )

    # Stable Diffusion + ControlNet 파이프라인 로드
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_path, controlnet=controlnet, torch_dtype=torch.float32, safety_checker=None
    )

    # 한국어 CLIP 적용
    koCLIP = "Bingsu/clip-vit-large-patch14-ko"
    pipe.text_encoder = CLIPTextModel.from_pretrained(koCLIP, torch_dtype=torch.float32)
    pipe.tokenizer = CLIPTokenizer.from_pretrained(koCLIP)

    # LoRA 적용
    if lora_path is not None:
        pipe.unet.load_attn_procs(lora_path)

    # 스케줄러 설정
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.use_karras_sigmas = True
    pipe.to(DEVICE)

    temp_img = Image.open(image.file).convert("RGB")
    img_np = np.array(temp_img)
    gray_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_np, 100, 200)
    edges_pil = Image.fromarray(edges).convert("RGB")

    images= pipe(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        image=edges_pil,
        num_inference_steps=request.inference_steps,
        height=temp_img.height,
        width=temp_img.width,
        guidance_scale=request.guidance_scale,
        clip_skip=request.clip_skip,
        num_images_per_prompt=request.imgNum,
    ).images

    items = []

    for image in images:
        s3_key, url = upload_image_to_s3(image, folder="img2img")
        items.append({"key": s3_key, "url": url})

    unload_model(pipe)

    return JSONResponse(content={"results": items})
