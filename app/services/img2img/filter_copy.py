from app.config import MODEL_PATHS, LORA_PATHS, DEVICE
from app.models.request_models import GenerateRequest
from app.utils.s3 import upload_image_to_s3
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
import torch
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import cv2
import numpy as np

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

def disney_filter(imgNum: int, image: Image.Image) -> Image.Image:
    model_id = "Yntec/DisneyPixarCartoon768"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    korean_clip_path = "Bingsu/clip-vit-large-patch14-ko"

    # 한국어 CLIP 모델 로드
    pipe.text_encoder = CLIPTextModel.from_pretrained(korean_clip_path, torch_dtype=torch.float16)
    pipe.tokenizer = CLIPTokenizer.from_pretrained(korean_clip_path)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.use_karras_sigmas = True

    pipe.to(DEVICE)

    prompt = "픽사 스타일의 3D 애니메이션 캐릭터, 최고 화질, 선명한 화질, 고해상도"
    negative_prompt = "저해상도, 세부 묘사 부족, 어두운 색상, 저품질, 흐릿한 이미지, 왜곡됨, 이상한 얼굴, 잘못된 해부학, 깨진 텍스처, 이상한 비율, 왜곡된 신체, 불균형한 구도, 반복되는 물방울 무늬"

    # img2img 실행
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        guidance_scale=10,
        height=image.height,
        width=image.width,
        num_inference_steps=30,
        num_images_per_prompt=imgNum,
    ).images

    unload_model(pipe)

    items = []

    for image in images:
        s3_key, url = upload_image_to_s3(image, folder="img2img")
        items.append({"key": s3_key, "url": url})

    return items

def ghibli_filter(imgNum:int, image: Image.Image) -> Image.Image:
    base_model = "nitrosocke/Ghibli-Diffusion"

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )

    pipe.load_ip_adapter(
      "h94/IP-Adapter",
      subfolder="models", 
      weight_name="ip-adapter-plus_sd15.bin"
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.use_karras_sigmas = True
    pipe.set_ip_adapter_scale(0.8)

    # 이미지 전처리
    img_np = np.array(image)
    gray_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_np, 100, 200)
    edges_pil = Image.fromarray(edges).convert("RGB")
    
    pipe.load_ip_adapter(
      "h94/IP-Adapter",
      subfolder="models", 
      weight_name="ip-adapter-plus_sd15.bin"
    )

    pipe.to(DEVICE)

    prompt = "ghibli style"

    images= pipe(
        prompt=prompt,
        image=edges_pil,
        ip_adapter_image=image,
        num_inference_steps=30,
        height=image.height,
        width=image.width,
        guidance_scale=10,
        num_images_per_prompt=imgNum,
    ).images

    unload_model(pipe)

    items = []

    for image in images:
        s3_key, url = upload_image_to_s3(image, folder="img2img")
        items.append({"key": s3_key, "url": url})

    return items

def filter_copy(
        filter: str,
        imgNum: int,
        image: UploadFile = File(...),
):
    
    temp_img = Image.open(image.file).convert("RGB")
    if filter not in ["Disney", "Ghibli"]:
        return JSONResponse(content={"error": "지원하지 않는 필터입니다."}, status_code=400)

    if filter == "Disney":
        # Disney 필터 적용
        items = disney_filter(imgNum, temp_img)
        return JSONResponse(content={"results": items})
    else :
        # Ghibli 필터 적용
        items = ghibli_filter(imgNum, temp_img)
        return JSONResponse(content={"results": items})

        
        
        
