from app.config import MODEL_PATHS, LORA_PATHS, DEVICE
from app.models.request_models import GenerateRequest
from app.utils.s3 import upload_image_to_s3
from app.utils.prompt_optimizer import enhance_prompt
from fastapi import APIRouter, UploadFile, File
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import boto3
import gc

router = APIRouter()
s3 = boto3.client("s3")

# inpaint 는 8의 배수의 크기만 지원
def resize_img(img: Image.Image, resample_method: int) -> Image.Image:
    w, h = img.size
    new_w = max(8, (w // 8) * 8)
    new_h = max(8, (h // 8) * 8)
    return img.resize((new_w, new_h), resample_method)

# 모델 해제 함수
def unload_model(pipe):
    try:
        pipe.to("cpu")
        del pipe
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("모델 메모리 해제 완료")
    except Exception as e:
        print(f"모델 메모리 해제 중 오류 발생: {e}")

# 메인 이미지 생성 함수
def redraw_image(
        request: GenerateRequest,
        image: UploadFile = File(...), 
        mask: UploadFile = File(...),
):
    print("redraw_image request:", request.dict())
    # 모델, LoRA 경로 설정
    if request.model not in MODEL_PATHS.get("inpaint", {}):
        return JSONResponse(content={"error": "지원하지 않는 모델입니다."}, status_code=400)
    model_path = MODEL_PATHS.get("inpaint", {}).get(request.model)

    loras = request.loras or []
    invalid = [l for l in loras if l not in LORA_PATHS]
    if invalid:
        return JSONResponse(content={"error": f"지원하지 않는 LoRA입니다: {invalid}"}, status_code=400)
    lora_paths = [LORA_PATHS[l] for l in loras]

    # UploadFile → PIL.Image
    input_img = Image.open(image.file).convert("RGB")
    mask_img  = Image.open(mask.file).convert("RGB")

    # 이미지 크기 조정 (8의 배수)
    input_img = resize_img(input_img, resample_method=Image.LANCZOS)
    mask_img  = resize_img(mask_img, resample_method=Image.NEAREST)


    # 파이프라인 로딩
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        safety_checker=None
    )

    # 한국어 CLIP 적용
    koCLIP = "Bingsu/clip-vit-large-patch14-ko"
    pipe.text_encoder = CLIPTextModel.from_pretrained(koCLIP, torch_dtype=torch.float32)
    pipe.tokenizer = CLIPTokenizer.from_pretrained(koCLIP)

    # LoRA 적용 (다중 지원)
    for lp in lora_paths:
        pipe.unet.load_attn_procs(lp)

    # DreamShaper 모델은 DPM 지원X
    if "DreamShaper" in request.model:
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
    pipe.scheduler.use_karras_sigmas = True
    pipe.to(DEVICE)

    optimized_prompt, optimized_negative, _ = enhance_prompt(
        request.prompt,
        request.negative_prompt,
        loras
    )

    # 한 번에 이미지 4장 생성
    images = pipe(
        prompt=optimized_prompt,
        image=input_img,
        mask_image=mask_img,
        negative_prompt=optimized_negative,
        num_inference_steps=request.inference_steps,
        guidance_scale=request.guidance_scale,
        clip_skip=request.clip_skip,
        height=input_img.height,
        width=input_img.width,
        num_images_per_prompt=request.imgNum,
    ).images

    items = []

    for image in images:
        s3_key, url = upload_image_to_s3(image, folder="inpaint")
        items.append({"key": s3_key, "url": url})

    unload_model(pipe)

    return JSONResponse(content={"results": items})
