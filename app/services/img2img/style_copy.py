from app.config import MODEL_PATHS, LORA_PATHS, DEVICE, BUCKET_NAME
from app.models.request_models import GenerateRequest
from app.utils.s3 import upload_image_to_s3
from app.utils.prompt_optimizer import enhance_prompt
from fastapi import APIRouter, UploadFile, File
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from ip_adapter import IPAdapter
from transformers import CLIPTextModel, CLIPTokenizer
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import boto3
import gc

s3 = boto3.client("s3")

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

def style_copy(
        request: GenerateRequest,
        image: UploadFile = File(...),
):
    print("style_copy request:", request.dict()) 
    # 모델, LoRA 경로 설정
    if request.model not in MODEL_PATHS.get("txt2img", {}):
        return JSONResponse(content={"error": "지원하지 않는 모델입니다."}, status_code=400)
    model_path = MODEL_PATHS.get("txt2img", {}).get(request.model)

    if request.lora and request.lora not in LORA_PATHS:
        return JSONResponse(content={"error": "지원하지 않는 LoRA입니다."}, status_code=400)
    lora_path = LORA_PATHS.get(request.lora)

    pipe = StableDiffusionPipeline.from_pretrained(
      model_path,
      torch_dtype=torch.float16,
      safety_checker=None
    ).to("cuda")

    pipe.load_ip_adapter(
      "h94/IP-Adapter",
      subfolder="models", 
      weight_name="ip-adapter-plus_sd15.bin"
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.use_karras_sigmas = True
    pipe.set_ip_adapter_scale(0.5)

    # 한국어 CLIP 적용
    koCLIP = "Bingsu/clip-vit-large-patch14-ko"
    pipe.text_encoder = CLIPTextModel.from_pretrained(koCLIP, torch_dtype=torch.float16)
    pipe.tokenizer = CLIPTokenizer.from_pretrained(koCLIP)

    # LoRA 적용
    if lora_path is not None:
        pipe.unet.load_attn_procs(lora_path)

    pipe.to(DEVICE)
    
    # UploadFile → PIL.Image
    input_img = Image.open(image.file).convert("RGB")

    optimized_prompt, optimized_negative, _ = enhance_prompt(
        request.prompt,
        request.negative_prompt,
        request.lora
    )

    images = pipe(
        prompt=optimized_prompt,
        ip_adapter_image=input_img,
        negative_prompt=optimized_negative,
        num_inference_steps=request.inference_steps,
        width=request.width,
        height=request.height,
        clip_skip=request.clip_skip,
        num_images_per_prompt=request.imgNum,
    ).images

    items = []

    for image in images:
        s3_key, url = upload_image_to_s3(image, folder="img2img")
        items.append({"key": s3_key, "url": url})

    unload_model(pipe)

    return JSONResponse(content={"results": items})
