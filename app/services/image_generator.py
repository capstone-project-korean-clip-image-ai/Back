from app.config import MODEL_PATHS, LORA_PATHS, DEVICE
from app.models.request_models import GenerateRequest
from app.utils.s3 import upload_image_to_s3
from app.utils.prompt_optimizer import enhance_prompt
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from fastapi.responses import JSONResponse
import torch
import boto3
import gc

# S3 설정
s3 = boto3.client("s3")

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
def generate_image(request: GenerateRequest):
    if request.model not in MODEL_PATHS.get("txt2img", {}):
        print(f"지원하지 않는 모델: {request.model}")
        return JSONResponse(content={"error": "지원하지 않는 모델입니다."}, status_code=400)
    model_path = MODEL_PATHS.get("txt2img", {}).get(request.model)

    if request.lora not in LORA_PATHS:
        print(f"지원하지 않는 LoRA: {request.lora}")
        return JSONResponse(content={"error": "지원하지 않는 LoRA입니다."}, status_code=400)
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

    if lora_path is not None:
        pipe.unet.load_attn_procs(lora_path)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.use_karras_sigmas = True
    pipe.to(DEVICE)

    optimized_prompt, optimized_negative, _ = enhance_prompt(
        request.prompt,
        request.negative_prompt,
        request.lora
    )

    images = pipe(
        prompt=optimized_prompt,
        negative_prompt=optimized_negative,
        num_inference_steps=request.inference_steps,
        height=request.height,
        width=request.width,
        guidance_scale=request.guidance_scale,
        clip_skip=request.clip_skip,
        num_images_per_prompt=request.imgNum,
    ).images

    items = []

    for image in images:
        s3_key, url = upload_image_to_s3(image, folder="generated_images")
        items.append({"key": s3_key, "url": url})

    unload_model(pipe)

    return JSONResponse(content={"results": items})
