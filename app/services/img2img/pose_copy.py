from app.config import MODEL_PATHS, LORA_PATHS, DEVICE, BUCKET_NAME
from app.models.request_models import GenerateRequest
from app.utils.s3 import upload_image_to_s3
from app.utils.prompt_optimizer import enhance_prompt
from fastapi import APIRouter, UploadFile, File
from diffusers import StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, ControlNetModel, UniPCMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from controlnet_aux import OpenposeDetector
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

def pose_copy(
        request: GenerateRequest,
        image: UploadFile = File(...),
):
    print("pose_copy request:", request.dict()) 
    # 모델, LoRA 경로 설정
    if request.model not in MODEL_PATHS.get("txt2img", {}):
        return JSONResponse(content={"error": "지원하지 않는 모델입니다."}, status_code=400)
    model_path = MODEL_PATHS.get("txt2img", {}).get(request.model)

    # 다중 LoRA 검증 및 경로 조회
    loras = request.loras or []
    invalid = [l for l in loras if l not in LORA_PATHS]
    if invalid:
        return JSONResponse(content={"error": f"지원하지 않는 LoRA입니다: {invalid}"}, status_code=400)
    lora_paths = [LORA_PATHS[l] for l in loras]

    # ControlNet(OpenPose) 모델 로드
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32, safety_checker=None
    )

    # Stable Diffusion + ControlNet 파이프라인 로드
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_path, controlnet=controlnet, torch_dtype=torch.float32, safety_checker=None
    )

    # 한국어 CLIP 적용
    koCLIP = "Bingsu/clip-vit-large-patch14-ko"
    pipe.text_encoder = CLIPTextModel.from_pretrained(koCLIP, torch_dtype=torch.float32)
    pipe.tokenizer = CLIPTokenizer.from_pretrained(koCLIP)

    # LoRA 적용 (여러 LoRA 지원)
    for lp in lora_paths:
        pipe.unet.load_attn_procs(lp)

    # 스케줄러 설정
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.use_karras_sigmas = True
    pipe.to(DEVICE)

    # OpenPose Detector 로드
    pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    # UploadFile → PIL.Image
    input_img = Image.open(image.file).convert("RGB")
    skeleton =  pose_detector(input_img)

    # 프롬프트 최적화 (다중 LoRA 전달)
    optimized_prompt, optimized_negative, _ = enhance_prompt(
        request.prompt,
        request.negative_prompt,
        loras
    )

    images = pipe(
        prompt=optimized_prompt,
        negative_prompt=optimized_negative,
        image=skeleton,
        num_inference_steps=request.inference_steps,
        height=input_img.height,
        width=input_img.width,
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
