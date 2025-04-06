from app.config import MODEL_PATHS, LORA_PATHS, DEVICE
from app.models.request_models import GenerateRequest
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from fastapi.responses import FileResponse
import torch
import uuid
import os

# 모델 해제 함수
def unload_model(pipe):
    try:
        pipe.to("cpu")
        del pipe
        torch.cuda.empty_cache()
        print("모델 메모리 해제 완료")
    except Exception as e:
        print(f"모델 메모리 해제 중 오류 발생: {e}")

def generate_image(request: GenerateRequest):
    # 사전에 모델이 로드 되어있는 경우 해제
    if 'pipe' in globals():
        print("기존 모델 해제 중...")
        unload_model(pipe)


    # 모델 확인
    model_path = MODEL_PATHS.get("txt2img", {}).get(request.model)
    lora_path = LORA_PATHS.get(request.lora)

    # 모델 설정
    pipe = DiffusionPipeline.from_pretrained(
        model_path, 
        torch_dtype=torch.float32,
        safety_checker=None
    )

    # 한국어 CLIP 교체
    koCLIP = "Bingsu/clip-vit-large-patch14-ko"
    korean_clip_model = CLIPTextModel.from_pretrained(
        koCLIP,
        torch_dtype=torch.float32
    )
    korean_tokenizer = CLIPTokenizer.from_pretrained(koCLIP)

    pipe.text_encoder = korean_clip_model
    pipe.tokenizer = korean_tokenizer

    # LoRA 적용
    if lora_path:
        pipe.unet.load_attn_procs(lora_path)

    # 스케쥴러 설정 
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.use_karras_sigmas = True

    pipe.to(DEVICE)  

    image = pipe(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.inference_steps, 
        height=512,
        width=512,
        guidance_scale = request.guidance_scale,
        clip_skip=request.clip_skip
    ).images[0]

    save_dir = "app/generated_images"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(save_dir, filename)
    image.save(image_path)
    
    print(image_path)

    # 생성 후 모델 해제
    unload_model(pipe)

    return FileResponse(image_path, media_type="image/jpeg")
