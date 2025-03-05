from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from fastapi.responses import FileResponse
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import uuid
import os
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hanbok = 'parrel777/hanbok-LoRA-ver2' 
hanok = 'parrel777/hanok-LoRA-ver1' 
base_model = 'Bingsu/my-korean-stable-diffusion-v1-5'

pipe = DiffusionPipeline.from_pretrained(
    base_model, 
    torch_dtype=torch.float16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.use_karras_sigmas = True
pipe.to("cuda")

class GenerateRequest(BaseModel):
    text: str
    model: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/generate")
async def ignore_get_generate():
    # 크롬 확장프로그램 관련
    return {"message": "GET 요청은 지원되지 않습니다. POST 요청을 사용하세요."}

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    print(f"받은 요청 데이터: {request}")  # 요청 데이터 확인
    prompt = request.text
    lora = request.model

    if lora == "hanbok":
        pipe.unet.load_attn_procs(hanbok)
    elif lora == "hanok":
        pipe.unet.load_attn_procs(hanok)
    elif lora == "k-food":
        print("미구현된 lora")
    elif lora == "base":
        pipe.unload_lora_weights()

    print(lora)
    
    if not prompt:
        return {"error": "프롬프트를 입력하세요."}

    image = pipe(
        prompt, 
        num_inference_steps=25, 
        guidance_scale=7.5,
        height=512,
        width=512,
    ).images[0]

    os.makedirs("generated_images", exist_ok=True)
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join("generated_images", filename)
    image.save(image_path)

    return FileResponse(image_path, media_type="image/jpg")