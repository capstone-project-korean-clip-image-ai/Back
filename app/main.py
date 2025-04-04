from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import txt2img

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(txt2img.router, prefix="/txt2img")
# app.include_router(img2img.router, prefix="/img2img")
# app.include_router(health_check.router, prefix="/health")

@app.get("/")
async def root():
    return {"message": "한국어 CLIP 기반 이미지 생성 서비스입니다!"}