from fastapi import APIRouter, UploadFile, File
from app.services.img2img.object_detect import object_detect_process
from app.services.img2img.get_obj_dect_img import get_obj_dect_img_process

router = APIRouter()

@router.post("/object_detect")
async def image_object_detect(file: UploadFile = File(...)):
    image_path, objects = await object_detect_process(file)
    return image_path, objects

@router.get("/get_obj_dect_img")
async def get_image(image_path: str):
    # 이미지 경로를 서비스 함수로 전달하여 처리
    response = await get_obj_dect_img_process(image_path)
    return response


@router.get("")
async def ignore_get_generate():
    return {"message": "GET 요청은 지원되지 않습니다. POST 요청을 사용하세요."}