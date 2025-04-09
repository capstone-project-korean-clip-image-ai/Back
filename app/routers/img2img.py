from fastapi import APIRouter, UploadFile, File, Form
from app.services.img2img.object_detect import object_detect_process
from app.services.img2img.detected_image import detected_image

router = APIRouter()

@router.post("/object_detect")
async def image_object_detect(file: UploadFile, x: int = Form(...), y: int = Form(...)):
    # x: int = Form(...) <- FromData 안의 필드로 받음
    print(f"이미지 파일: {file.filename}")
    print(f"좌표: ({x}, {y})")
    image_path, objects = await object_detect_process(file, x, y)
    return image_path, objects

@router.get("/detected_image")
async def get_image(image_path: str):
    # 이미지 경로를 서비스 함수로 전달하여 처리
    response = await detected_image(image_path)
    return response


@router.get("")
async def ignore_get_generate():
    return {"message": "GET 요청은 지원되지 않습니다. POST 요청을 사용하세요."}