from fastapi.responses import FileResponse
from pathlib import Path

async def detected_image(image_path: str):
    try:
        full_path = Path(image_path)

        # 파일 존재 여부 확인
        if not full_path.exists():
            print(f"{full_path}_파일이 존재하지 않습니다.")
            return {"error": "파일이 존재하지 않습니다."}

        # 이미지 파일 응답
        return FileResponse(str(full_path), media_type="image/jpg")
    except Exception as e:
        return {"error": f"파일 처리 중 오류 발생: {str(e)}"}