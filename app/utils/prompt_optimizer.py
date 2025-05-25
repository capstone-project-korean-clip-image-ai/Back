from typing import Tuple

# 프롬프트 강화 : 아직 미적용

hanbok_enhancers = ["""초고해상도, 정밀한 묘사, 영화 같은 조명, 섬세한 디테일, 자연스러운 옷 주름, 부드러운 명암 대비, 섬세한 자수, 전통 의상
선명한 얼굴, 자연스러운 피부 질감, 또렷한 눈동자, 균형 잡힌 얼굴, 아름다운 눈, 고해상도 피부, 머리에 아무것도 없는, 한복
""", """못생긴, 흐릿한, 저화질, 비현실적인, 얼굴 왜곡, 이상한 눈, 너무 많은 장신구, 과도한 액세서리, 장식이 많은 머리, 
이상한 옷 구조, 왜곡된 카라, 부자연스러운 옷고름, 흐릿한 얼굴, 이상한 손가락, 
비현실적인 옷 질감, 깨진 옷 디테일, 기괴한 자세, 전통 머리 장식, 머리 장신구, 화려한 머리 장식, 머리 장식, 머리에 무언가 있는
"""]

hansik_enhancers = ["""음식 사진 스타일, 선명한 디테일, 자연광 조명, 선명한 초점, 질감이 뚜렷한 재료, 실사풍 질감,
풍부한 색감, 음식 중심 구도""", """부자연스러운 색감, 비현실적인 재료, 어색한 조명, 
이상한 그릇 모양, 깨진 재료 텍스처, 왜곡된 음식 형상, 합성처럼 보이는 장면, 
노이즈, 비현실적인 반사광, 부자연스러운 그림자, 뒤틀린 구도, 부정확한 비율
"""]

hanok_enhancers = ["""고화질, 선명한 질감, 자연광, 
사실적인 재질 표현, 전통 건축의 정교한 문양, 조화로운 색상 배치, 부드러운 명암 대비, 풍부한 동적 범위, 시네마틱 구성
""", """기형적 형태, 왜곡된 원근감, 어색한 그림자, 부자연스러운 색감,
비현실적인 조명, 텍스처 깨짐, 합성처럼 보이는 표현, 왜곡된 구조물, 불분명한 배경, 노이즈
"""]

def enhance_prompt(
    prompt: str, 
    negative_prompt: str = None, 
    lora: str = None,
) -> Tuple[str, str, str]:
    
    if lora is "hanbok":
        prompt = f"{prompt}, {hanbok_enhancers[0]}"
        negative_prompt = f"{negative_prompt}, {hanbok_enhancers[1]}" if negative_prompt else hanbok_enhancers[1]
    elif lora is "hansik":
        prompt = f"{prompt}, {hansik_enhancers[0]}"
        negative_prompt = f"{negative_prompt}, {hansik_enhancers[1]}" if negative_prompt else hansik_enhancers[1]
    elif lora is "hanok":
        prompt = f"{prompt}, {hanok_enhancers[0]}"
        negative_prompt = f"{negative_prompt}, {hanok_enhancers[1]}" if negative_prompt else hanok_enhancers[1]
    else:
        # 기본 강화
        prompt = f"{prompt}, 최고 화질, 선명한 화질, 고해상도"
        negative_prompt = f"{negative_prompt}, 저해상도, 흐릿한 이미지, 세부 묘사 부족" if negative_prompt else "저해상도, 흐릿한 이미지, 세부 묘사 부족"
    
    return prompt, negative_prompt, lora

# 테스트 코드
if __name__ == "__main__":
    test_prompt = "a woman in red dress walking in the city"
    
    # 기본 강화
    enhanced, neg, lora = enhance_prompt(test_prompt)
    print(f"원본: {test_prompt}")
    print(f"강화됨: {enhanced}")
    print("-" * 50)