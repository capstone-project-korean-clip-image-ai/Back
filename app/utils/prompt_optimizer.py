from typing import Tuple, List

# 프롬프트 강화 : 아직 미적용

# 1) 토큰 단위로 관리
POSITIVE_ENHANCERS = {
    "hanbok": [
        "고해상도", "선명한 얼굴", "성별에 맞는 한복", "균형 잡힌 얼굴", "아름다운 눈", "머리에 아무것도 없는",
        "정밀한 묘사", "자연스러운 옷 주름", "부드러운 명암 대비", "전통 의상", "또렷한 눈동자"
    ],
    "hansik": [
        "한국 음식"
    ],
    "hanok": [
        "고화질", 
    ],
}

NEGATIVE_ENHANCERS = {
    "hanbok": [
        "못생긴", "여자 한복을 입은 남자", "흐릿한", "저화질", "비현실적인", "얼굴 왜곡", "이상한 눈",
        "너무 많은 장신구", "과도한 액세서리", "장식이 많은 머리",
        "이상한 옷 구조", "왜곡된 카라", "부자연스러운 옷고름",
        "흐릿한 얼굴", "이상한 손가락", "깨진 옷 디테일", "기괴한 자세", "전통 머리 장식", "노이즈"
    ],
    "hansik": [
        "말려있는"
    ],
    "hanok": [
        "흐릿한 이미지", "저해상도",
    ],
}

FOOD_REPLACEMENTS = {
    "시금치나물": "시금치나물무침",
    "콩국수": "뽀얀_국물에_면이_담긴_시원한_콩국수",
    "미역국": "맑은_국물에_미역이_담긴_전통_한식_국물요리",
    "보쌈": "삶은_돼지고기와_채소를_함께_담은_한식_요리",
    "식혜": "맑고_달콤한_전통_한국_음료에_밥알이_띄워진",
    "삼겹살": "불판_위에_굽고_있는_노릇노릇한_삼겹살_구이",
    "송편": "솔잎_위에_찐_전통_송편_떡",
    "떡국": "하얀_국물에_떡이_떠있는_깔끔한_전통_한식",
}

def enhance_prompt(
    prompt: str,
    negative_prompt: str = None,
    loras: List[str] = None,
) -> Tuple[str, str, List[str]]:
    # 음식 키워드 치환
    for key, val in FOOD_REPLACEMENTS.items():
        if key in prompt:
            prompt = prompt.replace(key, val)

    # 2) 선택된 loras에 따른 강화 토큰 수집
    default_pos = ["최고 화질", "선명한 화질", "고해상도"]
    default_neg = ["저해상도", "흐릿한 이미지", "세부 묘사 부족"]
    loras = loras or []
    pos_list: List[str] = []
    neg_list: List[str] = []
    if not loras:
        pos_list = default_pos
        neg_list = default_neg
    else:
        for single in loras:
            pos_list.extend(POSITIVE_ENHANCERS.get(single, default_pos))
            neg_list.extend(NEGATIVE_ENHANCERS.get(single, default_neg))
    # 중복 제거
    pos_list = list(dict.fromkeys(pos_list))
    neg_list = list(dict.fromkeys(neg_list))

    prompt = f"{prompt}, {', '.join(pos_list)}"
    negative_prompt = (
        f"{negative_prompt}, {', '.join(neg_list)}"
        if negative_prompt
        else ", ".join(neg_list)
    )
    print(f"Enhanced prompt: {prompt}")
    print(f"Enhanced negative prompt: {negative_prompt}")
    return prompt, negative_prompt, loras
