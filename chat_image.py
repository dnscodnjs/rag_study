import os
import cv2
import torch
import clip
from PIL import Image
import time
import streamlit as st
import numpy as np

def imread_unicode(file_path):
    # 파일을 바이너리로 읽기
    stream = np.fromfile(file_path, dtype=np.uint8)
    # 바이너리 데이터를 이미지로 디코딩
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    return img

# Streamlit 페이지 설정 (앱 최상단에 위치해야 합니다.)
st.set_page_config(page_title="Image Similarity Finder", layout="wide")

# 실행 시작 시간 기록 (전체 처리 시간 측정용)
start_time = time.time()

# 사용할 디바이스 선택 (GPU 있으면 GPU 사용)
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 모델과 전처리 함수 로딩 (ViT-L/14 사용)
st.write("Loading CLIP model...")
model, preprocess = clip.load("ViT-L/14", device=device)

st.title("Image Similarity Finder")
st.write("타겟 이미지와 아이콘 이미지 간의 유사도를 계산합니다.")

# 타겟 이미지 업로드 (없으면 기본 이미지 사용)
uploaded_target = st.file_uploader("타겟 이미지를 업로드하세요 (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])
if uploaded_target is not None:
    target_image = Image.open(uploaded_target).convert("RGB")
else:
    st.warning("타겟 이미지가 업로드되지 않았습니다. 기본 타겟 이미지를 사용합니다.")
    target_image_path = "page165_icon4.jpeg"  # 기본 타겟 이미지 경로
    target_image = Image.open(target_image_path).convert("RGB")

st.image(target_image, caption="타겟 이미지", width=300)

# 타겟 이미지 전처리 및 임베딩 추출, 정규화
target_input = preprocess(target_image).unsqueeze(0).to(device)
with torch.no_grad():
    target_embedding = model.encode_image(target_input)
target_embedding /= target_embedding.norm(dim=-1, keepdim=True)

# 아이콘 이미지들이 저장된 폴더 (예: "./icons")
folder_path = "./icons/계기판_디스플레이"
if not os.path.exists(folder_path):
    st.error(f"아이콘 이미지 폴더 '{folder_path}'가 존재하지 않습니다.")
else:
    st.write("아이콘 이미지 유사도 검색 중...")
    results = []  # 각 항목: (파일 경로, 최대 유사도)

    # 폴더 내의 이미지 파일들을 순회
    for filename in os.listdir(folder_path):
        if not (filename.lower().endswith((".png", ".jpg", ".jpeg"))):
            continue
        image_path = os.path.join(folder_path, filename)
        img_cv = imread_unicode(image_path)
        if img_cv is None:
            continue
        # OpenCV를 이용해 그레이스케일 변환 및 Otsu 이진화 진행
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 외부 컨투어만 검출
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        icon_similarities = []
        # 검출된 각 컨투어(아이콘 후보)에 대해
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
                continue  # 너무 작은 영역은 제외

            # 후보 영역 잘라내기
            crop = img_cv[y:y+h, x:x+w]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            # 전처리 및 임베딩 추출
            crop_input = preprocess(crop_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                crop_embedding = model.encode_image(crop_input)
            crop_embedding /= crop_embedding.norm(dim=-1, keepdim=True)

            # 코사인 유사도 계산 (내적)
            similarity = (target_embedding @ crop_embedding.T).item()
            icon_similarities.append(similarity)

        # 해당 이미지에서 아이콘 후보 중 최대 유사도를 계산
        if icon_similarities:
            image_similarity = max(icon_similarities)
        else:
            image_similarity = 0  # 후보가 없으면 0
        results.append((image_path, image_similarity))

    # 유사도가 높은 순(내림차순)으로 정렬
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 검색 완료 시간
    end_time = time.time()
    st.write(f"유사도 검색 완료 (총 소요 시간: {end_time - start_time:.2f}초)")
    
    st.subheader("유사도가 높은 이미지")
    if results.__len__() == 0:
        st.write("유사한 아이콘 이미지가 없습니다.")
    else:
        st.image(results[0][0], caption=f"최대 아이콘 유사도: {results[0][1]:.4f}", width=300)
        file_path = results[0][0]
        file_name = os.path.basename(file_path)
        st.write(f"파일명: {file_name}")
