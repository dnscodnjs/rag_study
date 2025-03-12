from openai import OpenAI  # OpenAI 라이브러리에서 OpenAI 클래스를 임포트
import openai  # 최신 OpenAI 인터페이스 사용
import streamlit as st  # Streamlit 라이브러리 임포트
import os  # 운영체제 관련 기능을 위한 os 모듈 임포트
import base64  # 이미지 인코딩을 위한 base64 모듈 임포트
import json  # JSON 파일 처리를 위한 json 모듈 임포트
from dotenv import load_dotenv  # .env 파일의 환경변수를 로드하기 위한 함수 임포트
from pinecone import Pinecone, ServerlessSpec  # 최신 Pinecone API 사용
import hashlib  # 파일 해시 계산을 위한 모듈
import io  # 메모리 내 파일 객체 생성을 위한 모듈
from gtts import gTTS  # 텍스트를 음성으로 변환하기 위한 gTTS 라이브러리

# .env 파일에 저장된 환경변수 로드
load_dotenv()

# OpenAI API 클라이언트 생성 및 API 키 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# 메인 타이틀 및 캡션 설정
st.title("KCC Auto Manager 🚗")
st.caption("자동차 사용 매뉴얼 및 서비스 센터 위치 찾기를 도와드립니다!")

# 사용자 입력 필드 및 버튼 배치
toggle_expander = st.session_state.get("toggle_expander", False)
col1, col2 = st.columns([0.9, 0.1])
with col1:
    prompt = st.chat_input("메시지를 입력하세요")
with col2:
    if st.button("🔗"):
        st.session_state.toggle_expander = not toggle_expander

# 음성 입력 및 이미지 업로드 영역
toggle_expander = st.session_state.get("toggle_expander", False)
if toggle_expander:
    st.markdown("**음성 입력 및 이미지 첨부**")
    audio_file = st.audio_input("음성을 녹음하세요.")
    uploaded_image = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg", "gif"])
else:
    audio_file = None
    uploaded_image = None

##########################################
# ChatGPT 응답 함수 (스트리밍 방식)
##########################################
def ask_chatgpt_stream(question):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}],
            stream=True
        )
        answer_container = st.empty()
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                answer_container.markdown(full_response)
        return full_response
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return ""

##########################################
# 사용자 입력 처리
##########################################
if audio_file is not None:
    transcript_result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="ko"
    )
    prompt = transcript_result.text

if prompt or uploaded_image:
    combined_prompt = prompt or ""
    st.chat_message("user").markdown(combined_prompt)
    
    if uploaded_image is not None:
        st.image(uploaded_image, width=150, caption="첨부된 이미지")
    
    response = ask_chatgpt_stream(combined_prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response)
        
        try:
            tts = gTTS(response, lang='ko')
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            st.audio(audio_fp, format="audio/mp3")
        except Exception as tts_error:
            st.error(f"TTS 에러: {tts_error}")
