from openai import OpenAI  # OpenAI 라이브러리에서 OpenAI 클래스를 임포트
import streamlit as st
import os
import base64
from dotenv import load_dotenv
import speech_recognition as sr
import io

# 음성 녹음 컴포넌트에서 audiorecorder 함수만 임포트
from streamlit_audiorecorder import audiorecorder

load_dotenv()  # .env 파일에 저장된 환경변수 로드

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

st.title("KCC Auto Manager 🚗")
st.caption("자동차 사용 매뉴얼 및 서비스 센터 위치 찾기를 도와드립니다!")

with st.sidebar:
    st.title('🤗💬 Auto Manager 🚗')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def ask_chatgpt_stream(question):
    try:
        conversation = [{"role": "system", "content": "너는 벤츠 S-class 사용 메뉴얼에 대해 전문가야."}]
        conversation.extend(st.session_state.messages)
        conversation.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model="o3-mini",
            messages=conversation,
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


st.subheader("음성 입력")
st.caption("녹음 버튼을 눌러 음성 입력 후 인식 결과를 확인할 수 있습니다.")
audio_bytes = audiorecorder()  # 음성 녹음 컴포넌트 호출

voice_input = ""
if audio_bytes is not None:
    audio_file = io.BytesIO(audio_bytes)
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
        voice_input = r.recognize_google(audio_data, language='ko-KR')
        st.success("음성 인식 결과:")
        st.write(voice_input)
    except sr.UnknownValueError:
        st.error("음성을 인식할 수 없습니다.")
    except sr.RequestError as e:
        st.error(f"음성 인식 서비스 요청 에러: {e}")

st.subheader("텍스트 입력")
prompt = st.chat_input("메시지를 입력하세요")
uploaded_image = st.file_uploader("이미지를 첨부하세요", type=["png", "jpg", "jpeg", "gif"])

combined_prompt = ""
if prompt:
    combined_prompt += prompt
if voice_input:
    if combined_prompt:
        combined_prompt += "\n[음성 입력]: " + voice_input
    else:
        combined_prompt = voice_input

if combined_prompt or uploaded_image:
    st.session_state.messages.append({"role": "user", "content": combined_prompt})

    with st.chat_message("user"):
        if prompt:
            st.markdown(prompt)
        if voice_input:
            st.markdown(f"**[음성 입력]:** {voice_input}")
        if uploaded_image is not None:
            st.image(uploaded_image, width=150, caption="첨부된 이미지")

    with st.chat_message("assistant"):
        response = ask_chatgpt_stream(combined_prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
