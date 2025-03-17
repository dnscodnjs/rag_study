from openai import OpenAI  # OpenAI 라이브러리에서 OpenAI 클래스를 임포트
import streamlit as st  # Streamlit 라이브러리 임포트
import os  # 운영체제 관련 기능을 위한 os 모듈 임포트
import base64  # 이미지 인코딩을 위한 base64 모듈 임포트
from dotenv import load_dotenv  # .env 파일의 환경변수를 로드하기 위한 load_dotenv 함수 임포트
import io  # 메모리 내 바이너리 데이터 처리용
import speech_recognition as sr  # 음성 인식을 위한 SpeechRecognition 라이브러리
from streamlit_audiorecorder import audiorecorder  # 음성 녹음을 위한 커스텀 컴포넌트

load_dotenv()  # .env 파일에 저장된 환경변수 로드

# OpenAI API 클라이언트 생성, 환경변수에서 OPENAI_API_KEY를 읽어옴
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

st.title("KCC Auto Manager 🚗")
st.caption("자동차 사용 매뉴얼 및 서비스 센터 위치 찾기를 도와드립니다!")

with st.sidebar:
    st.title('🤗💬 Auto Manager 🚗')

# 세션 상태에 대화 기록이 없으면 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 내역이 있다면 화면에 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ChatGPT에게 질문을 보내고 스트리밍 방식으로 응답 받는 함수
def ask_chatgpt_stream(question):
    try:
        # 시스템 메시지와 이전 대화 기록(Conversation History)을 포함한 대화 구성
        conversation = [{"role": "system", "content": "너는 벤츠 S-class 사용 메뉴얼에 대해 전문가야."}]
        conversation.extend(st.session_state.messages)
        conversation.append({"role": "user", "content": question})
        
        # OpenAI의 ChatCompletion API 호출 (stream=True로 스트리밍 활성화)
        response = client.chat.completions.create(
            model="o3-mini",  # 사용할 모델 지정 (사용중인 모델에 맞게 변경)
            messages=conversation,
            stream=True  # 응답을 스트리밍 방식으로 받아옴
        )

        answer_container = st.empty()  # 실시간 응답 출력을 위한 빈 컨테이너 생성
        full_response = ""  # 전체 응답을 저장할 변수 초기화

        # 스트리밍 응답 청크들을 순차적으로 처리
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content  # 청크의 텍스트를 전체 응답에 추가
                answer_container.markdown(full_response)  # 실시간 업데이트하여 출력
        return full_response  # 완성된 응답 반환

    except Exception as e:
        st.error(f"Error: {str(e)}")  # 에러 발생 시 에러 메시지 출력
        return ""

# ===== 사용자 입력 영역 =====

# 1. 텍스트 입력
prompt_text = st.chat_input("메시지를 입력하세요")

# 2. 이미지 업로더
uploaded_image = st.file_uploader("이미지를 첨부하세요", type=["png", "jpg", "jpeg", "gif"])

# 3. 음성 녹음 (streamlit-audiorecorder 사용)
audio_bytes = audiorecorder("음성 녹음", "녹음 시작", "녹음 중...", "녹음 종료")
voice_text = ""
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    # 음성 인식을 위한 Recognizer 생성
    recognizer = sr.Recognizer()
    try:
        # audio_bytes를 메모리 내 파일 객체로 변환하여 음성 인식
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        # Google Speech Recognition API 사용 (한국어: 'ko-KR')
        voice_text = recognizer.recognize_google(audio_data, language='ko-KR')
        st.markdown(f"**음성 인식 결과:** {voice_text}")
    except Exception as e:
        st.error(f"음성 인식 오류: {str(e)}")

# ===== 입력 결합 및 처리 =====

# 텍스트와 음성 입력을 결합 (둘 다 존재할 경우)
combined_prompt = ""
if prompt_text:
    combined_prompt += prompt_text
if voice_text:
    if combined_prompt:
        combined_prompt += "\n\n음성 메시지: " + voice_text
    else:
        combined_prompt = voice_text

# 이미지가 첨부된 경우, 이미지 파일 이름(또는 데이터를 활용 가능)을 프롬프트에 추가
if uploaded_image is not None:
    # 이미지 인식이 아닌, 단순 미리보기 및 첨부 정보 전달 (추가 분석 시 base64 변환 가능)
    combined_prompt += "\n\n첨부된 이미지: " + uploaded_image.name

if combined_prompt:
    # 사용자 메시지를 대화 기록에 추가
    st.session_state.messages.append({"role": "user", "content": combined_prompt})
    
    # 사용자 메시지를 화면에 표시 (텍스트, 음성 인식 결과, 이미지 미리보기)
    with st.chat_message("user"):
        st.markdown(combined_prompt)
        if uploaded_image is not None:
            st.image(uploaded_image, width=150, caption="첨부된 이미지")
    
    # 어시스턴트의 응답 메시지를 스트리밍 방식으로 받아 출력
    with st.chat_message("assistant"):
        response = ask_chatgpt_stream(combined_prompt)
    
    # 어시스턴트 응답을 대화 기록에 추가
    st.session_state.messages.append({"role": "assistant", "content": response})
