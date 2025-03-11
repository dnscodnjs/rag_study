from openai import OpenAI  # OpenAI 라이브러리에서 OpenAI 클래스 임포트
import streamlit as st  # Streamlit 라이브러리 임포트
import os  # OS 모듈 임포트
from dotenv import load_dotenv  # .env 파일에서 환경변수를 불러오기 위한 모듈

load_dotenv()  # .env 파일에 저장된 환경변수 로드

# OpenAI API 클라이언트 생성 (환경변수에서 API 키 읽어옴)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

st.title("Simple chat")  # 앱 제목 표시

# 세션 상태에 대화 내역이 없으면 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 저장된 대화 내역을 화면에 출력 (세션이 유지되는 동안)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def ask_chatgpt_stream(question):
    """
    ChatGPT API를 스트리밍 모드로 호출하여 실시간 응답을 출력하는 함수
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 벤츠 S-class 사용 메뉴얼에 대해 전문가야."},  # 시스템 메시지
                {"role": "user", "content": question}  # 사용자 질문
            ],
            stream=True  # 스트리밍 모드 활성화
        )
        answer_container = st.empty()  # 응답 출력을 위한 빈 컨테이너 생성
        full_response = ""

        # 스트리밍으로 응답 청크를 받아 실시간 업데이트
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                answer_container.markdown(full_response)

        return full_response

    except Exception as e:
        st.error(f"Error: {str(e)}")
    return ""

# 채팅 입력과 이미지 첨부 아이콘을 옆에 배치하기 위해 컬럼 사용
col1, col2 = st.columns([4, 1])
with col1:
    prompt = st.chat_input("What is up?")
with col2:
    # 파일 업로더에 "📷" 아이콘 라벨을 사용하여 이미지 첨부 가능 (지원 확장자: png, jpg, jpeg, gif)
    uploaded_image = st.file_uploader("📷", type=["png", "jpg", "jpeg", "gif"], key="img_upload", label_visibility="visible")

if prompt:
    # 사용자 입력과 함께 이미지 첨부 여부 확인
    combined_prompt = prompt
    if uploaded_image is not None:
        # 이미지가 첨부된 경우, 이미지 파일 이름을 텍스트 프롬프트에 추가
        combined_prompt += "\n\n첨부된 이미지: " + uploaded_image.name

    # 사용자 메시지를 대화 내역에 추가
    st.session_state.messages.append({"role": "user", "content": combined_prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        # 이미지가 첨부되었다면 미리보기로 표시
        if uploaded_image is not None:
            st.image(uploaded_image, width=150, caption="첨부된 이미지")

    # Spinner를 답변이 완전히 끝날 때까지 표시
    with st.spinner("답변을 생성중입니다... 잠시만 기다려주세요."):
        with st.chat_message("assistant"):
            response = ask_chatgpt_stream(combined_prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
