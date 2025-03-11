from openai import OpenAI  # OpenAI 라이브러리에서 OpenAI 클래스를 임포트
import streamlit as st  # Streamlit 라이브러리 임포트
import os  # 운영체제 관련 기능을 위한 os 모듈 임포트
from dotenv import load_dotenv  # .env 파일의 환경변수를 로드하기 위한 load_dotenv 함수 임포트

load_dotenv()  # .env 파일에 저장된 환경변수 로드

# OpenAI API 클라이언트 생성, 환경변수에서 OPENAI_API_KEY를 읽어옴
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

st.set_page_config(page_title="KCC Auto Manager 🚗", layout="centered")
st.title("KCC Auto Manager 🚗")
st.caption("자동차 사용 매뉴얼 및 서비스 센터 위치 찾기를 도와드립니다!")

# 세션 상태에 대화 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이미지 업로더 표시 여부 플래그 초기화
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

# 이전 대화 내용이 있다면 세션 상태에 저장된 메시지를 화면에 표시   
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ChatGPT에게 질문을 보내고 스트리밍 방식으로 응답 받는 함수 (대화 히스토리 포함)
def ask_chatgpt_stream(question):
    try:
        # 시스템 메시지와 이전 대화 내용을 포함한 전체 대화 이력을 구성
        conversation = [{"role": "system", "content": "너는 벤츠 S-class 사용 메뉴얼에 대해 전문가야."}]
        conversation.extend(st.session_state.messages)
        conversation.append({"role": "user", "content": question})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 사용할 모델 지정
            messages=conversation,
            stream=True  # 스트리밍 방식 활성화
        )

        answer_container = st.empty()
        full_response = ""
        # 스트리밍 응답 청크들을 순차적으로 처리
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                answer_container.markdown(full_response)
        return full_response

    except Exception as e:
        st.error(f"Error: {str(e)}")
    return ""

# 하단 영역: 텍스트 입력과 이미지 첨부 아이콘을 같은 행에 배치
col_text, col_attach = st.columns([4, 1])

with col_text:
    prompt = st.chat_input("메시지를 입력하세요")  # 채팅 입력칸

with col_attach:
    # 첨부 아이콘 버튼 (클릭 시 이미지 업로더 표시)
    if st.button("📎"):
        st.session_state.show_uploader = True

# 이미지 업로더가 활성화된 경우 표시
if st.session_state.show_uploader:
    uploaded_image = st.file_uploader("이미지 파일을 선택하세요", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="첨부된 이미지", use_column_width=True)
        # 업로드 후 벡터 DB 검색 로직 추가 가능
        st.session_state.show_uploader = False

# 텍스트 입력이 있는 경우 처리
if prompt:
    # 사용자 메시지를 세션 상태에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 어시스턴트의 응답 메시지 처리 (스트리밍)
    with st.chat_message("assistant"):
        response = ask_chatgpt_stream(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
