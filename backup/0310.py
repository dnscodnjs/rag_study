from openai import OpenAI  # OpenAI 라이브러리에서 OpenAI 클래스를 임포트
import streamlit as st  # Streamlit 라이브러리 임포트
import os  # 운영체제 관련 기능을 위한 os 모듈 임포트
import base64  # 이미지 인코딩을 위한 base64 모듈 임포트
from dotenv import load_dotenv  # .env 파일의 환경변수를 로드하기 위한 load_dotenv 함수 임포트

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

# 사용자 입력을 받음 (채팅 입력 창) 및 이미지 업로더 추가
prompt = st.chat_input("메시지를 입력하세요")
uploaded_image = st.file_uploader("이미지를 첨부하세요", type=["png", "jpg", "jpeg", "gif"])

# 텍스트나 이미지 입력이 있을 때 처리
if prompt or uploaded_image:
    # 기본 텍스트 입력을 combined_prompt로 사용
    combined_prompt = prompt if prompt else ""
    # 이미지가 첨부된 경우, 파일 데이터를 읽어 base64로 인코딩한 후 텍스트에 포함
    if uploaded_image is not None:
        image_bytes = uploaded_image.read()
        image_type = uploaded_image.type  # 예: image/jpeg
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        # 이미지 인식 대신, 이미지 데이터 URL을 텍스트에 추가합니다.
        #combined_prompt += f"\n\n첨부된 이미지: data:{image_type};base64,{image_base64}"

    # 사용자 메시지를 대화 기록에 추가
    st.session_state.messages.append({"role": "user", "content": combined_prompt})
    
    # 사용자 메시지를 화면에 표시 (텍스트와 함께 이미지 미리보기도 표시)
    with st.chat_message("user"):
        if prompt:
            st.markdown(prompt)
        if uploaded_image is not None:
            # 이미지 파일을 미리보기로 표시
            st.image(uploaded_image, width=150, caption="첨부된 이미지")
    
    # 어시스턴트의 응답 메시지를 스트리밍 방식으로 받아 출력
    with st.chat_message("assistant"):
        response = ask_chatgpt_stream(combined_prompt)
    
    # 어시스턴트 응답도 대화 기록에 추가
    st.session_state.messages.append({"role": "assistant", "content": response})
