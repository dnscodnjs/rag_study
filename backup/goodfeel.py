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

# ==========================================
# 설정 및 초기화
# ==========================================
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="KCC Auto Manager 🚗", layout="wide")
st.title("KCC Auto Manager 🚗")
st.caption("자동차 사용 매뉴얼 및 서비스 센터 위치 찾기를 도와드립니다!")

# 대화 기록 초기화 및 클리어 버튼 추가
if "messages" not in st.session_state:
    st.session_state.messages = []

def clear_conversation():
    st.session_state.messages = []
    st.success("대화 기록이 초기화되었습니다.")

with st.sidebar:
    st.title("🤗💬 Auto Manager 🚗")
    st.subheader("현재 차량: 벤츠 S 클래스")
    st.markdown("### 대화 이력")
    for msg in st.session_state.messages[-5:]:
        role = "나" if msg["role"] == "user" else "챗봇"
        st.markdown(f"**{role}:** {msg['content'][:30]}...")
    st.button("대화 초기화", on_click=clear_conversation)
    st.markdown("### 설정")
    user_language = st.selectbox("사용 언어를 선택하세요", ("한국어", "English", "Deutsch"))
    st.markdown(f"선택된 언어: **{user_language}**")
    st.markdown("### 바로가기")
    st.markdown("[메르세데스-벤츠 공식 홈페이지](https://www.mercedes-benz.co.kr/)")

# ==========================================
# Pinecone 인덱스 초기화 및 데이터 인덱싱 함수
# ==========================================
def get_embedding(text):
    # 빈 텍스트 전달 시 에러를 방지
    if not text.strip():
        raise ValueError("임베딩에 전달할 텍스트가 비어있습니다.")
    response = openai.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding

def get_file_hash(filename):
    h = hashlib.sha256()
    with open(filename, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def load_stored_hash(hash_file):
    try:
        with open(hash_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def save_hash(hash_file, hash_value):
    with open(hash_file, "w", encoding="utf-8") as f:
        f.write(hash_value)

def index_data():
    hash_file = "test.json.hash"
    current_hash = get_file_hash("test.json")
    stored_hash = load_stored_hash(hash_file)
    
    if stored_hash != current_hash:
        with open("test.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
        vectors = []
        for item in test_data:
            pdf_file = item.get("pdf_file", "")
            structure = item.get("structure", [])
            for i, section in enumerate(structure):
                title = section.get("title", "")
                sub_titles = section.get("sub_titles", [])
                content_text = title
                image_paths = []
                for sub in sub_titles:
                    sub_title = sub.get("title", "")
                    contents = sub.get("contents", [])
                    for content in contents:
                        if content.lower().endswith(('.jpeg', '.jpg', '.png', '.gif')):
                            image_paths.append(content)
                    non_image_contents = [c for c in contents if not c.lower().endswith(('.jpeg', '.jpg', '.png', '.gif'))]
                    content_text += "\n" + sub_title + "\n" + "\n".join(non_image_contents)
                doc_id = f"{pdf_file}_{i}"
                embedding = get_embedding(content_text)
                metadata = {
                    "pdf_file": pdf_file,
                    "section_title": title,
                    "content": content_text,
                    "image_paths": image_paths  # image_paths는 리스트 형태로 저장됨
                }
                vectors.append((doc_id, embedding, metadata))
        index.upsert(vectors=vectors)
        save_hash(hash_file, current_hash)
        st.info("Pinecone에 새 데이터를 인덱싱했습니다.")
    else:
        st.info("test.json 파일에 변경이 없으므로, 기존 인덱스를 사용합니다.")

# Pinecone 인덱스 생성
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "kcc-new"
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

with st.spinner("데이터 인덱싱 중..."):
    index_data()

# ==========================================
# Pinecone 기반 컨텍스트 생성 함수
# ==========================================
def get_pinecone_context(query_text):
    if not query_text.strip():
        return "", None  # 빈 쿼리면 문맥과 이미지를 반환하지 않음

    query_embedding = get_embedding(query_text)
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    pinecone_context = ""
    displayed_image = None
    for match in results["matches"]:
        metadata = match["metadata"]
        pinecone_context += (
            f"Section: {metadata.get('section_title', '')}\n"
            f"Content: {metadata.get('content', '')}\n\n"
        )
        if metadata.get("image_paths"):
            # image_paths가 리스트라면 첫 번째 요소를 사용하고,
            # 문자열인 경우에도 "[]" 또는 빈 문자열이면 무시
            displayed_image = metadata["image_paths"]
            if isinstance(displayed_image, list):
                if displayed_image:
                    displayed_image = displayed_image[0]
                else:
                    displayed_image = None
            elif isinstance(displayed_image, str):
                if displayed_image.strip() in ("", "[]"):
                    displayed_image = None
    return pinecone_context, displayed_image

# ==========================================
# ChatGPT 스트리밍 응답 함수 (스피너 포함)
# ==========================================
def ask_chatgpt_stream(question, pinecone_context):
    try:
        system_message = (
            "너는 벤츠 S-class 사용 매뉴얼에 대해 전문가야. "
            "아래는 관련 매뉴얼 정보입니다:\n" + pinecone_context
        )
        conversation = [{"role": "system", "content": system_message}]
        conversation.extend(st.session_state.messages)
        conversation.append({"role": "user", "content": question})
        
        full_response = ""
        answer_container = st.empty()
        with st.spinner("챗봇 응답 중..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversation,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    answer_container.markdown(full_response)
        return full_response

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return ""

# ==========================================
# 사용자 입력 처리 (텍스트, 음성, 이미지)
# ==========================================
user_prompt = st.chat_input("메시지를 입력하세요")

with st.expander("음성 입력 및 이미지 첨부 열기"):
    audio_file = st.audio_input("음성을 녹음하세요.")
    uploaded_image = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg", "gif"])

if audio_file is not None:
    with st.spinner("음성 인식 중..."):
        transcript_result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ko"
        )
    user_prompt = transcript_result.text

# 텍스트나 이미지가 있을 때만 진행
if user_prompt or uploaded_image:
    combined_prompt = user_prompt or ""
    if uploaded_image:
        st.image(uploaded_image, width=150, caption="첨부된 이미지")
    st.session_state.messages.append({"role": "user", "content": combined_prompt})
    
    # 빈 텍스트인 경우 Pinecone 검색을 건너뜁니다.
    if combined_prompt.strip():
        pinecone_context, related_image = get_pinecone_context(combined_prompt)
    else:
        pinecone_context, related_image = "", None

    if related_image:
        st.image(related_image, caption="관련 이미지", use_container_width=True)
    
    with st.chat_message("assistant"):
        assistant_response = ask_chatgpt_stream(combined_prompt, pinecone_context)
        try:
            tts = gTTS(assistant_response, lang='ko')
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            st.audio(audio_fp, format="audio/mp3")
        except Exception as tts_error:
            st.error(f"TTS 에러: {tts_error}")
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# ==========================================
# 전체 대화 기록 출력 (최종 업데이트)
# ==========================================
st.markdown("## 대화 기록")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
