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
import ast  # 문자열로 된 리스트 평가를 위한 모듈

# .env 파일에 저장된 환경변수 로드
load_dotenv()

# OpenAI API 클라이언트 생성 및 API 키 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# 세션 스테이트 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "toggle_expander" not in st.session_state:
    st.session_state.toggle_expander = False

##########################################
# 메인 화면 및 사이드바
##########################################
st.title("KCC Auto Manager 🚗")
st.caption("자동차 사용 매뉴얼 및 서비스 센터 위치 찾기를 도와드립니다!")

with st.sidebar:
    st.title("🤗💬 Auto Manager 🚗")
    st.subheader("현재 차량: 벤츠 S 클래스")
    st.markdown("### 대화 이력")
    st.markdown("- 1번 대화 (2023-01-01)")
    st.markdown("- 2번 대화 (2023-01-02)")
    st.markdown("- 3번 대화 (2023-01-03)")
    st.markdown("### 설정")
    user_language = st.selectbox("사용 언어를 선택하세요", ("한국어", "English", "Deutsch"))
    st.markdown(f"선택된 언어: **{user_language}**")
    st.markdown("### 바로가기")
    st.markdown("[메르세데스-벤츠 공식 홈페이지](https://www.mercedes-benz.co.kr/)")

##########################################
# Pinecone 초기화 및 인덱싱
##########################################
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

##########################################
# 임베딩 함수 정의 (text-embedding-ada-002 사용)
##########################################
def get_embedding(text):
    response = openai.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding

##########################################
# test.json 해시 비교 후 Pinecone에 인덱싱
##########################################
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
                "image_paths": image_paths
            }
            vectors.append((doc_id, embedding, metadata))
    index.upsert(vectors=vectors)
    save_hash(hash_file, current_hash)
    st.write("Pinecone에 새 데이터를 인덱싱했습니다.")
else:
    st.write("test.json 파일에 변경이 없으므로, 기존 인덱스를 사용합니다.")

##########################################
# 기존 대화 기록 출력
##########################################
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

##########################################
# ChatGPT 응답 함수 (스트리밍 방식)
##########################################
def ask_chatgpt_stream(question, pinecone_context):
    try:
        system_message = (
            "너는 벤츠 S-class 사용 매뉴얼에 대해 전문가야. "
            "아래는 관련 매뉴얼 정보입니다:\n" + pinecone_context
        )
        conversation = [{"role": "system", "content": system_message}]
        conversation.extend(st.session_state.messages)
        conversation.append({"role": "user", "content": question})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
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

##########################################
# 사용자 입력 UI (텍스트 입력과 토글 버튼)
##########################################
col1, col2 = st.columns([0.9, 0.1])
with col1:
    prompt = st.chat_input("메시지를 입력하세요") or ""
with col2:
    if st.button("🔗"):
        st.session_state.toggle_expander = not st.session_state.toggle_expander

# 토글 상태에 따라 음성 입력 및 이미지 업로드 영역 표시
if st.session_state.toggle_expander:
    st.markdown("**음성 입력 및 이미지 첨부**")
    audio_file = st.audio_input("음성을 녹음하세요.")
    uploaded_image = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg", "gif"])
else:
    audio_file = None
    uploaded_image = None

##########################################
# 사용자 입력 처리 및 응답
##########################################
# 음성 파일이 있으면 Whisper API로 텍스트 변환
if audio_file is not None:
    transcript_result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="ko"
    )
    prompt = transcript_result.text

if prompt or uploaded_image:
    combined_prompt = prompt or ""
    # 사용자 메시지 출력
    st.chat_message("user").markdown(combined_prompt)
    if uploaded_image is not None:
        st.image(uploaded_image, width=150, caption="첨부된 이미지")
    
    # Pinecone DB 검색
    query_embedding = get_embedding(combined_prompt)
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    pinecone_context = ""
    displayed_image = None
    for match in results["matches"]:
        metadata = match["metadata"]
        pinecone_context += (
            f"Section: {metadata.get('section_title', '')}\n"
            f"Content: {metadata.get('content', '')}\n\n"
        )
        # metadata의 image_paths가 문자열이라면 리스트로 변환 후 사용
        if not displayed_image and "image_paths" in metadata and metadata["image_paths"]:
            image_paths = metadata["image_paths"]
            if isinstance(image_paths, str):
                if image_paths.strip().startswith('['):
                    try:
                        image_paths = ast.literal_eval(image_paths)
                    except Exception as e:
                        st.error("이미지 데이터 파싱 오류: " + str(e))
                        image_paths = []
            if isinstance(image_paths, list) and len(image_paths) > 0:
                displayed_image = image_paths[0]
    
    # ChatGPT 호출 및 응답 처리
    response = ask_chatgpt_stream(combined_prompt, pinecone_context)
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
        if displayed_image:
            st.image(displayed_image, caption="관련 이미지", use_container_width=True)
    
    # 대화 기록에 어시스턴트 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
