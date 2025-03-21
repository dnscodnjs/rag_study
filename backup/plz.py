from openai import OpenAI  # OpenAI 라이브러리에서 OpenAI 클래스를 임포트
import openai  # 최신 OpenAI 인터페이스 사용
import streamlit as st  # Streamlit 라이브러리 임포트
import os  # 운영체제 관련 기능을 위한 os 모듈 임포트
import base64  # 이미지 인코딩을 위한 base64 모듈 임포트
import json  # JSON 파일 처리를 위한 json 모듈 임포트
from dotenv import load_dotenv  # .env 파일의 환경변수를 로드하기 위한 함수 임포트
from pinecone import Pinecone, ServerlessSpec  # 최신 Pinecone API 사용
import hashlib  # 파일 해시 계산을 위한 모듈
import tempfile
import io  # 메모리 내 파일 객체 생성을 위한 모듈
from st_audiorec import st_audiorec
from gtts import gTTS  # 텍스트를 음성으로 변환하기 위한 gTTS 라이브러리

# .env 파일에 저장된 환경변수 로드
load_dotenv()

# OpenAI API 클라이언트 생성 및 API 키 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# 메인 타이틀 및 캡션 설정
st.title("KCC Auto Manager 🚗")
st.caption("자동차 사용 매뉴얼 및 서비스 센터 위치 찾기를 도와드립니다!")

if "recording" not in st.session_state:
    st.session_state.recording = False

toggle_label = "음성 인식 중지" if st.session_state.recording else "음성 인식 시작"

##########################################
# 사이드바 디자인: 현재 차량, 대화 이력 등
##########################################
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
# Pinecone 초기화 및 인덱스 생성
##########################################
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "kcc-new"
# text-embedding-ada-002 모델의 임베딩 차원은 1536입니다.
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

##########################################
# OpenAI 임베딩 함수 정의 (text-embedding-ada-002 사용)
##########################################
def get_embedding(text):
    response = openai.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding

##########################################
# test.json 파일을 Pinecone에 인덱싱 (변경된 경우에만)
##########################################
def get_file_hash(filename):
    """파일의 SHA-256 해시값을 계산합니다."""
    h = hashlib.sha256()
    with open(filename, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def load_stored_hash(hash_file):
    """저장된 해시값을 읽어옵니다."""
    try:
        with open(hash_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def save_hash(hash_file, hash_value):
    """해시값을 파일에 저장합니다."""
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
            content_text = title  # 섹션 텍스트 시작
            image_paths = []      # 이미지 경로 저장 리스트
            for sub in sub_titles:
                sub_title = sub.get("title", "")
                contents = sub.get("contents", [])
                # 이미지 경로 추출: 파일명이 .jpeg, .jpg, .png, .gif로 끝나는 항목
                for content in contents:
                    if content.lower().endswith(('.jpeg', '.jpg', '.png', '.gif')):
                        image_paths.append(content)
                # 일반 텍스트 내용 결합 (이미지 경로 제외)
                non_image_contents = [c for c in contents if not c.lower().endswith(('.jpeg', '.jpg', '.png', '.gif'))]
                content_text += "\n" + sub_title + "\n" + "\n".join(non_image_contents)
            doc_id = f"{pdf_file}_{i}"
            embedding = get_embedding(content_text)
            metadata = {
                "pdf_file": pdf_file,
                "section_title": title,
                "content": content_text,
                "image_paths": image_paths  # 추출한 이미지 경로 저장
            }
            vectors.append((doc_id, embedding, metadata))
    index.upsert(vectors=vectors)
    save_hash(hash_file, current_hash)
    st.write("Pinecone에 새 데이터를 인덱싱했습니다.")
else:
    st.write("test.json 파일에 변경이 없으므로, 기존 인덱스를 사용합니다.")

##########################################
# 대화 기록 초기화 및 기존 대화 출력
##########################################
if "messages" not in st.session_state:
    st.session_state.messages = []

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
# 사용자 입력 처리 (텍스트 입력, 이미지 업로더, 음성 녹음)
##########################################
prompt = st.chat_input("메시지를 입력하세요")
uploaded_image = st.file_uploader("이미지를 첨부하세요", type=["png", "jpg", "jpeg", "gif"])

# 음성 녹음 토글 버튼
if st.button(toggle_label):
    st.session_state.recording = not st.session_state.recording

# 녹음 상태가 활성화되면 st_audiorec 컴포넌트를 보여줌
if st.session_state.recording:
    st.info("녹음 중입니다. 다시 버튼을 누르면 녹음이 종료됩니다.")
    # st_audiorec는 녹음이 완료되면 audio_bytes (wav 형식의 bytes)를 반환합니다.
    audio_bytes = st_audiorec()
    
    # 녹음이 완료되어 audio_bytes가 반환되면 처리
    if audio_bytes is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp_path = tmp.name
        
        # Whisper API를 통해 오디오 파일 전사 (텍스트 변환)
        with open(tmp_path, "rb") as audio_file:
            transcript_result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        os.remove(tmp_path)
        
        transcript = transcript_result.text
        st.success("음성 인식 결과:")
        st.write(transcript)
        # 음성 전사 결과를 세션에 저장
        st.session_state.voice_transcript = transcript
        st.session_state.messages.append({"role": "user", "content": transcript})
        with st.chat_message("user"):
            st.markdown(transcript)
        st.session_state.recording = False

# 텍스트 입력, 이미지 업로드, 또는 음성 전사된 텍스트가 있을 경우 처리
if prompt or uploaded_image or st.session_state.get("voice_transcript"):
    # 음성 전사가 있다면 우선 사용, 없으면 prompt 사용
    combined_prompt = prompt if prompt else st.session_state.get("voice_transcript", "")
    
    # (1) 실제로 combined_prompt가 비어있지 않을 때만 처리
    if combined_prompt.strip():
        # 이미지 업로드 시 (옵션)
        if uploaded_image is not None:
            image_bytes = uploaded_image.read()
            image_type = uploaded_image.type
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            # 필요 시 이미지 데이터 URL을 combined_prompt에 추가 가능:
            # combined_prompt += f"\n\n첨부된 이미지: data:{image_type};base64,{image_base64}"
        
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": combined_prompt})
        with st.chat_message("user"):
            # 텍스트 표시
            if prompt:
                st.markdown(prompt)
            # 이미지 표시
            if uploaded_image is not None:
                st.image(uploaded_image, width=150, caption="첨부된 이미지")
        
        ##########################################
        # Pinecone DB 검색
        ##########################################
        query_embedding = get_embedding(combined_prompt)
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        pinecone_context = ""
        displayed_image = None  # 관련 이미지를 저장할 변수
        for match in results["matches"]:
            metadata = match["metadata"]
            pinecone_context += (
                f"Section: {metadata.get('section_title', '')}\n"
                f"Content: {metadata.get('content', '')}\n\n"
            )
            if "image_paths" in metadata and metadata["image_paths"]:
                displayed_image = metadata["image_paths"]
    
              # 리스트가 비어있지 않다면 첫 번째 이미지 선택
            if isinstance(displayed_image, list) and len(displayed_image) > 0:
                  displayed_image = displayed_image[0]  # 첫 번째 이미지 경로 사용
            else:
               displayed_image = None

            if displayed_image and isinstance(displayed_image, str):
              st.image(displayed_image, caption="관련 이미지", use_container_width=True)

        
        ##########################################
        # ChatGPT에 질문 전송 및 응답 처리
        ##########################################
        with st.chat_message("assistant"):
            response = ask_chatgpt_stream(combined_prompt, pinecone_context)
            # 음성로 답변을 듣기 위한 TTS 변환 추가
            if response.strip():
                try:
                    tts = gTTS(response, lang='ko')
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)
                    st.audio(audio_fp, format="audio/mp3")
                except Exception as tts_error:
                    st.error(f"TTS 에러: {tts_error}")
            #if displayed_image:
                #st.image(displayed_image, caption="관련 이미지", use_container_width=True)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # (2) 전송 후 음성 전사 내용 초기화
    if st.session_state.get("voice_transcript"):
        del st.session_state["voice_transcript"]
