from openai import OpenAI  # OpenAI 라이브러리에서 OpenAI 클래스를 임포트
import openai  # 최신 OpenAI 인터페이스 사용
import streamlit as st  # Streamlit 라이브러리 임포트
import os  # 운영체제 관련 기능을 위한 os 모듈 임포트
import base64  # 이미지 인코딩을 위한 base64 모듈 임포트
import json  # JSON 파일 처리를 위한 json 모듈 임포트
from dotenv import load_dotenv  # .env 파일의 환경변수를 로드하기 위한 함수 임포트
import hashlib  # 파일 해시 계산을 위한 모듈
import io  # 메모리 내 파일 객체 생성을 위한 모듈
from gtts import gTTS  # 텍스트를 음성으로 변환하기 위한 gTTS 라이브러리
import re

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from pinecone import Pinecone  # 최신 Pinecone API 사용
from typing import Literal
from typing_extensions import List

# ==========================================
# 설정 및 초기화
# ==========================================
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

index_name = "kcc"
car_type = "EQS"
IMAGE_PATTERN = r'(https?://\S+\.(?:png|jpg|jpeg|gif))'

embedding = OpenAIEmbeddings(model="text-embedding-3-large")  # 예: text-embedding-ada-002
database = PineconeVectorStore(index_name=index_name, embedding=embedding)
retriever = database.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model='gpt-3.5-turbo')

st.set_page_config(page_title="KCC Auto Manager", layout="wide")
st.title("KCC Auto Manager")

st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        .responsive-image {
            width: 85% !important;
        }
    }
    @media (min-width: 769px) {
        .responsive-image {
            width: 50% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 기본 헤더, 푸터, 메뉴 숨기기
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;} /* 좌측 상단 햄버거 메뉴 */
    header {visibility: hidden;}    /* 상단 헤더 */
    footer {visibility: hidden;}    /* 하단 푸터 */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# session_state 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사이드바
with st.sidebar:
    st.title("🤗💬 Auto Manager 🚗")
    st.subheader("현재 차량: 벤츠 S 클래스")
    st.markdown("### 설정")
    user_language = st.selectbox("사용 언어를 선택하세요", ("한국어", "English", "Deutsch"))
    st.markdown(f"선택된 언어: **{user_language}**")
    st.markdown("### 바로가기")
    st.markdown("[메르세데스-벤츠 공식 홈페이지](https://www.mercedes-benz.co.kr/)")
    st.markdown("[메르세데스-벤츠 공식 사용 메뉴얼](https://www.mercedes-benz.co.kr/passengercars/services/manuals.html)")

# ==========================================
# Pinecone 인덱스 초기화 및 데이터 인덱싱 함수
# ==========================================
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
        json_file = "test.json"
        with open(json_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

            documents = []
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

                    metadata = {
                        "pdf_file": pdf_file,
                        "section_title": title,
                        "image_paths": json.dumps(image_paths)
                    }
                    documents.append(Document(page_content=content_text, metadata=metadata))
            database.add_documents(documents)
        save_hash(hash_file, current_hash)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# 인덱스 생성/확인
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
    )
index = pc.Index(index_name)

with st.spinner("데이터 인덱싱 중..."):
    index_data()

# ==========================================
# LangGraph 정의
# ==========================================
@tool
def vector_retrieve_tool(query: str) -> List[Document]:
    """Retrieve documents based on the given query."""
    return retriever.invoke(query)

def retrieve_search_node(state: MessagesState) -> Command:
    retrieve_search_agent = create_react_agent(
        llm, 
        tools=[vector_retrieve_tool],
        state_modifier = (
            "You are an expert on Mercedes Benz " + car_type + " car manuals."
            "Please consider the information you provided and reply. Please provide facts, not opinions. Please translate the answers into Korean and print them out."
            "Please refer to the structure below to organize the website information:\n\n"
            "{{제목}}\n"
            "Example: Mercedes Benz EQS: Driver Display Charge Status Window Function\n\n"
            "{{주요 정보}}\n"
            "- Feature Summary and Key Points\n\n"
            "{{상세 설명}}\n"
            "- Detailed description of features\n\n"
        )
    )
    result = retrieve_search_agent.invoke(state)
    state["retrieve_result"] = result['messages'][-1].content

    # 사용자 쿼리를 state에서 추출 (첫 번째 메시지가 HumanMessage라고 가정)
    user_query = state["messages"][0].content
    docs = retriever.invoke(user_query)
    
    displayed_image = None
    for doc in docs:
        meta = doc.metadata
        if "image_paths" in meta and meta["image_paths"]:
            try:
                image_paths = json.loads(meta["image_paths"])
            except Exception:
                image_paths = meta["image_paths"]
            if isinstance(image_paths, list) and len(image_paths) > 0:
                displayed_image = image_paths[0]
                break

    # retrieval 결과 앞에 이미지 링크(존재할 경우) 추가
    state["retrieve_result"] = (displayed_image or "") + '\n\n' + state["retrieve_result"] + "\n"
    
    state.setdefault("messages", []).append(
        HumanMessage(content=state["retrieve_result"], name="retrieve_search")
    )
    return Command(update={'messages': state["messages"]})


def evaluate_node(state: MessagesState) -> Command[Literal['web_search', END]]:
    if state["messages"][-1].content is None:
        st.write("답변이 충분하지 않습니다. 웹 검색을 시도합니다.")
        return Command(goto='web_search')

    # retrieval 결과가 충분할 수 있으므로 LLM 평가 프롬프트 실행
    eval_prompt = PromptTemplate.from_template(
        "You are an expert on Mercedes Benz " + car_type + " car manuals."
        "Please rate if the retrive results below provide sufficient answers."
        "You must assess that the answers or answers of more than 200 characters are sufficiently consistent with your question."
        "If you don't have enough information to judge, or if you don't provide it in your answer, answer 'yes', and answer no if enough is enough.\n\n"
        "Retrieve Results:\n{result}"
    )
    eval_chain = eval_prompt | llm
    evaluation = eval_chain.invoke({"result": state.get("retrieve_result")})
    if "yes" in evaluation.content.lower():
          st.write("답변이 충분하지 않습니다. 웹 검색을 시도합니다.")
          return Command(goto='web_search')
    else:
          return Command(goto=END)


def web_search_node(state: MessagesState) -> Command:
    tavily_search_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        include_images=True,
    )

    web_search_agent = create_react_agent(
        llm, 
        tools=[tavily_search_tool],
        state_modifier = (
            "You are an expert on Mercedes Benz " + car_type + " car manuals."
            "Please reply to the website information in detail. Please translate the answer into Korean and print it out."
            "Please refer to the structure below to organize the website information:\n\n"
            "## {{제목}}\n"
            "Example: Mercedes Benz EQS: Driver Display Charge Status Window Function\n\n"
            "### {{주요 정보}}\n"
            "- Feature Summary and Key Points\n\n"
            "### {{상세 설명}}\n"
            "- Detailed description of the feature. / {{출처}}: real website link\n\n"
            "Please translate the answer into Korean, separate each item into separate lines and print it out in a good way."
        )
    )

    result = web_search_agent.invoke(state)
    state["web_result"] = result['messages'][-1].content
    state.setdefault("messages", []).append(
        HumanMessage(content=state["web_result"], name="web_search")
    )
    return Command(update={'messages': state["messages"]})


# 그래프 구성
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("retrieve_search", retrieve_search_node)
graph_builder.add_node("evaluate", evaluate_node)
graph_builder.add_node("web_search", web_search_node)

graph_builder.add_edge(START, "retrieve_search")
graph_builder.add_edge("retrieve_search", "evaluate")
graph_builder.add_edge("web_search", END)

graph = graph_builder.compile()

# ==========================================
# LangGraph 기반 Agent 실행 함수
# ==========================================
def ask_lang_graph_agent(query):
    return graph.invoke({"messages": [("user", query)]})

# ==========================================
# 사용자 입력 및 처리
# ==========================================
# st.chat_input에 별도 key를 두지 않고, 매번 동일 key로 사용
user_prompt = st.chat_input("메시지를 입력하세요")

with st.expander("음성 입력 및 이미지 첨부 열기"):
    audio_file = st.audio_input("음성을 녹음하세요")
    uploaded_image = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg", "gif"])

# 음성 -> 텍스트 변환
if audio_file is not None:
    with st.spinner("음성 인식 중..."):
        transcript_result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ko"
        )
    user_prompt = transcript_result.text

# 사용자 입력이 존재하면 처리
if user_prompt or uploaded_image:
    combined_prompt = user_prompt or ""
    
    # 이미지 표시
    if uploaded_image:
        st.image(uploaded_image, width=150, caption="첨부된 이미지")

    # 사용자 질문을 대화 기록에 저장 (텍스트만)
    st.session_state.messages.append({"role": "user", "content": combined_prompt})
    
    # lang_graph 기반 챗봇 Agent 실행
    with st.spinner("답변을 생성 중입니다..."):
        response = ask_lang_graph_agent(combined_prompt)
    
    # 최종 답변 추출
    assistant_response = response["messages"][-1].content

    # 이미지 링크 검색
    match = re.search(IMAGE_PATTERN, assistant_response, flags=re.IGNORECASE)
    related_image = None
    if match:
        related_image = match.group(0)
        # 답변에서 이미지 URL 제거
        assistant_response = re.sub(IMAGE_PATTERN, '', assistant_response, count=1, flags=re.IGNORECASE)
    
    # TTS 처리
    if assistant_response.strip():
        try:
            tts = gTTS(assistant_response, lang='ko')
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            audio_bytes = audio_fp.getvalue()
            tts_audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as tts_error:
            st.error(f"TTS 에러: {tts_error}")
            tts_audio_b64 = None
    else:
        tts_audio_b64 = None

    # 대화 기록에 추가
    assistant_message = {
        "role": "assistant",
        "content": assistant_response,
        "tts": tts_audio_b64,
    }
    if related_image is not None:
        assistant_message["image"] = related_image

    st.session_state.messages.append(assistant_message)

# ==========================================
# 전체 대화 기록 출력
# ==========================================
st.markdown("### 대화 기록")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # assistant 메시지: 관련 이미지를 먼저 표시 (답변 바로 위에)
        if message["role"] == "assistant" and message.get("image"):
            st.markdown(
                f'<img class="responsive-image" src="{message["image"]}" alt="관련 이미지">',
                unsafe_allow_html=True
            )

        st.markdown(message["content"])
        # assistant 메시지에 TTS가 저장되어 있으면 base64 디코딩 후 오디오 출력
        if message["role"] == "assistant" and message.get("tts"):
            audio_bytes = base64.b64decode(message["tts"])
            st.audio(audio_bytes, format="audio/mp3")
