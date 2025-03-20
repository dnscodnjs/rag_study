# 외부 라이브러리 및 모듈 임포트
from openai import OpenAI  # OpenAI 라이브러리에서 OpenAI 클래스를 임포트 (구버전)
import openai  # 최신 OpenAI 인터페이스 사용
import streamlit as st  # 웹 애플리케이션을 만들기 위한 Streamlit 라이브러리 임포트
import os  # 운영체제 관련 기능(예: 환경 변수, 파일 경로 등) 사용을 위한 모듈 임포트
import base64  # 이미지나 오디오 데이터를 인코딩/디코딩하기 위한 base64 모듈 임포트
import json  # JSON 데이터 읽기/쓰기를 위한 모듈 임포트
from dotenv import load_dotenv  # .env 파일에 저장된 환경변수를 로드하기 위한 함수 임포트
import hashlib  # 파일의 해시값 계산(무결성 검사 등)을 위한 모듈 임포트
import io  # 메모리 내 파일 객체를 다루기 위한 모듈 임포트
from gtts import gTTS  # 텍스트를 음성으로 변환하는 gTTS (Google Text-to-Speech) 라이브러리 임포트
import re  # 정규표현식을 사용하기 위한 모듈 임포트

# LangChain, LangGraph, Pinecone 관련 모듈 임포트
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # OpenAI 임베딩과 채팅 모델 사용을 위한 모듈 임포트
from langchain_core.prompts import PromptTemplate  # 프롬프트 템플릿을 생성하기 위한 모듈 임포트
from langchain_community.tools import TavilySearchResults  # 커뮤니티 도구 중 웹 검색 관련 도구 임포트
from langchain.tools import tool  # LangChain의 도구(함수)를 데코레이터 방식으로 사용하기 위한 모듈 임포트
from langchain_pinecone import PineconeVectorStore  # Pinecone 벡터 스토어를 사용하기 위한 모듈 임포트
from langchain.docstore.document import Document  # 문서 객체 생성에 사용되는 Document 클래스 임포트
from langchain_core.messages import HumanMessage  # 대화에서 사용자의 메시지를 나타내는 클래스 임포트
from langgraph.prebuilt import create_react_agent  # 사전 구축된 React 에이전트를 생성하기 위한 함수 임포트
from langgraph.graph import StateGraph, MessagesState, START, END  # 대화 상태 흐름(그래프) 관련 모듈 임포트
from langgraph.types import Command  # 상태 전환 명령(Command) 관련 타입 임포트
from pinecone import Pinecone  # 최신 Pinecone API 사용을 위한 모듈 임포트
from typing import Literal  # 타입 힌팅을 위한 Literal 임포트
from typing_extensions import List  # 타입 힌팅을 위한 List 임포트

# ==========================================
# 설정 및 초기화
# ==========================================
load_dotenv()  # .env 파일에 정의된 환경변수를 로드하여 API 키 등 필요한 값들을 사용할 수 있게 함

# OpenAI 클라이언트와 최신 openai 모듈의 API 키를 환경변수에서 가져와 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# 기본 설정 값 정의
index_name = "test"  # Pinecone에 생성할 인덱스 이름
car_type = "EQS"  # 차량 종류 (예: Mercedes Benz EQS)
IMAGE_PATTERN = r'(https?://\S+\.(?:png|jpg|jpeg|gif))'  # 텍스트 내에서 이미지 URL을 찾기 위한 정규표현식

# OpenAI 임베딩 모델 초기화 (예: text-embedding-3-large 모델 사용)
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
# Pinecone 벡터 스토어 생성: 지정한 인덱스 이름과 임베딩 모델을 사용
database = PineconeVectorStore(index_name=index_name, embedding=embedding)
# 문서 검색을 위한 retriever 객체 생성 (최대 3개의 관련 문서 반환)
retriever = database.as_retriever(search_kwargs={"k": 3})
# OpenAI의 gpt-3.5-turbo 모델을 사용하여 채팅 에이전트 초기화
llm = ChatOpenAI(model='gpt-3.5-turbo')

# Streamlit 페이지 설정: 페이지 제목과 레이아웃을 지정
st.set_page_config(page_title="KCC Auto Manager", layout="wide")
st.title("KCC Auto Manager")  # 페이지 상단에 제목 출력

# 페이지 내 CSS 스타일 정의: 반응형 이미지 크기, 메뉴 및 헤더/푸터 숨김 처리
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
    /* 기본 헤더, 푸터, 메뉴 숨기기 */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit 세션 상태 초기화: 대화 기록과 마지막으로 출력한 메시지 인덱스를 저장
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_displayed" not in st.session_state:
    st.session_state.last_displayed = 0

# ==========================================
# 사이드바 설정
# ==========================================
with st.sidebar:
    st.title("🤗💬 Auto Manager 🚗")  # 사이드바 제목
    st.subheader("현재 차량: 벤츠 S 클래스")  # 현재 차량 정보 출력
    st.markdown("### 설정")
    # 사용자에게 선택 가능한 언어 옵션 제공 (한국어, English, Deutsch)
    user_language = st.selectbox("사용 언어를 선택하세요", ("한국어", "English", "Deutsch"))
    st.markdown(f"선택된 언어: **{user_language}**")
    st.markdown("### 바로가기")
    # Mercedes-Benz 공식 홈페이지 링크 추가
    st.markdown("[메르세데스-벤츠 공식 홈페이지](https://www.mercedes-benz.co.kr/)")

# ==========================================
# 데이터 인덱싱 관련 함수들
# ==========================================
def get_file_hash(filename):
    """
    파일의 내용을 읽어 SHA-256 해시값을 계산하여 반환.
    이를 통해 파일의 변경 여부를 확인할 수 있음.
    """
    h = hashlib.sha256()
    with open(filename, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def load_stored_hash(hash_file):
    """
    저장된 해시값을 지정된 파일에서 읽어 반환.
    파일이 없으면 None을 반환.
    """
    try:
        with open(hash_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def save_hash(hash_file, hash_value):
    """
    계산된 해시값을 지정된 파일에 저장.
    """
    with open(hash_file, "w", encoding="utf-8") as f:
        f.write(hash_value)

def index_data():
    """
    'usage.json' 파일의 데이터를 읽어와 문서(Document) 객체로 변환한 후,
    Pinecone 벡터 스토어에 추가하는 함수.
    이전에 저장된 해시값과 비교하여 파일 내용이 변경된 경우에만 재인덱싱 수행.
    """
    hash_file = "usage.json.hash"
    current_hash = get_file_hash("usage.json")
    stored_hash = load_stored_hash(hash_file)
    
    if stored_hash != current_hash:
        # JSON 파일을 열어 데이터 로드
        with open("usage.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)

        documents = []
        # JSON 데이터의 각 항목에 대해 문서 생성
        for item in test_data:
            pdf_file = item.get("pdf_file", "")
            structure = item.get("structure", [])
            for section in structure:
                title = section.get("title", "")
                sub_titles = section.get("sub_titles", [])
                content_text = title  # 섹션 제목을 기본 내용으로 설정
                image_paths = []
                for sub in sub_titles:
                    sub_title = sub.get("title", "")
                    contents = sub.get("contents", [])
                    # 내용 중 이미지 파일 경로(파일 확장자로 판별) 추출
                    for content in contents:
                        if content.lower().endswith(('.jpeg', '.jpg', '.png', '.gif')):
                            image_paths.append(content)
                    # 'images' 키에 포함된 이미지 URL도 추가
                    for img_url in sub.get("images", []):
                        if isinstance(img_url, str):
                            image_paths.append(img_url)
                    # 이미지가 아닌 텍스트 내용만 따로 추가
                    non_image_contents = [c for c in contents if not c.lower().endswith(('.jpeg', '.jpg', '.png', '.gif'))]
                    content_text += "\n" + sub_title + "\n" + "\n".join(non_image_contents)
                # 메타데이터 생성 (PDF 파일명, 섹션 제목, 이미지 경로 목록)
                metadata = {
                    "pdf_file": pdf_file,
                    "section_title": title,
                    "image_paths": json.dumps(image_paths)
                }
                # Document 객체 생성 후 리스트에 추가
                documents.append(Document(page_content=content_text, metadata=metadata))
        # 생성된 문서를 Pinecone 데이터베이스에 추가
        database.add_documents(documents)
        # 새 해시값을 저장하여 이후 변경 여부를 판단
        save_hash(hash_file, current_hash)

# Pinecone 초기화: 환경변수에 저장된 API 키 사용
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Pinecone 인덱스 생성 또는 확인: 지정한 이름의 인덱스가 없으면 새로 생성
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=3072,  # 임베딩 벡터의 차원 (모델에 따라 다름)
        metric="cosine",  # 코사인 유사도를 사용하여 벡터 간 유사도 계산
    )
index = pc.Index(index_name)

# 데이터 인덱싱 진행 (스피너를 통해 진행 상황 표시)
with st.spinner("데이터 인덱싱 중..."):
    index_data()

# ==========================================
# LangGraph 정의 및 에이전트 설정
# ==========================================

# LangChain 도구 데코레이터를 사용하여 벡터 검색 도구 정의
@tool
def vector_retrieve_tool(query: str) -> List[Document]:
    """
    입력된 쿼리에 기반하여 Pinecone 벡터 스토어에서 관련 문서를 검색하는 함수.
    반환값은 Document 객체 리스트.
    """
    return retriever.invoke(query)

def retrieve_search_node(state: MessagesState) -> Command:
    """
    LangGraph의 상태 노드 중 하나로, 사용자의 질문에 대해 내부 검색(벡터 검색)을 수행하고
    검색 결과를 기반으로 응답을 생성합니다.
    
    - 검색 에이전트는 차량 매뉴얼 전문가 역할을 하며, 사실 기반의 답변을 생성합니다.
    - 생성된 답변은 한국어로 번역됩니다.
    - 검색 결과 중 이미지가 있다면 첫 번째 이미지를 선택할 수 있도록 처리합니다.
    """
    retrieve_search_agent = create_react_agent(
        llm, 
        tools=[vector_retrieve_tool],
        state_modifier=(
            "You are an expert on Mercedes Benz " + car_type + " car manuals. "
            "Please consider the information you provided and reply. Provide factual answers, not opinions. "
            "Translate the answers into Korean. "
            "Refer to the structure below to organize the website information:\n\n"
            "{{제목}}\nExample: Mercedes Benz EQS: Driver Display Charge Status Window Function\n\n"
            "{{주요 정보}}\n- Feature Summary and Key Points\n\n"
            "{{상세 설명}}\n- Detailed description of features\n\n"
        )
    )
    # 에이전트를 호출하여 검색 결과 생성
    result = retrieve_search_agent.invoke(state)
    # 검색 결과 중 마지막 메시지의 내용을 저장
    state["retrieve_result"] = result['messages'][-1].content

    # 사용자의 입력(첫 메시지)을 기반으로 관련 문서를 검색
    user_query = state["messages"][0].content
    docs = retriever.invoke(user_query)
    
    displayed_image = None
    # 검색 결과에서 메타데이터에 저장된 이미지 경로가 있는 경우, 첫 번째 이미지를 선택
    for doc in docs:
        meta = doc.metadata
        if "image_paths" in meta and meta["image_paths"]:
            try:
                image_paths = json.loads(meta["image_paths"])
            except Exception:
                image_paths = meta["image_paths"]
            if isinstance(image_paths, list) and image_paths:
                displayed_image = image_paths[0]
                break

    # 검색 결과를 대화 기록에 추가 (대화 흐름에 사용)
    state.setdefault("messages", []).append(
        HumanMessage(content=state["retrieve_result"], name="retrieve_search")
    )
    return Command(update={'messages': state["messages"]})

def evaluate_node(state: MessagesState) -> Command[Literal['web_search', END]]:
    """
    LangGraph 상태 노드로, 생성된 검색 결과가 충분한지 평가합니다.
    
    - 만약 생성된 답변의 내용이 충분하지 않으면(예: 200자 이하) 웹 검색 노드로 전환합니다.
    - 그렇지 않으면 대화 흐름을 종료(END)합니다.
    """
    if state["messages"][-1].content is None:
        st.write("답변이 충분하지 않습니다. 웹 검색을 시도합니다.")
        return Command(goto='web_search')

    # 평가를 위한 프롬프트 템플릿 생성: 답변의 길이에 따라 'yes' 또는 'no'를 반환하도록 함
    eval_prompt = PromptTemplate.from_template(
        "You are an expert on Mercedes Benz " + car_type + " car manuals. "
        "Please rate if the retrive results below provide sufficient answers. "
        "If the answer has more than 200 characters of detailed info, answer 'no'. Otherwise, answer 'yes'.\n\n"
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
    """
    LangGraph 상태 노드로, 웹 검색 도구를 활용하여 보다 상세한 정보를 검색합니다.
    
    - 웹 검색 에이전트는 전문가 역할로 답변을 생성하며, 
      정해진 템플릿 구조(제목, 주요 정보, 상세 설명)를 사용하여 결과를 구성합니다.
    - 생성된 웹 검색 결과는 대화 기록에 추가됩니다.
    """
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
        state_modifier=(
            "You are an expert on Mercedes Benz " + car_type + " car manuals. "
            "Reply with detailed website information. Translate the answer into Korean. "
            "Use the following structure:\n\n"  
            "## {{제목}}\nExample: Mercedes Benz EQS: Driver Display Charge Status Window Function\n\n"
            "### {{주요 정보}}\n- Feature Summary and Key Points\n\n"
            "### {{상세 설명}}\n- Detailed description of the feature. / {{출처}}: real website link\n\n"
            "각 항목은 별도 줄로 구분해 출력해주세요."
        )
    )
    # 웹 검색 에이전트를 호출하여 결과 생성
    result = web_search_agent.invoke(state)
    state["web_result"] = result['messages'][-1].content
    # 웹 검색 결과를 대화 기록에 추가
    state.setdefault("messages", []).append(
        HumanMessage(content=state["web_result"], name="web_search")
    )
    return Command(update={'messages': state["messages"]})

# ==========================================
# 그래프 구성 (대화 흐름 제어)
# ==========================================
graph_builder = StateGraph(MessagesState)
# 각 상태 노드를 그래프에 추가
graph_builder.add_node("retrieve_search", retrieve_search_node)
graph_builder.add_node("evaluate", evaluate_node)
graph_builder.add_node("web_search", web_search_node)

# 노드 간 전이 설정: 시작 -> 검색 -> 평가 -> (웹 검색 또는 종료)
graph_builder.add_edge(START, "retrieve_search")
graph_builder.add_edge("retrieve_search", "evaluate")
graph_builder.add_edge("web_search", END)

# 최종 그래프 컴파일
graph = graph_builder.compile()

def ask_lang_graph_agent(query):
    """
    사용자의 질문(query)을 받아 LangGraph 에이전트를 호출하여 전체 대화 흐름에 따른 응답을 생성.
    """
    return graph.invoke({"messages": [("user", query)]})

# ==========================================
# 사용자 입력 및 처리
# ==========================================
# 사용자로부터 채팅 입력을 받음 (텍스트 입력창)
user_prompt = st.chat_input("메시지를 입력하세요")

# 음성 입력과 이미지 업로드를 위한 확장 영역 제공
with st.expander("음성 입력 및 이미지 첨부 열기"):
    audio_file = st.audio_input("음성을 녹음하세요")
    uploaded_image = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg", "gif"])

# 음성 파일이 있는 경우: OpenAI의 Whisper 모델을 이용하여 음성을 텍스트로 변환
if audio_file is not None:
    with st.spinner("음성 인식 중..."):
        transcript_result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ko"
        )
    user_prompt = transcript_result.text

# 사용자가 텍스트 입력 또는 이미지를 업로드한 경우 처리
if user_prompt or uploaded_image:
    # 사용자 입력된 텍스트와 이미지 정보를 결합하여 하나의 프롬프트로 구성
    combined_prompt = user_prompt or ""
    
    # 이미지가 업로드된 경우, 화면에 해당 이미지를 표시
    if uploaded_image:
        st.image(uploaded_image, width=150, caption="첨부된 이미지")

    # 사용자의 메시지를 세션 상태(대화 기록)에 추가
    st.session_state.messages.append({"role": "user", "content": combined_prompt})
    
    # LangGraph 에이전트를 호출하여 답변 생성 (처리 중 스피너 표시)
    with st.spinner("답변을 생성 중입니다..."):
        response = ask_lang_graph_agent(combined_prompt)
    
    # assistant_response에 에이전트의 응답 저장
    assistant_response = response

    # 이후 assistant_response를 활용하여 추가 처리(예: 이미지 추출, TTS 변환 등)를 진행할 수 있음.
    # 여기서는 코드가 중단된 부분이므로, 후속 처리 코드가 추가되어야 합니다.
