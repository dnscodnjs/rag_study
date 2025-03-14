from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langchain_community.tools import TavilySearchResults
from typing import Literal
from langchain.tools import tool

import json
import hashlib
from typing_extensions import List
import os

load_dotenv()

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

@tool
def vector_retrieve_tool(query: str) -> List[Document]:
    """Retrieve documents based on the given query."""
    return retriever.invoke(query)


car_type = "eqs"
embedding = OpenAIEmbeddings(model="text-embedding-3-large") #text-embedding-ada-002, text-embedding-3-large

# Pinecone 인덱스 생성
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "kcc"
# text-embedding-ada-002 모델의 임베딩 차원은 1536입니다.
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
    )

index = pc.Index(index_name)
database = PineconeVectorStore(index_name=index_name, embedding=embedding)

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

llm = ChatOpenAI(model='gpt-4o')
retriever = database.as_retriever(search_kwargs={"k": 3})

def retrieve_search_node(state: MessagesState) -> Command:
    retrieve_search_agent = create_react_agent(
        llm, 
        tools=[vector_retrieve_tool],
        state_modifier = (
        '당신은 메르세데스 벤츠 ' + car_type + ' 자동차 매뉴얼 조사관입니다. '
        '제공한 정보를 고려하고 답변해 주세요. 의견이 아닌 사실만 제공해 주세요. 답변은 한국어로 번역해서 출력해주시고, '
        '각 정보 항목을 별도의 줄로 구분하여 이쁘게 포맷팅해 주세요. '
        )
    )
    result = retrieve_search_agent.invoke(state)
    # 내부 retrieval 결과를 상태에 저장
    state["retrieve_result"] = result['messages'][-1].content
    
    # 사용자 쿼리를 state에서 추출 (첫 번째 메시지가 HumanMessage라고 가정)
    user_query = state["messages"][0].content
    # retriever를 직접 호출하여 Document 리스트를 가져옴
    docs = retriever.invoke(user_query)
    
    displayed_image = None
    for doc in docs:
        meta = doc.metadata
        if "image_paths" in meta and meta["image_paths"]:
            try:
                # image_paths는 json.dumps로 저장되어 있으므로 디코딩
                image_paths = json.loads(meta["image_paths"])
            except Exception:
                image_paths = meta["image_paths"]
            if isinstance(image_paths, list) and len(image_paths) > 0:
                displayed_image = image_paths[0]
                break
    # retrieval 결과 앞에 이미지 링크(존재할 경우) 추가
    state["retrieve_result"] = (displayed_image or "") + '\n\n' + state["retrieve_result"] + "\n"
    
    # 기존 messages 리스트에 retrieval 결과를 추가
    state.setdefault("messages", []).append(
        HumanMessage(content=state["retrieve_result"], name="retrieve_search")
    )
    # 전체 messages 리스트를 업데이트
    return Command(update={'messages': state["messages"]})


def evaluate_node(state: MessagesState) -> Command[Literal['web_search', END]]:
    retrieve_result = state.get("retrieve_result", "").strip()

    # retrieval 결과가 충분할 수 있으므로 LLM 평가 프롬프트 실행
    eval_prompt = (
        "아래의 retrieval 결과가 충분한 답변을 제공하는지 평가해주세요. "
        "200자 이상의 답변 또는 답변 내용이 질문에 충분히 부합하는지 평가해야합니다."
        "불충분한 경우 'yes'라고, 충분한 경우 'no'라고 답해주세요.\n\n"
        "Retrieve Results:\n{result}"
    )
    evaluation = llm.invoke(eval_prompt.format(result=retrieve_result))
    
    if "yes" in evaluation.content.lower():
          print("답변이 충분하지 않습니다. 웹 검색을 시도합니다.")
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
        '당신은 메르세데스 벤츠 ' + car_type + ' 자동차 매뉴얼 조사관입니다. '
        '웹사이트 정보를 상세히 회신해 주세요. 답변은 한국어로 번역해서 출력해주시고, '
        '각 정보 항목을 별도의 줄로 구분하여 이쁘게 포맷팅해 주세요. '
        '마지막에 웹사이트 별 정보 소스의 링크를 각 정보 본문 뒤에 출력해 주세요.'
        )
    )
    result = web_search_agent.invoke(state)
    state["web_result"] = result['messages'][-1].content
    state.setdefault("messages", []).append(
        HumanMessage(content=state["web_result"], name="web_search")
    )
    return Command(update={'messages': state["messages"]})


# 그래프 생성
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("retrieve_search", retrieve_search_node)
graph_builder.add_node("evaluate", evaluate_node)
graph_builder.add_node("web_search", web_search_node)

graph_builder.add_edge(START, "retrieve_search")
graph_builder.add_edge("retrieve_search", "evaluate")
graph_builder.add_edge("web_search", END)

graph = graph_builder.compile()

# 쿼리 실행
while 1:
    query = input("질문을 입력하세요: ")
    if query == 'exit':
        break
    for chunk in graph.stream(
        {"messages": [("user", query)]}, stream_mode="values"
    ):
        chunk['messages'][-1].pretty_print()

