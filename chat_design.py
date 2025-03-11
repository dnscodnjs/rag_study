import streamlit as st
import time

# 페이지 설정
st.set_page_config(page_title="KCC Auto Manager", page_icon="🚍")

st.title("KCC Auto Manager 🚗")
st.caption("자동차 사용 매뉴얼 및 서비스 센터 위치 찾기를 도와드립니다!")

# 세션 상태에 채팅 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "show_uploader" not in st.session_state:
    st.session_state["show_uploader"] = False

# 기존 메시지 출력
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# CSS를 이용해 아이콘 버튼 영역에 고유 id 부여 및 스타일 적용
st.markdown("""
    <style>
    /* id가 upload_icon인 영역 안의 버튼에 아이콘 배경을 적용 */
    #upload_icon button {
        background-size: 30px 30px;
        background-repeat: no-repeat;
        width: 40px;
        height: 40px;
        border: none;
        background-color: transparent;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

# 1) 채팅 입력 영역 + 아이콘 버튼 (아이콘은 업로드 영역 토글)
col_input, col_icon = st.columns([8, 1])
with col_input:
    user_input = st.text_input("", placeholder="차량에 관련된 궁금한 내용들을 말씀해주세요!")

with col_icon:
    st.markdown('<div id="upload_icon">', unsafe_allow_html=True)
    icon_clicked = st.button("", key="icon_click", help="사진 첨부", 
                              on_click=lambda: st.session_state.update({"show_uploader": not st.session_state["show_uploader"]}))
    st.markdown('</div>', unsafe_allow_html=True)

# 2) 텍스트 입력 처리 및 더미 AI 응답
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("답변을 생성하는 중입니다..."):
        time.sleep(1)
        dummy_resp = "이것은 데모 응답입니다."
    st.session_state["messages"].append({"role": "ai", "content": dummy_resp})
    with st.chat_message("ai"):
        st.write(dummy_resp)

# 3) 이미지 업로드 영역 (아이콘 버튼 클릭 시 표시)
if st.session_state["show_uploader"]:
    uploaded_image = st.file_uploader("사진을 첨부해주세요", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="업로드된 이미지", use_container_width=True)
        st.session_state["messages"].append({"role": "user", "content": "[사진 업로드]"})
        st.session_state["show_uploader"] = False
