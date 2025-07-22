import streamlit as st
import os
import time
import google.generativeai as genai
import logging
from dotenv import load_dotenv
from utils import (
    
    check_data_files,
    load_data,
    run_parallel_agents,
    run_head_agent,
    get_conversation_history
)

# --- 환경 변수 및 Gemini API 설정 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 페이지 설정
st.set_page_config(
    page_title="관세법 판례 기반 챗봇",
    page_icon="⚖️",
    layout="wide",
)

# 애플리케이션 제목
st.title("⚖️ 관세법 판례 기반 챗봇")
st.markdown("관세법 판례 정보를 활용한 AI 기반 법률 챗봇입니다.")


# 대화 관련 설정
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

# 대화 맥락 관리 설정
if "context_enabled" not in st.session_state:
    st.session_state.context_enabled = True

# 데이터 저장을 위한 세션 상태 설정
if "loaded_data" not in st.session_state:
    st.session_state.loaded_data = {
        "court_cases": [],
        "tax_cases": [],
        "preprocessed_data": {}
    }

with st.sidebar:
    st.header("설정")
    
    
    # 대화 관리 옵션들
    st.header("대화 관리")
    
    # 대화 맥락 활용 옵션
    context_enabled = st.checkbox("이전 대화 맥락 활용", value=st.session_state.context_enabled)
    if context_enabled != st.session_state.context_enabled:
        st.session_state.context_enabled = context_enabled
        if context_enabled:
            st.success("이전 대화 맥락을 활용합니다.")
        else:
            st.info("각 질문을 독립적으로 처리합니다.")
    
    # 최근 대화 유지 수 선택
    if st.session_state.context_enabled:
        max_history = st.slider("최근 대화 유지 수", min_value=2, max_value=10, value=5)
        st.session_state.max_history = max_history
    
    # 새로운 대화 시작 버튼
    if st.button("새로운 대화 시작하기"):
        # 메시지 기록만 초기화 (데이터는 유지)
        st.session_state.messages = []
        st.session_state.processing = False
        st.success("새로운 대화가 시작되었습니다.")

# 실행 시 데이터 파일 존재 여부 확인
has_data_files = check_data_files()
if not has_data_files:
    st.warning("일부 데이터 파일이 없습니다. 예시 데이터를 사용하거나 필요한 파일을 추가해주세요.")
else:
    # 데이터가 아직 로드되지 않았다면 로드
    if not st.session_state.loaded_data["court_cases"]:
        with st.spinner("데이터를 로드하고 전처리 중입니다..."):
            court_cases, tax_cases, preprocessed_data = load_data()
            st.session_state.loaded_data = {
                "court_cases": court_cases,
                "tax_cases": tax_cases,
                "preprocessed_data": preprocessed_data
            }
            st.success("데이터 로드 및 전처리가 완료되었습니다.")

# 저장된 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 처리 시작
    st.session_state.processing = True
    
    # 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                # 저장된 데이터 사용
                court_cases = st.session_state.loaded_data["court_cases"]
                tax_cases = st.session_state.loaded_data["tax_cases"]
                preprocessed_data = st.session_state.loaded_data["preprocessed_data"]
                
                # 대화 맥락 가져오기
                conversation_history = ""
                if st.session_state.context_enabled:
                    conversation_history = get_conversation_history(
                        max_messages=st.session_state.get('max_history', 5)
                    )
                
                # 프로그레스 바 표시
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                # 단계별 진행 상태 표시
                progress_text.text("1/3 에이전트 실행 중...")
                progress_bar.progress(33)
                
                # 에이전트 실행 (대화 기록 전달 및 전처리된 데이터 활용)
                agent_responses = run_parallel_agents(
                    court_cases, tax_cases, preprocessed_data, prompt, conversation_history
                )
                
                progress_text.text("2/3 결과 통합 중...")
                progress_bar.progress(66)
                
                # Head Agent로 최종 응답 생성 (대화 기록 전달)
                
                head_response = run_head_agent(
                    agent_responses, prompt, conversation_history
                )
                
                # 응답 텍스트 추출 (수정된 함수 반환값에 맞춤)
                if isinstance(head_response, dict):
                    final_response = head_response.get("response", "응답을 생성할 수 없습니다.")
                    already_displayed = head_response.get("already_displayed", False)
                else:
                    # 이전 버전 호환성을 위한 처리
                    final_response = head_response
                    already_displayed = False
                
                progress_text.text("3/3 답변 생성 완료")
                progress_bar.progress(100)
                time.sleep(0.5)  # 완료 상태 잠시 표시
                
                # 프로그레스 바 제거
                progress_text.empty()
                progress_bar.empty()
                
                # 이미 스트리밍으로 표시되지 않은 경우에만 최종 응답 표시
                if not already_displayed:
                    st.markdown(final_response)

                # 각 에이전트의 응답을 expander로 표시
                with st.expander("🤖 각 에이전트 답변 보기"):
                    for i, agent_resp in enumerate(agent_responses):
                        st.subheader(f"📋 {agent_resp['agent']}")
                        st.markdown(agent_resp['response'])
                        if i < len(agent_responses) - 1:  # 마지막 에이전트가 아니면 구분선 추가
                            st.divider()
                
                # 응답 저장
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                logging.error(f"전체 처리 오류: {str(e)}")
                # 오류 메시지도 저장
                error_message = f"오류가 발생했습니다: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # 처리 완료
    st.session_state.processing = False

# 사이드바에 사용 예시 및 정보 추가
with st.sidebar:
    st.subheader("프로젝트 정보")
    st.markdown("""
    이 챗봇은 관세분야야 판례를 기반으로 답변을 생성합니다.
    - 7개의 AI 에이전트 활용
    - 일반 agent : Google Gemini 2.0 Flash 모델 사용
    - Head agent : Google Gemini 2.5 Flash 모델 사용
    - 질문과 관련성이 높은 판례 검색 기능
    - 관련 자료 기반 정확한 응답 생성
    """)