import streamlit as st
import json
import os
import time
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import tempfile

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
st.markdown("관세법 판례 및 조세심판 결정례 정보를 활용한 AI 기반 법률 챗봇입니다.")

# API 키 설정
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False

# Gemini API 초기화 함수
def initialize_gemini_api(api_key):
    try:
        genai.configure(api_key=api_key)
        # 간단한 API 테스트
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Test")
        return True
    except Exception as e:
        st.error(f"API 초기화 오류: {str(e)}")
        return False

# 사이드바에 API 키 입력 폼
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("Google API 키를 입력하세요", type="password")
    if st.button("API 키 저장"):
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            if initialize_gemini_api(api_key):
                st.session_state.api_key_set = True
                st.success("API 키가 저장되었습니다.")
            else:
                st.error("API 키가 유효하지 않습니다.")
        else:
            st.error("API 키를 입력해주세요.")
    
    # 새로운 대화 시작 버튼 추가
    st.header("대화 관리")
    if st.button("새로운 대화 시작하기"):
        # 메시지 기록만 초기화 (데이터는 유지)
        st.session_state.messages = []
        st.session_state.processing = False
        st.success("새로운 대화가 시작되었습니다.")

# 데이터 파일 존재 여부 확인 함수
def check_data_files():
    court_file = "관세분야판례423개.json"
    tax_file = "조세심판결정례_final.zip"
    
    files_exist = True
    if not os.path.exists(court_file):
        st.sidebar.error(f"파일을 찾을 수 없습니다: {court_file}")
        files_exist = False
    if not os.path.exists(tax_file):
        st.sidebar.error(f"파일을 찾을 수 없습니다: {tax_file}")
        files_exist = False
        
    return files_exist

# ZIP 파일 압축 해제 함수
def extract_zip_file(zip_path):
    """ZIP 파일을 임시 디렉토리에 압축 해제하고 JSON 파일 내용 반환"""
    try:
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            # ZIP 파일 압축 해제
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # JSON 파일 찾기 (첫 번째 JSON 파일 사용)
            json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
            if not json_files:
                raise FileNotFoundError("ZIP 파일 내에 JSON 파일이 없습니다.")
            
            # JSON 파일 로드
            json_path = os.path.join(temp_dir, json_files[0])
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data
    except Exception as e:
        st.sidebar.error(f"ZIP 파일 처리 오류: {str(e)}")
        logging.error(f"ZIP 파일 처리 오류: {str(e)}")
        return []

# 데이터 로드 함수
@st.cache_data
def load_data():
    """판례 및 조세심판 결정례 데이터 로드"""
    try:
        # 판례 데이터 로드
        with open("관세분야판례423개.json", "r", encoding="utf-8") as f:
            court_cases = json.load(f)
        st.sidebar.success(f"판례 데이터 로드 완료: {len(court_cases)}건")
        
        # 조세심판 결정례 로드 (ZIP 파일)
        tax_cases = extract_zip_file("조세심판결정례_final.zip")
        if tax_cases:
            st.sidebar.success(f"조세심판결정례 데이터 로드 완료: {len(tax_cases)}건")
        else:
            st.sidebar.error("조세심판결정례 데이터를 로드할 수 없습니다.")
            tax_cases = []
        
    except FileNotFoundError as e:
        st.sidebar.error(f"파일을 찾을 수 없습니다: {e}")
        st.error("필수 데이터 파일을 찾을 수 없습니다. 애플리케이션 디렉토리에 필요한 파일이 있는지 확인하세요.")
        court_cases = []
        tax_cases = []
    except json.JSONDecodeError as e:
        st.sidebar.error(f"JSON 파일 파싱 오류: {e}")
        st.error("JSON 파일 형식이 올바르지 않습니다. 파일 형식을 확인하세요.")
        court_cases = []
        tax_cases = []
    
    return court_cases, tax_cases

def split_tax_cases(tax_cases):
    """조세심판결정례를 4개의 청크로 분할"""
    # 데이터 개수
    total_cases = len(tax_cases)
    chunk_size = max(1, total_cases // 4)  # 최소 1개는 되도록
    
    # 4개의 청크로 분할
    chunks = []
    for i in range(4):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < 3 else total_cases
        chunks.append(tax_cases[start_idx:end_idx])
    
    # 분할 정보 로그
    st.sidebar.info(f"조세심판결정례 분할: 총 {total_cases}건을 {[len(chunk) for chunk in chunks]}건씩 배분")
    
    return chunks

# 에이전트 프롬프트 정의
def get_agent_prompt(agent_type):
    """에이전트 유형에 따른 프롬프트 생성"""
    base_prompt = """
# Role
- 당신은 관세법 분야 전문성을 갖춘 법학 교수입니다.
- 당신은 판결문의 논리와 판사의 의도를 이해하고, 복잡한 법적 문제를 분석하는 능력이 탁월합니다.
- 사용자의 질문에 대해 주어진 데이터를 활용하여 상세하게 답변합니다.
- 주요 답변 내용:
    1. 판결문의 주요 내용 요약
    2. 주요 법적 쟁점 도출
    3. 법원의 판단 요지 및 그 근거 요약
    4. 법원이 인용한 주요 법률 조항 및 판례 설명
- 모든 답변은 두괄식으로 작성합니다.
"""
    if agent_type == "court_case":
        return base_prompt + "\n# 판례 데이터를 기반으로 응답하세요."
    elif agent_type == "tax_case":
        return base_prompt + "\n# 조세심판 결정례 데이터를 기반으로 응답하세요."
    else:  # head agent
        return """
# Role
- 당신은 관세법 분야 전문성을 갖춘 법학 교수이자 여러 자료를 통합하여 종합적인 답변을 제공하는 전문가입니다.
- 여러 에이전트로부터 받은 답변을 분석하고 통합하여 사용자의 질문에 가장 적합한 최종 답변을 제공합니다.
- 주요 역할:
    1. 서로 다른 정보 소스에서 나온 답변을 비교 분석
    2. 가장 관련성 높은 정보 선별
    3. 일관된 논리구조로 통합된 답변 생성
    4. 중복 정보 제거 및 핵심 정보 강조
- 모든 답변은 두괄식으로 작성합니다.
"""

# 텍스트 전처리 함수
def preprocess_text(text):
    """텍스트 정규화 및 전처리"""
    if not text or not isinstance(text, str):
        return ""
    # 공백 정규화 및 특수문자 처리
    text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나로
    text = text.strip()  # 앞뒤 공백 제거
    return text

# 데이터에서 텍스트 추출 함수
def extract_text_from_item(item, data_type):
    """데이터 아이템에서 검색에 사용할 텍스트 추출"""
    if data_type == "court_case":
        # 판례 데이터에서 텍스트 추출
        text_parts = []
        for key in ['사건번호', '선고일자\n(종결일자)', '판결주문', '청구취지', '판결이유']:
            if key in item and item[key]:
                sub_text = f'{key}: {item[key]} \n\n'
                text_parts.append(sub_text)
        return ' '.join(text_parts)
    else:  # tax_case
        # 조세심판 결정례에서 텍스트 추출
        text_parts = []
        for key in ['[청구번호]', '[제 목]', '[결정요지]', "[주    문]", "[이    유]"]:
            if key in item and item[key]:
                sub_text = f'{key}: {item[key]} \n\n'
                text_parts.append(sub_text)
        return ' '.join(text_parts)

# 관련성 높은 데이터 검색 함수
def search_relevant_data(data, query, data_type, top_n=15):
    """질문과 관련성이 높은 데이터 항목을 검색"""
    if not data:
        return []
    
    # 데이터에서 텍스트 추출
    corpus = []
    for item in data:
        text = extract_text_from_item(item, data_type)
        corpus.append(preprocess_text(text))
    
    # 쿼리 전처리
    query = preprocess_text(query)
    
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = vectorizer.transform([query])
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
        
        # 유사도 기준으로 상위 n개 항목 선택
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # 유사도가 0보다 큰 항목만 선택
        relevant_data = []
        for idx in top_indices:
            if similarities[idx] > 0:
                relevant_data.append(data[idx])
        
        return relevant_data
    except Exception as e:
        logging.error(f"검색 오류: {str(e)}")
        # 오류 발생 시 원본 데이터의 일부 반환
        return data[:min(top_n, len(data))]

# 에이전트 실행 함수
def run_agent(agent_type, data, user_query, agent_index=None):
    """특정 유형의 에이전트 실행"""
    # 프롬프트 생성
    prompt = get_agent_prompt(agent_type)
    
    # 질문과 관련성이 높은 데이터 검색
    data_type = "court_case" if agent_type == "court_case" else "tax_case"
    relevant_data = search_relevant_data(data, user_query, data_type)
    
    # 관련 데이터가 없는 경우 처리
    if not relevant_data:
        agent_label = f"Agent {agent_index}" if agent_index else "Head Agent"
        return {
            "agent": agent_label,
            "response": "관련된 데이터를 찾을 수 없습니다."
        }
    
    # 데이터 문자열로 변환
    data_str = json.dumps(relevant_data, ensure_ascii=False, indent=2)
    
    # 전체 프롬프트 구성
    full_prompt = f"{prompt}\n\n# 데이터\n{data_str}\n\n# 질문\n{user_query}"
    logging.info(f"Agent {agent_index if agent_index else 'Head'} 실행 시작 (관련 데이터: {len(relevant_data)}건)")
    
    try:
        # Gemini 모델 호출 - gemini-2.0-flash 모델 사용
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(full_prompt)
        
        agent_label = f"Agent {agent_index}" if agent_index else "Head Agent"
        logging.info(f"{agent_label} 응답 생성 완료")
        return {
            "agent": agent_label,
            "response": response.text
        }
    except Exception as e:
        error_msg = f"오류 발생: {str(e)}"
        logging.error(f"Agent {agent_index if agent_index else 'Head'} 오류: {error_msg}")
        return {
            "agent": f"Agent {agent_index}" if agent_index else "Head Agent",
            "response": error_msg
        }

# 병렬 에이전트 실행
def run_parallel_agents(court_cases, tax_cases, user_query):
    """모든 에이전트를 병렬로 실행하고 결과 반환"""
    results = []
    
    try:
        # 조세심판 결정례 데이터를 4개로 분할
        tax_cases_chunks = split_tax_cases(tax_cases)
        
        # ThreadPoolExecutor로 병렬 처리
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Agent 1: 판례 검색
            court_agent_future = executor.submit(run_agent, "court_case", court_cases, user_query, 1)
            
            # Agent 2-5: 조세심판 결정례 검색 (데이터 분할 필요)
            tax_agent_futures = []
            for i, chunk in enumerate(tax_cases_chunks, start=2):
                tax_agent_futures.append(executor.submit(run_agent, "tax_case", chunk, user_query, i))
            
            # 결과 수집
            results.append(court_agent_future.result())
            for future in tax_agent_futures:
                results.append(future.result())
    except Exception as e:
        logging.error(f"병렬 에이전트 실행 오류: {str(e)}")
        results.append({
            "agent": "Error Agent",
            "response": f"에이전트 실행 중 오류가 발생했습니다: {str(e)}"
        })
    
    return results

# Head Agent를 실행하여 최종 응답 생성
def run_head_agent(agent_responses, user_query):
    """각 에이전트의 응답을 통합하여 최종 응답 생성"""
    # 응답 데이터 준비
    responses_str = ""
    for resp in agent_responses:
        responses_str += f"\n## {resp['agent']} 응답:\n{resp['response']}\n\n"
    
    # Head Agent 프롬프트 생성
    prompt = get_agent_prompt("head")
    full_prompt = f"{prompt}\n\n# 에이전트 응답\n{responses_str}\n\n# 질문\n{user_query}\n\n# 지시사항\n위 에이전트들의 응답을 통합하여 사용자의 질문에 가장 적합한 최종 답변을 작성하세요."
    
    try:
        # Gemini 모델 호출
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        error_msg = f"Head Agent 오류 발생: {str(e)}"
        logging.error(error_msg)
        return error_msg

# 챗 인터페이스 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 저장된 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 프로그레스 상태 저장
if "processing" not in st.session_state:
    st.session_state.processing = False

# 실행 시 데이터 파일 존재 여부 확인
has_data_files = check_data_files()
if not has_data_files:
    st.warning("일부 데이터 파일이 없습니다. 예시 데이터를 사용하거나 필요한 파일을 추가해주세요.")

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    if not st.session_state.api_key_set:
        st.error("Google API 키를 먼저 설정해주세요.")
    else:
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
                    # 데이터 로드
                    court_cases, tax_cases = load_data()
                    
                    # 프로그레스 바 표시
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    # 단계별 진행 상태 표시
                    progress_text.text("1/3 에이전트 실행 중...")
                    progress_bar.progress(33)
                    
                    # 에이전트 실행
                    agent_responses = run_parallel_agents(court_cases, tax_cases, prompt)
                    
                    progress_text.text("2/3 결과 통합 중...")
                    progress_bar.progress(66)
                    
                    # Head Agent로 최종 응답 생성
                    final_response = run_head_agent(agent_responses, prompt)
                    
                    progress_text.text("3/3 답변 생성 완료")
                    progress_bar.progress(100)
                    time.sleep(0.5)  # 완료 상태 잠시 표시
                    
                    # 프로그레스 바 제거
                    progress_text.empty()
                    progress_bar.empty()
                    
                    # 최종 응답 표시
                    st.markdown(final_response)
                    
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
    st.subheader("사용 예시")
    st.markdown("""
    다음과 같은 질문을 시도해보세요:
    - 관세경정거부처분이란 무엇인가요?
    - HSK 분류와 관련된 주요 판례는?
    - 품목분류 관련 주요 법적 쟁점은?
    - 관세법 제42조의 가산세 면제 조건은?
    """)
    
    st.subheader("프로젝트 정보")
    st.markdown("""
    이 챗봇은 관세법 판례와 조세심판 결정례를 기반으로 답변을 생성합니다.
    - 6개의 AI 에이전트 활용
    - Google Gemini 2.0 Flash 모델 사용
    - 질문과 관련성이 높은 판례/결정례 검색 기능
    - 관련 자료 기반 정확한 응답 생성
    """)