import streamlit as st
import json
import os
from concurrent.futures import ThreadPoolExecutor
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import tempfile
from dotenv import load_dotenv

# --- 환경 변수 및 Gemini API 설정 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
from google import genai
from google.genai import types
client = genai.Client(api_key=GOOGLE_API_KEY)

# 대화 기록 관리 함수
def get_conversation_history(max_messages=10):
    """최근 대화 기록을 문자열로 반환"""
    if "messages" not in st.session_state or len(st.session_state.messages) <= 1:
        return ""
    
    # 가장 최근 메시지는 현재 처리중인 사용자 질문이므로 제외
    messages = st.session_state.messages[:-1]
    
    # 최대 메시지 수를 제한하여 컨텍스트 길이 관리
    if len(messages) > max_messages:
        messages = messages[-max_messages:]
    
    conversation = ""
    for msg in messages:
        role = "사용자" if msg["role"] == "user" else "챗봇"
        conversation += f"{role}: {msg['content']}\n\n"
    
    return conversation

# 데이터 파일 존재 여부 확인 함수
def check_data_files():
    """필요한 데이터 파일 존재 여부 확인"""
    court_file = "관세분야판례423개.json"
    tax_file = "국가법령정보센터_관세판례.json"
    
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
    else:  # 국가법령정보센터_관세판례
        # 국가법령정보센터_관세판례에서 텍스트 추출
        text_parts = []
        for key in ['제목', '판례번호', '내용']:
            if key in item and item[key]:
                sub_text = f'{key}: {item[key]} \n\n'
                text_parts.append(sub_text)
        return ' '.join(text_parts)
    

# 데이터 로드 함수 - 초기화 시 1번만 호출
@st.cache_data
def load_data():
    """판례 데이터 로드"""
    try:
        # 판례 데이터 로드1
        with open("관세분야판례423개.json", "r", encoding="utf-8") as f:
            court_cases = json.load(f)
        st.sidebar.success(f"판례 데이터 로드 완료: {len(court_cases)}건")

        # 판례 데이터 로드2
        with open("국가법령정보센터_관세판례.json", "r", encoding="utf-8") as f:
            tax_cases = json.load(f)
        st.sidebar.success(f"판례 데이터 로드 완료: {len(tax_cases)}건")
        
        # 데이터 전처리 및 벡터화 
        preprocessed_data = preprocess_data(court_cases, tax_cases)
        return court_cases, tax_cases, preprocessed_data
        
    except FileNotFoundError as e:
        st.sidebar.error(f"파일을 찾을 수 없습니다: {e}")
        st.error("필수 데이터 파일을 찾을 수 없습니다. 애플리케이션 디렉토리에 필요한 파일이 있는지 확인하세요.")
        return [], [], {}
    except json.JSONDecodeError as e:
        st.sidebar.error(f"JSON 파일 파싱 오류: {e}")
        st.error("JSON 파일 형식이 올바르지 않습니다. 파일 형식을 확인하세요.")
        return [], [], {}

# 새로운 함수: 데이터 전처리 및 벡터화 (최초 1회만 실행)
def preprocess_data(court_cases, tax_cases):
    """데이터 전처리 및 벡터화 - 검색에 필요한 모든 정보를 미리 준비"""
    result = {
        "court_corpus": [],
        "tax_corpus": [],
        "court_chunks": [],
        "court_vectorizers": [],
        "court_tfidf_matrices": [],
        "tax_chunks": [],
        "tax_vectorizers": [],
        "tax_tfidf_matrices": []
    }
    
    # 1. 판례 데이터 분할 및 전처리
    court_chunks = split_court_cases(court_cases)
    result["court_chunks"] = court_chunks

    # 불용어 정의
    LEGAL_STOPWORDS = [
        # 기본 불용어
        '것', '등', '때', '경우', '바', '수', '점', '면', '이', '그', '저', '은', '는', '을', '를', '에', '의', '으로', 
        '따라', '또는', '및', '있다', '한다', '되어', '인한', '대한', '관한', '위한', '통한', '같은', '다른',
        
        # 법령 구조 불용어
        '조항', '규정', '법률', '법령', '조문', '항목', '세부', '내용', '사항', '요건', '기준', '방법', '절차',
        
        # 일반적인 동사/형용사
        '해당', '관련', '포함', '제외', '적용', '시행', '준용', '의하다', '하다', '되다', '있다', '없다', '같다'
    ]
    
    # 2. 각 판례 데이터 청크별 전처리 및 벡터화
    for chunk in court_chunks:
        court_corpus = []
        for item in chunk:
            text = extract_text_from_item(item, "court_case")
            court_corpus.append(preprocess_text(text))
        
        result["court_corpus"].append(court_corpus)
        
        if court_corpus:
            court_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words=LEGAL_STOPWORDS,
                min_df=1,
                max_df=0.8,
                sublinear_tf=True,
                use_idf=True,
                smooth_idf=True,
                norm='l2'
            )
            court_tfidf_matrix = court_vectorizer.fit_transform(court_corpus)
            result["court_vectorizers"].append(court_vectorizer)
            result["court_tfidf_matrices"].append(court_tfidf_matrix)
    
    # 3. 국가법령정보센터 관세판례 데이터 분할
    tax_chunks = split_tax_cases(tax_cases)
    result["tax_chunks"] = tax_chunks
    
    # 4. 각 국가법령정보센터 관세판례 청크별 전처리 및 벡터화
    for chunk in tax_chunks:
        tax_corpus = []
        for item in chunk:
            text = extract_text_from_item(item, "tax_case")
            tax_corpus.append(preprocess_text(text))
        
        result["tax_corpus"].append(tax_corpus)
        
        if tax_corpus:
            tax_vectorizer = TfidfVectorizer()
            tax_tfidf_matrix = tax_vectorizer.fit_transform(tax_corpus)
            result["tax_vectorizers"].append(tax_vectorizer)
            result["tax_tfidf_matrices"].append(tax_tfidf_matrix)
    
    logging.info("데이터 전처리 및 벡터화 완료")
    return result

# 추가된 함수: 법원 판례 데이터를 2개의 청크로 분할
def split_court_cases(court_cases):
    """관세분야판례423개를 2개의 청크로 분할"""
    # 데이터 개수
    total_cases = len(court_cases)
    chunk_size = max(1, total_cases // 2)  # 최소 1개는 되도록
    
    # 2개의 청크로 분할
    chunks = []
    for i in range(2):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < 1 else total_cases
        chunks.append(court_cases[start_idx:end_idx])
    
    # 분할 정보 로그
    st.sidebar.info(f"관세분야판례 분할: 총 {total_cases}건을 {[len(chunk) for chunk in chunks]}건씩 배분")
    
    return chunks

def split_tax_cases(tax_cases):
    """국가법령정보센터_관세판례를 4개의 청크로 분할"""
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
    st.sidebar.info(f"국가법령정보센터 판례 분할: 총 {total_cases}건을 {[len(chunk) for chunk in chunks]}건씩 배분")
    
    return chunks

# 관련성 높은 데이터 검색 함수 - 최적화 버전
def search_relevant_data(data, query, data_type, preprocessed_data, chunk_idx=None, top_n=10, conversation_history=""):
    """질문과 관련성이 높은 데이터 항목을 검색 (미리 벡터화된 데이터 활용)"""
    if not data:
        return []

    # 쿼리 전처리
    enhanced_query = query
    if conversation_history:
        enhanced_query = f"{query} {conversation_history}"
    
    enhanced_query = preprocess_text(enhanced_query)
    
    try:
        if data_type == "court_case":
            # 판례 데이터 검색 (특정 청크)
            vectorizer = preprocessed_data["court_vectorizers"][chunk_idx]
            tfidf_matrix = preprocessed_data["court_tfidf_matrices"][chunk_idx]
        else:  # tax_case
            # 국가법령정보센터 관세판례 검색 (특정 청크)
            vectorizer = preprocessed_data["tax_vectorizers"][chunk_idx]
            tfidf_matrix = preprocessed_data["tax_tfidf_matrices"][chunk_idx]
        
        # 쿼리 벡터화
        query_vec = vectorizer.transform([enhanced_query])
        
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
        return base_prompt + "\n# 판례 데이터를 기반으로 응답하세요. 모르면 모른다고 하세요."
    elif agent_type == "tax_case":
        return base_prompt + "\n# 판례 데이터를 기반으로 응답하세요. 모르면 모른다고 하세요."
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
    5. 이전 대화 맥락을 고려하여 답변 작성
- 모든 답변은 두괄식으로 작성합니다.
- 이전 대화에서 언급된 내용이 있다면 그것을 기억하고 관련 내용을 참조하여 응답합니다.
"""

# 에이전트 실행 함수 - 최적화 버전
def run_agent(agent_type, data, user_query, preprocessed_data, agent_index=None, chunk_idx=None, conversation_history=""):
    """특정 유형의 에이전트 실행 (최적화된 데이터 사용)"""
    # 프롬프트 생성
    prompt = get_agent_prompt(agent_type)
    
    # 질문과 관련성이 높은 데이터 검색 (미리 처리된 데이터 활용)
    data_type = "court_case" if agent_type == "court_case" else "tax_case"
    relevant_data = search_relevant_data(
        data, user_query, data_type, preprocessed_data, 
        chunk_idx=chunk_idx, conversation_history=conversation_history
    )
    
    # 관련 데이터가 없는 경우 처리
    if not relevant_data:
        agent_label = f"Agent {agent_index}" if agent_index else "Head Agent"
        return {
            "agent": agent_label,
            "response": "관련된 데이터를 찾을 수 없습니다."
        }
    
    # 데이터 문자열로 변환
    data_str = json.dumps(relevant_data, ensure_ascii=False, indent=2)
    
    # 대화 기록 추가 (에이전트별로 다르게 처리)
    context_str = ""
    if conversation_history:
        context_str = f"\n\n# 이전 대화 기록\n{conversation_history}"
    
    # 전체 프롬프트 구성
    full_prompt = f"{prompt}{context_str}\n\n# 데이터\n{data_str}\n\n# 질문\n{user_query}"
    logging.info(f"Agent {agent_index if agent_index else 'Head'} 실행 시작 (관련 데이터: {len(relevant_data)}건)")
    
    try:
        # Gemini 모델 호출 - gemini-2.0-flash 모델 사용
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                top_k=5,
                top_p=0.8
            )
        )
        
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

# 병렬 에이전트 실행 - 최적화 버전
def run_parallel_agents(court_cases, tax_cases, preprocessed_data, user_query, conversation_history=""):
    """모든 에이전트를 병렬로 실행하고 결과 반환 (최적화 버전)"""
    results = []
    
    try:
        # 미리 분할된 데이터 활용
        court_cases_chunks = preprocessed_data["court_chunks"]
        tax_cases_chunks = preprocessed_data["tax_chunks"]
        
        # ThreadPoolExecutor로 병렬 처리
        with ThreadPoolExecutor(max_workers=6) as executor:  # 에이전트 수 증가로 max_workers 조정
            # Agent 1-2: 판례 검색 (2개 청크)
            court_agent_futures = []
            for i, chunk in enumerate(court_cases_chunks, start=1):
                court_agent_futures.append(
                    executor.submit(
                        run_agent, "court_case", chunk, user_query, 
                        preprocessed_data, i, i-1, conversation_history
                    )
                )
            
            # Agent 3-6: 국가법령정보센터 관세판례 검색 (4개 청크)
            tax_agent_futures = []
            for i, chunk in enumerate(tax_cases_chunks, start=3):
                tax_agent_futures.append(
                    executor.submit(
                        run_agent, "tax_case", chunk, user_query, 
                        preprocessed_data, i, i-3, conversation_history
                    )
                )
            
            # 결과 수집
            for future in court_agent_futures:
                results.append(future.result())
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
def run_head_agent(agent_responses, user_query, conversation_history=""):
    """각 에이전트의 응답을 통합하여 최종 응답 생성"""
    # 응답 데이터 준비
    responses_str = ""
    for resp in agent_responses:
        responses_str += f"\n## {resp['agent']} 응답:\n{resp['response']}\n\n"
    
    # Head Agent 프롬프트 생성
    prompt = get_agent_prompt("head")
    
    # 대화 맥락 추가
    context_str = ""
    if conversation_history:
        context_str = f"\n\n# 이전 대화 기록\n{conversation_history}"
    
    full_prompt = f"{prompt}{context_str}\n\n# 에이전트 응답\n{responses_str}\n\n# 질문\n{user_query}\n\n# 지시사항\n위 에이전트들의 응답을 통합하여 사용자의 질문에 가장 적합한 최종 답변을 작성하세요. 이전 대화 맥락을 고려하여 일관성 있게 응답하세요."
    
    try:
        # Gemini 모델 호출
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                top_k=5,
                top_p=0.8
            )
        )
        
        
        logging.info("Head Agent 응답 생성 완료")
        return {
            "agent": "Head Agent",
            "response": response.text
        }

    except Exception as e:
        error_msg = f"Head Agent 오류 발생: {str(e)}"
        logging.error(error_msg)
        return error_msg