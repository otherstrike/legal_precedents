# 관세법 판례 기반 챗봇 (Customs Law Case-Based Chatbot)

## 프로젝트 개요

이 프로젝트는 관세법 판례 데이터를 기반으로 한 AI 법률 챗봇입니다. Google의 Gemini 2.0 Flash 모델을 활용하여 사용자의 법률 관련 질문에 대해 전문적인 답변을 제공합니다.

## 주요 기능

- **관세법 판례 및 조세심판 결정례 검색**: 사용자의 질문과 관련성이 높은 판례 및 결정례를 검색합니다.
- **다중 에이전트 아키텍처**: 여러 AI 에이전트를 병렬로 실행하여 다양한 데이터를 분석합니다.
- **맥락 기반 대화**: 이전 대화를 고려하여 일관성 있는 응답을 생성합니다.
- **직관적인 인터페이스**: Streamlit을 통해 사용하기 쉬운 웹 인터페이스를 제공합니다.

## 시스템 아키텍처

이 챗봇은 다음과 같은 멀티 에이전트 아키텍처를 기반으로 합니다:

1. **관세법령 관세법 판례 기반 챗봇 (Customs Law Case-Based Chatbot)

## 프로젝트 개요

이 프로젝트는 관세법 판례 데이터를 기반으로 한 AI 법률 챗봇입니다. Google의 Gemini 2.0 Flash 모델을 활용하여 사용자의 법률 관련 질문에 대해 전문적인 답변을 제공합니다.

## 주요 기능

- **관세법 판례 및 조세심판 결정례 검색**: 사용자의 질문과 관련성이 높은 판례 및 결정례를 검색합니다.
- **다중 에이전트 아키텍처**: 여러 AI 에이전트를 병렬로 실행하여 다양한 데이터를 분석합니다.
- **맥락 기반 대화**: 이전 대화를 고려하여 일관성 있는 응답을 생성합니다.
- **직관적인 인터페이스**: Streamlit을 통해 사용하기 쉬운 웹 인터페이스를 제공합니다.

## 시스템 아키텍처

이 챗봇은 다음과 같은 멀티 에이전트 아키텍처를 기반으로 합니다:

1. **관세법령정보포털 판례 분석 에이전트 (Agent 1~2)**: 관세법 판례 데이터를 분석합니다.
2. **국가법령정보센터 관세분야 판례 분석 에이전트 (Agent 3~6)**: 조세심판 결정례 데이터를 분할하여 병렬로 분석합니다.
3. **통합 에이전트 (Head Agent)**: 각 에이전트의 결과를 통합하여 최종 응답을 생성합니다.

## 설치 및 실행 방법

### 필수 요구사항

- Python 3.7 이상
- Google API 키 (Gemini 2.0 Flash 모델 접근 권한)

### 설치 단계

1. 저장소 클론:
   ```bash
   git clone https://github.com/YSCHOI-github/legal_precedents.git
   cd legal_precedents
   ```

2. 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

3. 필요한 데이터 파일 준비:
   - `관세분야판례423개.json` - 관세분야 판례 데이터
   - `국가법령정보센터_관세판례.json` - 관세분야 판례 데이터

### 실행 방법

```bash
streamlit run main.py
```

## 사용 방법

1. 웹 브라우저에서 Streamlit 앱이 실행되면 사이드바에 Google API 키를 입력합니다.
2. API 키를 저장한 후 질문 입력창에 관세법 관련 질문을 입력합니다.
3. 챗봇이 질문과 관련된 판례 및 결정례를 분석하여 답변을 생성합니다.

## 질문 예시

- 관세경정거부처분이란 무엇인가요?
- HSK 분류와 관련된 주요 판례는?
- 품목분류 관련 주요 법적 쟁점은?
- 관세법 제42조의 가산세 면제 조건은?

## 주요 구성 요소

### main.py
- Streamlit 웹 애플리케이션의 주요 파일
- 사용자 인터페이스 및 상호작용 로직 포함
- 대화 기록 관리 및 세션 관리

### utils.py
- 데이터 처리 및 에이전트 실행 관련 기능
- API 초기화 및 연결
- 텍스트 유사도 기반 검색 기능
- 병렬 처리를 위한 에이전트 실행 로직

## 기술적 특징

- **텍스트 유사도 분석**: TF-IDF 벡터화와 코사인 유사도를 사용하여 관련 데이터를 검색합니다.
- **병렬 처리**: ThreadPoolExecutor를 사용하여 다수의 에이전트를 병렬로 실행합니다.
- **대화 맥락 관리**: 사용자가 선택한 대화 맥락 유지 수에 따라 이전 대화 기록을 활용합니다.
- **오류 처리**: 다양한 예외 상황에 대한 견고한 오류 처리 메커니즘을 포함합니다.

## 데이터 소스

- **관세분야판례423개.json**: 423개의 관세법 관련 판례를 포함합니다.
- **조세심판결정례_final.zip**: 조세심판 결정례 데이터를 압축 파일 형태로 포함합니다.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

```
MIT License

Copyright (c) 2025 YSCHOI-github

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
