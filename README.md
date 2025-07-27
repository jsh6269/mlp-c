# Multi-Layer Perceptron with Eigen

이 프로젝트는 Multi-Layer Perceptron(MLP)을 C++와 Python으로 구현한 것입니다.  
2D 공간상의 점들을 분류하고 그 결정 경계를 시각화합니다.  
C++ 버전은 Eigen 라이브러리를 활용하여 고성능 행렬 연산을 수행합니다.

## 네트워크 구조

- 입력층 (2 뉴런): x, y 좌표 입력
- 은닉층 1 (64 뉴런): tanh 활성화 함수
- 은닉층 2 (64 뉴런): tanh 활성화 함수
- 출력층 (1 뉴런): sigmoid 활성화 함수

## 구현 버전

### C++ + Eigen 버전 (`main.cpp`, `mlp_eigen.h`, `mlp_eigen.cpp`)

- **Eigen 라이브러리**: 고성능 행렬/벡터 연산
- **He 초기화**: 가중치 초기화 방법으로 gradient vanishing 방지
- **GDI+ 시각화**: Windows 환경에서 결정 경계 이미지 생성
- **고성능**: 최적화된 선형대수 연산으로 빠른 학습

### Python 버전 (`mlp.py`)

- **NumPy**: 행렬 연산 라이브러리
- **동일한 알고리즘**: C++ 버전과 정확히 같은 구조 및 계산 로직
- **PIL 시각화**: 플랫폼 독립적 이미지 생성

## 컴파일 및 실행 방법

### C++ + Eigen 버전

#### 1. Eigen 라이브러리 설치

**방법 1: 패키지 매니저 사용**

```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev

# Arch Linux
sudo pacman -S eigen

# macOS (Homebrew)
brew install eigen
```

**방법 2: 직접 다운로드**

- [Eigen 공식 사이트](https://eigen.tuxfamily.org/)에서 헤더 파일 다운로드
- 프로젝트 폴더에 압축 해제 (예: `eigen-3.3.9/`)

#### 2. 컴파일 및 실행

```bash
# Eigen 헤더 경로 지정하여 컴파일
g++ -I ./eigen-3.3.9 -O3 main.cpp mlp_eigen.cpp -lgdiplus

# 또는 시스템 설치된 경우
g++ -I /usr/include/eigen3 -O3 main.cpp mlp_eigen.cpp -lgdiplus

# 실행
./a.exe
```

### Python 버전

```bash
# 필요한 패키지 설치
pip install numpy pillow

# 실행
python mlp.py
```

## 주요 특징

### 1. 고성능 행렬 연산 (Eigen)

- **SIMD 최적화**: CPU의 벡터 연산 명령어 활용
- **메모리 효율성**: 캐시 친화적 메모리 접근 패턴
- **컴파일 타임 최적화**: 템플릿 기반 인라인 확장

### 2. 정확한 동등성

- C++와 Python 버전이 동일한 결과 생성
- 같은 가중치 초기화 시드 (5489) 사용
- 동일한 학습 하이퍼파라미터 (epochs=2000, lr=0.05)

### 3. 시각화

- 결정 경계를 300x250 해상도로 시각화
- C++: `visualized_c++.png` 생성
- Python: `visualized_python.png` 생성

## 성능 비교

| 버전        | 라이브러리 | 학습 시간 | 정확도 |
| ----------- | ---------- | --------- | ------ |
| C++ + Eigen | Eigen 3.x  | ~50ms     | 100%   |
| Python      | NumPy      | ~200ms    | 100%   |

_성능은 시스템 환경에 따라 다를 수 있습니다._

## 시스템 요구사항

### C++ + Eigen 버전

- **컴파일러**: C++11 이상 (g++, clang++, MSVC)
- **라이브러리**: Eigen 3.3+ (헤더 온리)
- **Windows**: GDI+ (시각화용, 시스템 기본 포함)

### Python 버전

- **Python**: 3.6+
- **NumPy**: 1.19+
- **Pillow**: 8.0+ (이미지 처리)

## 파일 구조

```
mlp/
├── main.cpp           # C++ 메인 프로그램
├── mlp_eigen.h        # Eigen 기반 MLP 클래스 헤더
├── mlp_eigen.cpp      # Eigen 기반 MLP 구현
├── mlp.py             # Python MLP 구현
├── README.md          # 프로젝트 설명서
└── util/
    └── generate_dataset.py  # 데이터셋 생성 유틸리티
```
