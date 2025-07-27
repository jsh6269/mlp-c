# Simple Multi-Layer Perceptron

이 프로젝트는 간단한 Multi-Layer Perceptron(MLP)을 C++와 Python으로 각각 구현한 것입니다.  
2D 공간상의 점들을 분류하고 그 결정 경계를 시각화합니다.  
cpu 기준으로 C++이 Python보다 얼마나 더 빠른지 알아봅시다~!

## 구조

- 입력층 (2 뉴런): x, y 좌표 입력
- 은닉층 (4 뉴런): sigmoid 활성화 함수 사용
- 출력층 (1 뉴런): sigmoid 활성화 함수 사용

## 구현 버전

### C++ 버전 (`main.cpp`, `mlp.h`, `mlp.cpp`)

- GDI+ 사용하여 시각화
- Windows 환경에서 실행 가능
- 컴파일 및 실행이 빠름 (약 26ms)

### Python 버전 (`mlp.py`)

- PIL(Python Imaging Library) 사용하여 시각화
- 플랫폼 독립적
- 구현이 직관적이지만 실행 속도가 상대적으로 느림 (약 400ms)

## 컴파일 및 실행 방법

### C++ 버전

```bash
# MinGW g++ 사용
g++ main.cpp mlp.cpp -lgdiplu

# 실행
./a
```

### Python 버전

```bash
# 필요한 패키지 설치
pip install numpy pillow

# 실행
python mlp.py
```

## 결과

두 버전 모두:

- 학습 데이터에 대해 100% 정확도 달성
- `visualized.png` 파일로 결정 경계 시각화 저장
- 매 100 에포크마다 loss 출력

## 특이사항

1. 정밀도

   - 두 버전 모두 64비트 부동소수점(C++: double, Python: float64) 사용
   - 연산 과정의 차이로 인해 loss 값에 미세한 차이 존재

2. 성능 차이
   - C++ 버전이 Python 버전보다 약 15배 빠름
   - 컴파일된 언어와 인터프리터 언어의 특성 차이

## 시스템 요구사항

### C++ 버전

- Windows 운영체제
- MinGW g++ 컴파일러
- GDI+ 라이브러리 (Windows에 기본 포함)

### Python 버전

- Python 3.x
- NumPy
- Pillow (PIL)
