#include <windows.h>
#include <objidl.h>
#include <gdiplus.h>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <memory>

#pragma comment(lib, "gdiplus.lib")

using namespace Gdiplus;

// CLSID를 문자열로 변환하는 헬퍼 함수
int GetEncoderClsid(const WCHAR* format, CLSID* pClsid) {
    UINT num = 0;
    UINT size = 0;

    ImageCodecInfo* pImageCodecInfo = NULL;

    GetImageEncodersSize(&num, &size);
    if(size == 0) return -1;

    pImageCodecInfo = (ImageCodecInfo*)(malloc(size));
    if(pImageCodecInfo == NULL) return -1;

    GetImageEncoders(num, size, pImageCodecInfo);

    for(UINT j = 0; j < num; ++j) {
        if(wcscmp(pImageCodecInfo[j].MimeType, format) == 0) {
            *pClsid = pImageCodecInfo[j].Clsid;
            free(pImageCodecInfo);
            return j;
        }
    }

    free(pImageCodecInfo);
    return -1;
}

// 활성화 함수
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

class MLP {
private:
    std::vector<double> input_weights;    // 입력층 -> 은닉층 가중치
    std::vector<double> hidden_weights;   // 은닉층 -> 출력층 가중치
    double input_bias;                    // 입력층 편향
    double hidden_bias;                   // 은닉층 편향
    const int hidden_size = 4;            // 은닉층 뉴런 수
    
public:
    MLP() : input_weights(8), hidden_weights(4), input_bias(0), hidden_bias(0) {
        // 가중치 초기화
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 1);
        
        for(auto& w : input_weights) w = d(gen) * 0.1;
        for(auto& w : hidden_weights) w = d(gen) * 0.1;
        input_bias = d(gen) * 0.1;
        hidden_bias = d(gen) * 0.1;
    }
    
    double forward(double x, double y) const {
        std::vector<double> hidden(hidden_size);
        
        // 입력층 -> 은닉층
        for(int i = 0; i < hidden_size; i++) {
            hidden[i] = sigmoid(x * input_weights[i] + 
                              y * input_weights[i + hidden_size] + 
                              input_bias);
        }
        
        // 은닉층 -> 출력층
        double output = 0;
        for(int i = 0; i < hidden_size; i++) {
            output += hidden[i] * hidden_weights[i];
        }
        output = sigmoid(output + hidden_bias);
        
        return output;
    }
    
    void train(const std::vector<std::vector<double>>& data, 
              const std::vector<double>& labels,
              int epochs = 1000, 
              double learning_rate = 0.1) {
        for(int epoch = 0; epoch < epochs; epoch++) {
            double total_loss = 0;
            
            for(size_t i = 0; i < data.size(); i++) {
                double x = data[i][0];
                double y = data[i][1];
                double target = labels[i];
                
                // Forward pass
                std::vector<double> hidden(hidden_size);
                std::vector<double> hidden_raw(hidden_size);
                
                for(int j = 0; j < hidden_size; j++) {
                    hidden_raw[j] = x * input_weights[j] + 
                                  y * input_weights[j + hidden_size] + 
                                  input_bias;
                    hidden[j] = sigmoid(hidden_raw[j]);
                }
                
                double output_raw = 0;
                for(int j = 0; j < hidden_size; j++) {
                    output_raw += hidden[j] * hidden_weights[j];
                }
                output_raw += hidden_bias;
                double output = sigmoid(output_raw);
                
                // 손실 계산
                double loss = 0.5 * (target - output) * (target - output);
                total_loss += loss;
                
                // Backward pass
                double d_output = (output - target) * sigmoid_derivative(output_raw);
                
                // 은닉층 -> 출력층 가중치 업데이트
                for(int j = 0; j < hidden_size; j++) {
                    hidden_weights[j] -= learning_rate * d_output * hidden[j];
                }
                hidden_bias -= learning_rate * d_output;
                
                // 입력층 -> 은닉층 가중치 업데이트
                for(int j = 0; j < hidden_size; j++) {
                    double d_hidden = d_output * hidden_weights[j] * 
                                    sigmoid_derivative(hidden_raw[j]);
                    input_weights[j] -= learning_rate * d_hidden * x;
                    input_weights[j + hidden_size] -= learning_rate * d_hidden * y;
                }
                input_bias -= learning_rate * d_output;
            }
            
            if(epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " 
                         << total_loss/data.size() << std::endl;
            }
        }
    }
};

void saveVisualization(const MLP& mlp, 
                      const std::vector<std::vector<double>>& data,
                      const std::vector<double>& labels,
                      const wchar_t* filename) {
    // GDI+ 초기화
    GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;
    GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

    const int width = 300;
    const int height = 250;
    const int step = 5;
    Bitmap bitmap(width, height);
    Graphics graphics(&bitmap);
    graphics.Clear(Color::White);

    // 결정 경계 시각화 (배경 먼저)
    for (int x = 0; x < width; x += step) {
        for (int y = 0; y < height; y += step) {
            double input_x = x / static_cast<double>(width);
            double input_y = y / static_cast<double>(height);
            double output = mlp.forward(input_x, input_y);

            int r = static_cast<int>((1 - output) * 135 + output * 255);
            int g = static_cast<int>((1 - output) * 206 + output * 105);
            int b = static_cast<int>((1 - output) * 235 + output * 180);

            SolidBrush brush(Color(255, r, g, b));
            graphics.FillRectangle(&brush, x, y, step, step);
        }
    }

    // 학습 데이터 포인트 그리기 (배경 위에 나중에 그리기)
    for (size_t i = 0; i < data.size(); i++) {
        float x = static_cast<float>(data[i][0] * width);
        float y = static_cast<float>(data[i][1] * height);

        // 포인트 색상 (불투명 빨강 or 파랑)
        Color pointColor = (labels[i] == 1) 
            ? Color(255, 255, 0, 0)
            : Color(255, 0, 0, 255);
        SolidBrush fillBrush(pointColor);

        float radius = 6.0f;
        graphics.FillEllipse(&fillBrush, x - radius, y - radius, radius * 2, radius * 2);
    }

    // PNG 저장
    CLSID pngClsid;
    GetEncoderClsid(L"image/png", &pngClsid);
    bitmap.Save(filename, &pngClsid, NULL);
    GdiplusShutdown(gdiplusToken);
}

int main() {
    // 학습 데이터 생성
    std::vector<std::vector<double>> data = {
        {10, 23}, {4, 21}, {8, 17}, {17, 22}, {3, 12}, {9, 12}, {16, 14}, {21, 20}, {26, 22}, // upper
        {4, 4}, {13, 7}, {16, 6}, {14, 3}, {22, 8}, {24, 5}, {28, 6}, {27, 14}               // lower
    };

    std::vector<double> labels = {
        1, 1, 1, 1, 1, 1, 1, 1, 1,   // upper
        0, 0, 0, 0, 0, 0, 0, 0      // lower
    };

    // 학습 전 데이터 정규화
    for (auto& point : data) {
        point[0] /= 30.0;  // x축: 0~30 기준
        point[1] /= 25.0;  // y축: 0~25 기준
    }

    // MLP 모델 생성 및 학습
    MLP mlp;
    mlp.train(data, labels);
    
    // 결과를 이미지 파일로 저장
    saveVisualization(mlp, data, labels, L"mlp_visualization.png");
    
    std::cout << "시각화가 'mlp_visualization.png' 파일로 저장되었습니다." << std::endl;
    
    return 0;
}
