#include "mlp.h"
#include <windows.h>
#include <objidl.h>
#include <gdiplus.h>
#include <memory>

#pragma comment(lib, "gdiplus.lib")

using namespace Gdiplus;

// GDI+ 리소스를 위한 커스텀 deleter
struct GdiPlusDeleter {
    void operator()(GdiplusStartupInput* input) { 
        GdiplusShutdown(gdiplusToken); 
        delete input;
    }
    ULONG_PTR gdiplusToken;
};

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

void saveVisualization(const MLP& mlp, 
                      const std::vector<std::vector<double>>& data,
                      const std::vector<double>& labels,
                      const wchar_t* filename) {
    // GDI+ 초기화를 RAII로 관리
    auto gdiplusInput = std::make_unique<GdiplusStartupInput>();
    GdiPlusDeleter deleter;
    GdiplusStartup(&deleter.gdiplusToken, gdiplusInput.get(), NULL);
    std::unique_ptr<GdiplusStartupInput, GdiPlusDeleter> gdiplus(gdiplusInput.release(), deleter);

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

            {
                SolidBrush brush(Color(255, r, g, b));
                graphics.FillRectangle(&brush, x, y, step, step);
            } // brush는 여기서 자동으로 해제됨
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
            
        {
            SolidBrush fillBrush(pointColor);
            float radius = 6.0f;
            graphics.FillEllipse(&fillBrush, x - radius, y - radius, radius * 2, radius * 2);
        } // fillBrush는 여기서 자동으로 해제됨
    }

    // PNG 저장
    CLSID pngClsid;
    GetEncoderClsid(L"image/png", &pngClsid);
    bitmap.Save(filename, &pngClsid, NULL);
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Size of double: " << sizeof(double) << " bytes" << std::endl;
    
    double test_value = 0.1;
    unsigned char* bytes = reinterpret_cast<unsigned char*>(&test_value);
    std::cout << "Internal representation of 0.1: ";
    for(size_t i = 0; i < sizeof(double); ++i) {
        std::cout << std::hex << static_cast<int>(bytes[i]) << " ";
    }
    std::cout << std::dec << std::endl;
    std::cout << "--------------------------------" << std::endl;
    
    // 학습 데이터 생성
    std::vector<std::vector<double>> data = {
        {11, 21}, {28, 17}, {19, 15}, {6, 9}, {7, 1}, {8, 10}, 
        {14, 21}, {15, 14}, {14, 17}, {15, 20}, {28, 21}, {9, 13}, 
        {19, 23}, {29, 11}, {23, 18}, {2, 19}, {27, 15}, {6, 6}, 
        {12, 18}, {0, 4}, {13, 14}, {3, 20}, {6, 2}, {3, 24}, 
        {9, 10}, {14, 10}, {8, 14}, {10, 6}, {13, 15}, {9, 16}, 
        {12, 7}, {20, 22}, {17, 16}, {19, 16}, {11, 18}, {19, 11}, 
        {0, 10}, {13, 16}, {16, 24}, {23, 15}, {12, 18}, {0, 0}, 
        {22, 12}, {8, 13}, {25, 9}, {26, 21}, {6, 16}, {28, 16}, 
        {15, 6}, {4, 10}, {26, 5}, {6, 8}, {15, 14}, {18, 7}, 
        {3, 24}, {1, 15}, {9, 16}, {4, 23}, {2, 20}, {15, 12}, 
        {29, 11}, {13, 19}, {23, 22}, {23, 1}, {14, 17}, {9, 24}, 
        {16, 12}, {21, 17}, {9, 7}, {21, 16}, {21, 12}, {12, 24}, 
        {18, 13}, {16, 8}, {6, 1}, {9, 7}, {18, 5}, {8, 14}, 
        {12, 21}, {8, 22}
    };

    std::vector<double> labels = {
        1, 1, 0, 0, 0, 0, 
        1, 0, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 0, 
        1, 0, 1, 1, 0, 1, 
        0, 0, 1, 0, 1, 1, 
        0, 1, 0, 0, 1, 0, 
        0, 1, 1, 1, 1, 0, 
        0, 1, 0, 1, 1, 1, 
        0, 0, 0, 0, 0, 0, 
        1, 0, 1, 1, 1, 0, 
        1, 1, 1, 0, 1, 1, 
        0, 1, 0, 1, 0, 1, 
        0, 0, 0, 0, 0, 1, 
        1, 1
    };

    // 학습 전 데이터 정규화
    for (auto& point : data) {
        point[0] /= 30.0;  // x축: 0~30 기준
        point[1] /= 25.0;  // y축: 0~25 기준
    }

    // MLP 모델 생성 및 학습
    MLP mlp;
    mlp.train(data, labels, 2000, 0.05);  // Python과 동일한 파라미터
    
    // 결과를 이미지 파일로 저장
    saveVisualization(mlp, data, labels, L"visualized_c++.png");
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Saved Result: visualized_c++.png" << std::endl;
    std::cout << "Train Accuracy: " << 100 * mlp.getAccuracy(data, labels) << "%" << std::endl;
    std::cout << "Train Time: " << mlp.getTrainingTime() << "ms" << std::endl;
    std::cout << "Overhead: " << total_time - mlp.getTrainingTime() << "ms" << std::endl;
    std::cout << "Total Time: " << total_time << "ms" << std::endl;
    std::cout << std::endl;
    return 0;
}
