// main.cpp
#include "mlp_eigen.h"
#include <windows.h>
#include <objidl.h>
#include <gdiplus.h>
#include <memory>
#include <vector>
#include <iostream>

#pragma comment(lib, "gdiplus.lib")
using namespace Gdiplus;

// GDI+ RAII deleter
struct GdiPlusDeleter {
    void operator()(GdiplusStartupInput* p) {
        GdiplusShutdown(token);
        delete p;
    }
    ULONG_PTR token;
};

int GetEncoderClsid(const WCHAR* format, CLSID* pClsid) {
    UINT num = 0, size = 0;
    GetImageEncodersSize(&num, &size);
    if (!size) return -1;
    auto pInfo = (ImageCodecInfo*)malloc(size);
    GetImageEncoders(num, size, pInfo);
    for (UINT i = 0; i < num; ++i) {
        if (wcscmp(pInfo[i].MimeType, format) == 0) {
            *pClsid = pInfo[i].Clsid;
            free(pInfo);
            return i;
        }
    }
    free(pInfo);
    return -1;
}

void saveVisualization(const MLP_Eigen& mlp,
                       const std::vector<std::vector<double>>& data,
                       const std::vector<double>& labels,
                       const wchar_t* filename)
{
    // init GDI+
    auto input = std::make_unique<GdiplusStartupInput>();
    GdiPlusDeleter del{0};
    GdiplusStartup(&del.token, input.get(), nullptr);
    std::unique_ptr<GdiplusStartupInput, GdiPlusDeleter> init(input.release(), del);

    const int W = 300, H = 250, STEP = 5;
    Bitmap bmp(W, H);
    Graphics g(&bmp);
    g.Clear(Color::White);

    // decision boundary
    for (int x = 0; x < W; x += STEP) {
        for (int y = 0; y < H; y += STEP) {
            double ix = double(x)/W;
            double iy = double(y)/H;
            double o = mlp.forward(ix, iy);
            int r = int((1-o)*135 + o*255);
            int gg = int((1-o)*206 + o*105);
            int b = int((1-o)*235 + o*180);
            SolidBrush brush(Color(255, r, gg, b));
            g.FillRectangle(&brush, x, y, STEP, STEP);
        }
    }

    // training points
    const float R = 6.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        float px = float(data[i][0]*W);
        float py = float(data[i][1]*H);
        Color c = labels[i] == 1.0
                ? Color(255,255,0,0)
                : Color(255,0,0,255);
        SolidBrush brush(c);
        g.FillEllipse(&brush,
                      px-R, py-R,
                      R*2, R*2);
    }

    // save PNG
    CLSID clsid;
    GetEncoderClsid(L"image/png", &clsid);
    bmp.Save(filename, &clsid, nullptr);
}

int main() {
    // raw data
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

    // normalize
    for (auto &pt : data) {
        pt[0] /= 30.0;
        pt[1] /= 25.0;
    }

    MLP_Eigen mlp;
    mlp.train(data, labels, 2000, 0.04);
    saveVisualization(mlp, data, labels, L"visualized_c++.png");

    std::wcout << L"Train Accuracy: " 
               << int(mlp.getAccuracy(data, labels)*100) 
               << L"%\n";
    std::wcout << L"Train Time: " 
               << mlp.getTrainingTime() 
               << L" ms\n";
    return 0;
}
