// mlp.h
#ifndef MLP_H
#define MLP_H

#include <vector>
#include <array>
#include <chrono>
#include <cmath>

class MLP {
public:
    MLP();

    // x, y ∈ [0,1]
    double forward(double x, double y) const;

    // data: N×2 배열, labels: 0 or 1
    void train(const std::vector<std::vector<double>>& data,
               const std::vector<double>& labels,
               int epochs = 2000,
               double learning_rate = 0.05);

    double getAccuracy(const std::vector<std::vector<double>>& data,
                       const std::vector<double>& labels) const;

    // 마지막 train() 수행 시간 (ms)
    int getTrainingTime() const;

private:
    const int hidden_size1 = 64;
    const int hidden_size2 = 64;

    // weights & biases
    std::vector<double> input_weights;    // [hidden_size1 × 2]
    std::vector<double> hidden_weights;   // [hidden_size2 × hidden_size1]
    std::vector<double> output_weights;   // [hidden_size2]
    std::vector<double> input_biases;     // [hidden_size1]
    std::vector<double> hidden_biases;    // [hidden_size2]
    double output_bias;

    int training_time_ms;

    // 순전파·역전파 임시 버퍼
    mutable std::vector<double> hidden1_raw, hidden1;
    mutable std::vector<double> hidden2_raw, hidden2;
    mutable std::vector<double> d_hidden1, d_hidden2;

    // 활성화 함수
    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    static double sigmoid_derivative_from_raw(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }
    static double tanh_fn(double x) {
        return std::tanh(x);
    }
    static double tanh_derivative_from_raw(double x) {
        double t = std::tanh(x);
        return 1.0 - t*t;
    }
};

#endif // MLP_H
