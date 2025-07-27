#ifndef MLP_H
#define MLP_H

#include <vector>
#include <iostream>
#include <chrono>

class MLP {
private:
    const int hidden_size1;                           // 64
    const int hidden_size2;                           // 64
    std::vector<std::vector<double>> input_weights;   // (hidden_size1, 2)
    std::vector<std::vector<double>> hidden_weights;  // (hidden_size2, hidden_size1)
    std::vector<double> output_weights;               // (hidden_size2,)
    std::vector<double> input_biases;                 // (hidden_size1,)
    std::vector<double> hidden_biases;                // (hidden_size2,)
    double output_bias;                               // scalar
    double training_time;                             // 학습 시간

    // Helper functions
    std::vector<double> matmul(const std::vector<std::vector<double>>& A, 
                              const std::vector<double>& x) const;
    std::vector<std::vector<double>> outer(const std::vector<double>& a, 
                                         const std::vector<double>& b) const;
    double sigmoid(double x) const;
    double sigmoid_derivative(double x) const;
    double tanh(double x) const;
    double tanh_derivative(double x) const;

public:
    MLP();

    double forward(double x, double y) const;

    void train(const std::vector<std::vector<double>>& data, 
               const std::vector<double>& labels,
               int epochs = 2000, 
               double learning_rate = 0.05);

    double getAccuracy(const std::vector<std::vector<double>>& data, 
                      const std::vector<double>& labels) const;

    double getTrainingTime() const;
};

#endif
