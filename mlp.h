#ifndef MLP_H
#define MLP_H

#include <vector>
#include <iostream>
#include <chrono>

class MLP {
private:
    std::vector<double> input_weights;    // 입력층 -> 은닉층 가중치
    std::vector<double> hidden_weights;   // 은닉층 -> 출력층 가중치
    double input_bias;                    // 입력층 편향
    double hidden_bias;                   // 은닉층 편향
    const int hidden_size;                // 은닉층 뉴런 수
    double training_time;                 // 학습 시간

public:
    MLP();

    double forward(double x, double y) const;

    void train(const std::vector<std::vector<double>>& data, 
               const std::vector<double>& labels,
               int epochs = 1000, 
               double learning_rate = 0.1);

    double getAccuracy(const std::vector<std::vector<double>>& data, 
                       const std::vector<double>& labels) const;

    double getTrainingTime() const;
};

#endif
