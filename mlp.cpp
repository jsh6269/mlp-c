#include "mlp.h"
#include <random>
#include <cmath>

namespace {
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoid_derivative(double x) {
        double sig = sigmoid(x);
        return sig * (1.0 - sig);
    }
}

MLP::MLP() 
    : hidden_size(4),
      input_weights(8), 
      hidden_weights(4), 
      input_bias(0), 
      hidden_bias(0) {
    // 가중치 초기화
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    
    for(auto& w : input_weights) w = d(gen) * 0.1;
    for(auto& w : hidden_weights) w = d(gen) * 0.1;
    input_bias = d(gen) * 0.1;
    hidden_bias = d(gen) * 0.1;
}

double MLP::forward(double x, double y) const {
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

void MLP::train(const std::vector<std::vector<double>>& data, 
                const std::vector<double>& labels,
                int epochs, 
                double learning_rate) {
    auto start = std::chrono::high_resolution_clock::now();
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
            std::cout << "Epoch " << epoch << ", "
                      << "Loss: " << total_loss/data.size() << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    training_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

double MLP::getAccuracy(const std::vector<std::vector<double>>& data, 
                       const std::vector<double>& labels) const {
    int correct = 0;
    for(size_t i = 0; i < data.size(); i++) {
        double x = data[i][0];
        double y = data[i][1];
        double target = labels[i];
        double output = forward(x, y);
        if(std::abs(output - target) < 0.5) correct++;
    }
    return static_cast<double>(correct) / data.size();
}

double MLP::getTrainingTime() const {
    return training_time;
}
