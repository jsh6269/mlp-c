#include "mlp.h"
#include <random>
#include <cmath>

double MLP::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double MLP::sigmoid_derivative(double x) const {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

double MLP::tanh(double x) const {
    return std::tanh(x);
}

double MLP::tanh_derivative(double x) const {
    double t = std::tanh(x);
    return 1.0 - t * t;
}

// Matrix multiplication: A @ x where A is matrix and x is vector
std::vector<double> MLP::matmul(const std::vector<std::vector<double>>& A,
                               const std::vector<double>& x) const {
    std::vector<double> result(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < x.size(); j++) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

// Outer product: a[:, None] @ b[None, :] in numpy
std::vector<std::vector<double>> MLP::outer(const std::vector<double>& a,
                                          const std::vector<double>& b) const {
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(b.size()));
    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < b.size(); j++) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

MLP::MLP() 
    : hidden_size1(64),
      hidden_size2(64),
      input_weights(hidden_size1, std::vector<double>(2)),
      hidden_weights(hidden_size2, std::vector<double>(hidden_size1)),
      output_weights(hidden_size2),
      input_biases(hidden_size1),
      hidden_biases(hidden_size2),
      output_bias(0.0) {
    
    std::mt19937 gen(5489);  // Python과 동일한 시드값 사용
    
    double scale_input = std::sqrt(2.0 / 2);
    double scale_h = std::sqrt(2.0 / hidden_size1);
    double scale_out = std::sqrt(2.0 / hidden_size2);
    
    // Initialize input_weights (hidden_size1, 2)
    std::normal_distribution<> d_input(0, scale_input);
    for (auto& row : input_weights) {
        for (auto& w : row) {
            w = d_input(gen);
        }
    }
    
    // Initialize hidden_weights (hidden_size2, hidden_size1)
    std::normal_distribution<> d_h(0, scale_h);
    for (auto& row : hidden_weights) {
        for (auto& w : row) {
            w = d_h(gen);
        }
    }
    
    // Initialize output_weights (hidden_size2,)
    std::normal_distribution<> d_out(0, scale_out);
    for (auto& w : output_weights) {
        w = d_out(gen);
    }
    
    // Initialize biases to zeros
    std::fill(input_biases.begin(), input_biases.end(), 0.0);
    std::fill(hidden_biases.begin(), hidden_biases.end(), 0.0);
}

double MLP::forward(double x, double y) const {
    std::vector<double> input_vec = {x, y};
    
    // First hidden layer
    auto hidden1_raw = matmul(input_weights, input_vec);
    for (size_t i = 0; i < hidden1_raw.size(); i++) {
        hidden1_raw[i] += input_biases[i];
    }
    std::vector<double> hidden1(hidden1_raw.size());
    for (size_t i = 0; i < hidden1.size(); i++) {
        hidden1[i] = tanh(hidden1_raw[i]);
    }
    
    // Second hidden layer
    auto hidden2_raw = matmul(hidden_weights, hidden1);
    for (size_t i = 0; i < hidden2_raw.size(); i++) {
        hidden2_raw[i] += hidden_biases[i];
    }
    std::vector<double> hidden2(hidden2_raw.size());
    for (size_t i = 0; i < hidden2.size(); i++) {
        hidden2[i] = tanh(hidden2_raw[i]);
    }
    
    // Output layer
    double output_raw = 0.0;
    for (size_t i = 0; i < hidden_size2; i++) {
        output_raw += output_weights[i] * hidden2[i];
    }
    output_raw += output_bias;
    
    return sigmoid(output_raw);
}

void MLP::train(const std::vector<std::vector<double>>& data,
                const std::vector<double>& labels,
                int epochs,
                double learning_rate) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        
        for (size_t idx = 0; idx < data.size(); idx++) {
            std::vector<double> input_vec = {data[idx][0], data[idx][1]};
            double target = labels[idx];
            
            // Forward pass
            auto hidden1_raw = matmul(input_weights, input_vec);
            for (size_t i = 0; i < hidden1_raw.size(); i++) {
                hidden1_raw[i] += input_biases[i];
            }
            std::vector<double> hidden1(hidden1_raw.size());
            for (size_t i = 0; i < hidden1.size(); i++) {
                hidden1[i] = tanh(hidden1_raw[i]);
            }
            
            auto hidden2_raw = matmul(hidden_weights, hidden1);
            for (size_t i = 0; i < hidden2_raw.size(); i++) {
                hidden2_raw[i] += hidden_biases[i];
            }
            std::vector<double> hidden2(hidden2_raw.size());
            for (size_t i = 0; i < hidden2.size(); i++) {
                hidden2[i] = tanh(hidden2_raw[i]);
            }
            
            double output_raw = 0.0;
            for (size_t i = 0; i < hidden_size2; i++) {
                output_raw += output_weights[i] * hidden2[i];
            }
            output_raw += output_bias;
            double output = sigmoid(output_raw);
            
            // Loss calculation
            double loss = 0.5 * (target - output) * (target - output);
            total_loss += loss;
            
            // Backward pass
            double d_output = (output - target) * sigmoid_derivative(output_raw);
            
            // d_hidden2: element-wise multiplication
            std::vector<double> d_hidden2(hidden_size2);
            for (size_t i = 0; i < hidden_size2; i++) {
                d_hidden2[i] = d_output * output_weights[i] * tanh_derivative(hidden2_raw[i]);
            }
            
            // d_hidden1: matrix multiplication and element-wise multiplication
            std::vector<double> d_hidden1(hidden_size1, 0.0);
            for (size_t i = 0; i < hidden_size1; i++) {
                for (size_t j = 0; j < hidden_size2; j++) {
                    d_hidden1[i] += d_hidden2[j] * hidden_weights[j][i];
                }
                d_hidden1[i] *= tanh_derivative(hidden1_raw[i]);
            }
            
            // Update weights and biases
            // Update output layer
            for (size_t i = 0; i < hidden_size2; i++) {
                output_weights[i] -= learning_rate * d_output * hidden2[i];
            }
            output_bias -= learning_rate * d_output;
            
            // Update hidden layer
            auto d_hidden2_outer = outer(d_hidden2, hidden1);
            for (size_t i = 0; i < hidden_size2; i++) {
                for (size_t j = 0; j < hidden_size1; j++) {
                    hidden_weights[i][j] -= learning_rate * d_hidden2_outer[i][j];
                }
                hidden_biases[i] -= learning_rate * d_hidden2[i];
            }
            
            // Update input layer
            auto d_hidden1_outer = outer(d_hidden1, input_vec);
            for (size_t i = 0; i < hidden_size1; i++) {
                for (size_t j = 0; j < 2; j++) {
                    input_weights[i][j] -= learning_rate * d_hidden1_outer[i][j];
                }
                input_biases[i] -= learning_rate * d_hidden1[i];
            }
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss/data.size() << std::endl;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    training_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

double MLP::getAccuracy(const std::vector<std::vector<double>>& data,
                       const std::vector<double>& labels) const {
    int correct = 0;
    for (size_t i = 0; i < data.size(); i++) {
        double output = forward(data[i][0], data[i][1]);
        if ((output >= 0.5 && labels[i] == 1) || (output < 0.5 && labels[i] == 0)) {
            correct++;
        }
    }
    return static_cast<double>(correct) / data.size();
}

double MLP::getTrainingTime() const {
    return training_time;
}
