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

void MLP::matmul(const std::vector<double>& A,
                 const std::array<double, 2>& x,
                 std::vector<double>& result) const {
    for (int i = 0; i < hidden_size1; i++) {
        result[i] = A[i] * x[0] + A[i + hidden_size1] * x[1];
    }
}

void MLP::matmul(const std::vector<double>& A,
                 const std::vector<double>& x,
                 std::vector<double>& result) const {
    for (int i = 0; i < hidden_size2; i++) {
        double sum = 0.0;
        for (int j = 0; j < hidden_size1; j++) {
            sum += A[i * hidden_size1 + j] * x[j];
        }
        result[i] = sum;
    }
}

MLP::MLP() 
    : hidden_size1(64),
      hidden_size2(64),
      input_weights(hidden_size1 * 2),
      hidden_weights(hidden_size2 * hidden_size1),
      output_weights(hidden_size2),
      input_biases(hidden_size1),
      hidden_biases(hidden_size2),
      output_bias(0.0),
      hidden1_raw_buffer(hidden_size1),
      hidden1_buffer(hidden_size1),
      hidden2_raw_buffer(hidden_size2),
      hidden2_buffer(hidden_size2),
      d_hidden1_buffer(hidden_size1),
      d_hidden2_buffer(hidden_size2) {
    
    std::mt19937 gen(5489);
    
    double scale_input = std::sqrt(2.0 / 2);
    double scale_h = std::sqrt(2.0 / hidden_size1);
    double scale_out = std::sqrt(2.0 / hidden_size2);
    
    std::normal_distribution<> d_input(0, scale_input);
    for (auto& w : input_weights) w = d_input(gen);
    
    std::normal_distribution<> d_h(0, scale_h);
    for (auto& w : hidden_weights) w = d_h(gen);
    
    std::normal_distribution<> d_out(0, scale_out);
    for (auto& w : output_weights) w = d_out(gen);
    
    std::fill(input_biases.begin(), input_biases.end(), 0.0);
    std::fill(hidden_biases.begin(), hidden_biases.end(), 0.0);
}

double MLP::forward(double x, double y) const {
    const std::array<double, 2> input_vec = {x, y};
    
    // First hidden layer
    matmul(input_weights, input_vec, hidden1_raw_buffer);
    for (int i = 0; i < hidden_size1; i++) {
        hidden1_raw_buffer[i] += input_biases[i];
        hidden1_buffer[i] = tanh(hidden1_raw_buffer[i]);
    }
    
    // Second hidden layer
    matmul(hidden_weights, hidden1_buffer, hidden2_raw_buffer);
    for (int i = 0; i < hidden_size2; i++) {
        hidden2_raw_buffer[i] += hidden_biases[i];
        hidden2_buffer[i] = tanh(hidden2_raw_buffer[i]);
    }
    
    // Output layer
    double output_raw = 0.0;
    for (int i = 0; i < hidden_size2; i++) {
        output_raw += hidden2_buffer[i] * output_weights[i];
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
            const std::array<double, 2> input_vec = {data[idx][0], data[idx][1]};
            const double target = labels[idx];
            
            // Forward pass
            matmul(input_weights, input_vec, hidden1_raw_buffer);
            for (int i = 0; i < hidden_size1; i++) {
                hidden1_raw_buffer[i] += input_biases[i];
                hidden1_buffer[i] = tanh(hidden1_raw_buffer[i]);
            }
            
            matmul(hidden_weights, hidden1_buffer, hidden2_raw_buffer);
            for (int i = 0; i < hidden_size2; i++) {
                hidden2_raw_buffer[i] += hidden_biases[i];
                hidden2_buffer[i] = tanh(hidden2_raw_buffer[i]);
            }
            
            double output_raw = 0.0;
            for (int i = 0; i < hidden_size2; i++) {
                output_raw += hidden2_buffer[i] * output_weights[i];
            }
            output_raw += output_bias;
            double output = sigmoid(output_raw);
            
            // Loss calculation
            double loss = 0.5 * (target - output) * (target - output);
            total_loss += loss;
            
            // Backward pass
            // Use output*(1-output) instead of sigmoid_derivative
            double d_output = (output - target) * (output * (1.0 - output));
            
            // Calculate d_hidden2 using stored activation values
            for (int i = 0; i < hidden_size2; i++) {
                const double g2 = 1.0 - hidden2_buffer[i] * hidden2_buffer[i];  // tanh derivative using stored activation
                d_hidden2_buffer[i] = d_output * output_weights[i] * g2;
            }
            
            // Calculate d_hidden1 using stored activation values
            for (int i = 0; i < hidden_size1; i++) {
                double sum = 0.0;
                for (int j = 0; j < hidden_size2; j++) {
                    sum += d_hidden2_buffer[j] * hidden_weights[j * hidden_size1 + i];
                }
                const double g1 = 1.0 - hidden1_buffer[i] * hidden1_buffer[i];  // tanh derivative using stored activation
                d_hidden1_buffer[i] = sum * g1;
            }
            
            // Update weights and biases
            for (int i = 0; i < hidden_size2; i++) {
                output_weights[i] -= learning_rate * d_output * hidden2_buffer[i];
            }
            output_bias -= learning_rate * d_output;
            
            // Update hidden weights directly without outer product buffer
            for (int i = 0; i < hidden_size2; i++) {
                const double gi = d_hidden2_buffer[i];
                for (int j = 0; j < hidden_size1; j++) {
                    hidden_weights[i * hidden_size1 + j] -= learning_rate * gi * hidden1_buffer[j];
                }
                hidden_biases[i] -= learning_rate * gi;
            }
            
            // Update input weights directly without outer product buffer
            for (int i = 0; i < hidden_size1; i++) {
                const double gi = d_hidden1_buffer[i];
                input_weights[i] -= learning_rate * gi * input_vec[0];
                input_weights[i + hidden_size1] -= learning_rate * gi * input_vec[1];
                input_biases[i] -= learning_rate * gi;
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
