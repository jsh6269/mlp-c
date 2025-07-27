// mlp.cpp
#include "mlp.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

MLP::MLP()
  : input_weights(hidden_size1*2),
    hidden_weights(hidden_size2*hidden_size1),
    output_weights(hidden_size2),
    input_biases(hidden_size1, 0.0),
    hidden_biases(hidden_size2, 0.0),
    output_bias(0.0),
    training_time_ms(0),
    hidden1_raw(hidden_size1), hidden1(hidden_size1),
    hidden2_raw(hidden_size2), hidden2(hidden_size2),
    d_hidden1(hidden_size1), d_hidden2(hidden_size2)
{
    // C++ mt19937, seed = 5489
    std::mt19937 gen(5489);

    double scale_input = std::sqrt(2.0/2.0);
    double scale_h     = std::sqrt(2.0/hidden_size1);
    double scale_out   = std::sqrt(2.0/hidden_size2);

    std::normal_distribution<> d_input(0, scale_input);
    for (auto &w : input_weights)  w = d_input(gen);

    std::normal_distribution<> d_h(0, scale_h);
    for (auto &w : hidden_weights) w = d_h(gen);

    std::normal_distribution<> d_out(0, scale_out);
    for (auto &w : output_weights) w = d_out(gen);
}

double MLP::forward(double x, double y) const {
    // input → hidden1
    for (int i = 0; i < hidden_size1; ++i) {
        hidden1_raw[i] = input_weights[i*2 + 0] * x
                       + input_weights[i*2 + 1] * y
                       + input_biases[i];
        hidden1[i]     = tanh_fn(hidden1_raw[i]);
    }
    // hidden1 → hidden2
    for (int i = 0; i < hidden_size2; ++i) {
        double sum = 0.0;
        for (int j = 0; j < hidden_size1; ++j) {
            sum += hidden_weights[i*hidden_size1 + j] * hidden1[j];
        }
        hidden2_raw[i] = sum + hidden_biases[i];
        hidden2[i]     = tanh_fn(hidden2_raw[i]);
    }
    // hidden2 → output
    double out_raw = output_bias;
    for (int i = 0; i < hidden_size2; ++i) {
        out_raw += output_weights[i] * hidden2[i];
    }
    return sigmoid(out_raw);
}

void MLP::train(const std::vector<std::vector<double>>& data,
                const std::vector<double>& labels,
                int epochs, double lr)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    int N = data.size();
    for (int ep = 0; ep < epochs; ++ep) {
        double total_loss = 0.0;
        for (int idx = 0; idx < N; ++idx) {
            double x = data[idx][0];
            double y = data[idx][1];
            double target = labels[idx];

            // --- Forward ---
            for (int i = 0; i < hidden_size1; ++i) {
                hidden1_raw[i] = input_weights[i*2 + 0]*x
                               + input_weights[i*2 + 1]*y
                               + input_biases[i];
                hidden1[i]     = tanh_fn(hidden1_raw[i]);
            }
            for (int i = 0; i < hidden_size2; ++i) {
                double sum = 0.0;
                for (int j = 0; j < hidden_size1; ++j) {
                    sum += hidden_weights[i*hidden_size1 + j] * hidden1[j];
                }
                hidden2_raw[i] = sum + hidden_biases[i];
                hidden2[i]     = tanh_fn(hidden2_raw[i]);
            }
            double out_raw = output_bias;
            for (int i = 0; i < hidden_size2; ++i) {
                out_raw += output_weights[i] * hidden2[i];
            }
            double output = sigmoid(out_raw);

            // loss
            double diff = (output - target);
            total_loss += 0.5 * diff * diff;

            // --- Backward ---
            // d_output
            double d_output = diff * sigmoid_derivative_from_raw(out_raw);

            // d_hidden2
            for (int i = 0; i < hidden_size2; ++i) {
                d_hidden2[i] = d_output
                             * output_weights[i]
                             * tanh_derivative_from_raw(hidden2_raw[i]);
            }
            // d_hidden1
            for (int i = 0; i < hidden_size1; ++i) {
                double sum = 0.0;
                for (int j = 0; j < hidden_size2; ++j) {
                    sum += hidden_weights[j*hidden_size1 + i] * d_hidden2[j];
                }
                d_hidden1[i] = sum * tanh_derivative_from_raw(hidden1_raw[i]);
            }

            // --- Update weights & biases ---
            // output layer
            for (int i = 0; i < hidden_size2; ++i) {
                output_weights[i] -= lr * d_output * hidden2[i];
            }
            output_bias -= lr * d_output;

            // hidden → hidden2
            for (int i = 0; i < hidden_size2; ++i) {
                for (int j = 0; j < hidden_size1; ++j) {
                    hidden_weights[i*hidden_size1 + j]
                      -= lr * d_hidden2[i] * hidden1[j];
                }
                hidden_biases[i] -= lr * d_hidden2[i];
            }

            // input → hidden1
            for (int i = 0; i < hidden_size1; ++i) {
                input_weights[i*2 + 0] -= lr * d_hidden1[i] * x;
                input_weights[i*2 + 1] -= lr * d_hidden1[i] * y;
                input_biases[i]        -= lr * d_hidden1[i];
            }
        }

        if (ep % 100 == 0) {
            std::cout << "Epoch " << ep
                      << ", Loss: " << total_loss / N
                      << std::endl;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    training_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}

double MLP::getAccuracy(const std::vector<std::vector<double>>& data,
                        const std::vector<double>& labels) const
{
    int correct = 0;
    int N = data.size();
    for (int i = 0; i < N; ++i) {
        double p = forward(data[i][0], data[i][1]);
        if ((p >= 0.5 && labels[i] == 1.0) ||
            (p <  0.5 && labels[i] == 0.0)) {
            ++correct;
        }
    }
    return double(correct) / N;
}

int MLP::getTrainingTime() const {
    return training_time_ms;
}
