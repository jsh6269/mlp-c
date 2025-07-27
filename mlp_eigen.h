// mlp_eigen.h
#ifndef MLP_EIGEN_H
#define MLP_EIGEN_H
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <chrono>

class MLP_Eigen {
public:
    static constexpr int H1 = 64;
    static constexpr int H2 = 64;

    MLP_Eigen();

    // Forward pass: x,y in [0,1]
    double forward(double x, double y) const;

    // Train data (Nx2) & labels (0/1)
    void train(const std::vector<std::vector<double>>& data,
               const std::vector<double>& labels,
               int epochs = 2000, double lr = 0.05);

    double getAccuracy(const std::vector<std::vector<double>>& data,
                       const std::vector<double>& labels) const;

    int getTrainingTime() const { return training_time_ms; }

private:
    Eigen::Matrix<double, H1, 2>  W1; // 64x2
    Eigen::Matrix<double, H2, H1> W2; // 64x64
    Eigen::Matrix<double, H2, 1>  W3; // 64x1 (column)
    Eigen::Matrix<double, H1, 1>  b1;
    Eigen::Matrix<double, H2, 1>  b2;
    double b3;

    int training_time_ms{};

    // utilities
    static double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }
};

#endif // MLP_EIGEN_H 