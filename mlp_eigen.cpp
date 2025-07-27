#include "mlp_eigen.h"
#include <random>
#include <chrono>
#include <iostream>

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;

MLP_Eigen::MLP_Eigen() {
    std::mt19937 gen(5489);
    std::normal_distribution<> d1(0, std::sqrt(2.0/2)); // for W1
    std::normal_distribution<> d2(0, std::sqrt(2.0/H1));
    std::normal_distribution<> d3(0, std::sqrt(2.0/H2));

    for(int i=0;i<H1;i++){
        W1(i,0)=d1(gen);
        W1(i,1)=d1(gen);
        b1(i)=0.0;
    }
    for(int i=0;i<H2;i++){
        W3(i)=d3(gen);
        b2(i)=0.0;
        for(int j=0;j<H1;j++) W2(i,j)=d2(gen);
    }
    b3=0.0;
}

double MLP_Eigen::forward(double x, double y) const {
    Eigen::Matrix<double,2,1> inp; inp<<x,y;
    Eigen::Matrix<double,H1,1> z1 = W1*inp + b1;
    Eigen::Matrix<double,H1,1> a1 = z1.array().tanh();
    Eigen::Matrix<double,H2,1> z2 = W2*a1 + b2;
    Eigen::Matrix<double,H2,1> a2 = z2.array().tanh();
    double z3 = W3.dot(a2) + b3;
    return sigmoid(z3);
}

void MLP_Eigen::train(const std::vector<std::vector<double>>& data,
                      const std::vector<double>& labels,
                      int epochs,double lr){
    auto t0=std::chrono::high_resolution_clock::now();
    const int N=data.size();
    for(int ep=0;ep<epochs;ep++){
        double loss=0.0;
        for(int n=0;n<N;n++){
            double x=data[n][0], y=data[n][1];
            Eigen::Matrix<double,2,1> inp; inp<<x,y;
            // forward
            Eigen::Matrix<double,H1,1> z1=W1*inp+b1;
            Eigen::Matrix<double,H1,1> a1=z1.array().tanh();
            Eigen::Matrix<double,H2,1> z2=W2*a1+b2;
            Eigen::Matrix<double,H2,1> a2=z2.array().tanh();
            double z3=W3.dot(a2)+b3;
            double out=sigmoid(z3);
            loss+=0.5*(labels[n]-out)*(labels[n]-out);
            // backward
            double d3=(out-labels[n])*(out*(1.0-out));
            Eigen::Matrix<double,H2,1> g2=(1.0 - a2.array().square()).matrix();
            Eigen::Matrix<double,H2,1> d2=W3*d3; d2.array()*=g2.array();
            Eigen::Matrix<double,H1,1> g1=(1.0 - a1.array().square()).matrix();
            Eigen::Matrix<double,H1,1> d1=W2.transpose()*d2; d1.array()*=g1.array();
            // update
            W3 -= lr*d3*a2;
            b3 -= lr*d3;
            W2 -= lr*(d2*a1.transpose());
            b2 -= lr*d2;
            W1 -= lr*(d1*inp.transpose());
            b1 -= lr*d1;
        }
        if(ep%100==0) std::cout<<"Epoch "<<ep<<", Loss: "<<loss/N<<std::endl;
    }
    training_time_ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-t0).count();
}

double MLP_Eigen::getAccuracy(const std::vector<std::vector<double>>& d,const std::vector<double>& lab) const {
    int correct=0; int N=d.size();
    for(int i=0;i<N;i++){
        double p=forward(d[i][0],d[i][1]);
        if((p>=0.5 && lab[i]==1) || (p<0.5 && lab[i]==0)) correct++;
    }
    return double(correct)/N;
} 