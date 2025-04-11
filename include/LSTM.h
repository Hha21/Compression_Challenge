#ifndef LSTM_H
#define LSTM_H

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>

class LSTM {

    private:
        const int input_dim;
        const int hidden_dim;
        const int output_dim;

        // PARAMETERS AND STATES
        Eigen::MatrixXd Wf, Wi, Wo, Wc;
        Eigen::MatrixXd Uf, Ui, Uo, Uc;
        Eigen::VectorXd bf, bi, bo, bc;

        Eigen::MatrixXd Why;
        Eigen::VectorXd by;

        Eigen::VectorXd h, c;

    public:

        LSTM(int input_dim_, int hidden_dim_, int output_dim_);

        Eigen::VectorXd forward(const Eigen::VectorXd& x);              //FORWARD PASS
        void backward(const Eigen::VectorXd& dL_dy);
        void updateWeights(double learning_rate);

        void reset();
};

#endif //LSTM_H