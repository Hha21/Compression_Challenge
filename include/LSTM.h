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
        Eigen::MatrixXd Wf, Wi, Wo, Wc;         ///< INPUT -> GATES
        Eigen::MatrixXd Uf, Ui, Uo, Uc;         ///< HIDDEN -> GATES
        Eigen::VectorXd bf, bi, bo, bc;         ///< BIASES

        Eigen::MatrixXd Why;                    ///< HIDDEN -> OUTPUT
        Eigen::VectorXd by;                     ///< OUTPUT BIAS

        // STATE
        Eigen::VectorXd h_t; 
        Eigen::VectorXd c_t;

        // GATE OUTPUTS
        Eigen::VectorXd f_t, i_t, o_t, c_hat_t;

        // ADAM PARAMETERS
        

    public:

        LSTM(int input_dim_, int hidden_dim_, int output_dim_);

        Eigen::VectorXd forward(const Eigen::VectorXd& x_t);              //FORWARD PASS
        void backward(const Eigen::VectorXd& dL_dy);
        void updateWeights(double learning_rate);

        void reset();
};

#endif //LSTM_H