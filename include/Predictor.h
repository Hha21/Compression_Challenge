#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "wavReader.h"
// #include "LSTM.h"
#include "LSTMTorch.h"

// #include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

class Predictor {

    private:

        wavReader& reader;                      ///< Reference to wavReader class.
        //LSTM model;
        Net model;

    public:

        Predictor(wavReader& reader_ref);
};

#endif // PREDICTOR_H