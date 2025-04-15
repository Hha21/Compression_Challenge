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

#include <torch/torch.h>

class Predictor {

    private:

        Net model;                                      ///< LSTM Model

        const double train_split = 0.8;                 ///< % of files to treat as train vs test

        int num_files;                                  ///< Total Number of Files
        int64_t vocab_size;                             ///< Command-Line Argument, number of Tokens
        size_t number_tokens_stream;

        // TRAINING PARAMETERS
        const int SEQUENCE_LENGTH = 10;                 ///< Prediction sequence length for LSTM
        const int BATCH_SIZE = 16;                       ///< Num Batches to process in parallel

        // DATA
        std::vector<std::vector<int>> train_streams;
        std::vector<std::vector<int>> val_streams;

        // RNG
        std::mt19937 rng;

    public:

        Predictor(const int DICTIONARY_SIZE);

        std::pair<torch::Tensor, torch::Tensor> generateBatch();

        void trainModel(int num_epochs, float learning_rate);
};

#endif // PREDICTOR_H