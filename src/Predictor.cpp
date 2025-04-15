#include "../include/Predictor.h"

Predictor::Predictor(const int DICTIONARY_SIZE) {

    wavReader dataReader = wavReader(DICTIONARY_SIZE);

    this->num_files = dataReader.getNumFiles();
    this->vocab_size = dataReader.getNumTokens();

    const int64_t embed_size = 128;
    const int64_t hidden_size = 256;
    const int64_t num_layers = 2;

    // INIT LSTM MODEL
    this->model = Net(this->vocab_size, embed_size, hidden_size, num_layers);

    // NUM TRAINING FILES
    size_t train_size = static_cast<size_t>(this->train_split * num_files);
    this->train_streams.reserve(train_size);
    this->val_streams.reserve(num_files - train_size);

    for (size_t i = 0; i < num_files; ++i) {
        if (i < train_size) {
            this->train_streams.push_back(dataReader.getTokenStream(i));
        } else {
            this->val_streams.push_back(dataReader.getTokenStream(i));
        }
    }

    std::random_device rd;
    this->rng = std::mt19937(rd());

    std::cout << "CONSTRUCTOR - NUM FILES: " << this->num_files << ", VOCAB SIZE: " << this->vocab_size << std::endl;
}

std::pair<torch::Tensor, torch::Tensor> Predictor::generateBatch() {
    torch::Tensor inputs = torch::empty({this->BATCH_SIZE, this->SEQUENCE_LENGTH}, torch::kLong);
    torch::Tensor targets = torch::empty({this->BATCH_SIZE, this->SEQUENCE_LENGTH}, torch::kLong);

    std::uniform_int_distribution<> file_dist(0, train_streams.size() - 1);
    for (int i = 0; i < this->BATCH_SIZE; ++i) {
        const auto& stream = this->train_streams[file_dist(rng)];

        if (stream.size() <= SEQUENCE_LENGTH) {
            throw std::runtime_error("SEQUENCE TOO SHORT TO GENERATE BATCH...");
        }

        std::uniform_int_distribution<> start_dist(0, stream.size() - this->SEQUENCE_LENGTH - 1);
        int start = start_dist(this->rng);

        for (int j = 0; j < this->SEQUENCE_LENGTH; ++j) {
            inputs[i][j] = static_cast<int64_t>(stream[start + j]);
            targets[i][j] = static_cast<int64_t>(stream[start + j + 1]);
        }
    }

    return {inputs, targets};
}

void Predictor::train() {
    std::pair<torch::Tensor, torch::Tensor> batch = this->generateBatch();

    torch::Tensor inputs = batch.first;
    torch::Tensor targets = batch.second;

    for (int i = 0; i < this->BATCH_SIZE; ++i) {
        std::cout << "INPUT: " << inputs[i] << std::endl;
        std::cout << "TARGET: " << targets[i] << std::endl;
    }
}