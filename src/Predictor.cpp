#include "../include/Predictor.h"

Predictor::Predictor(wavReader& reader_ref) : reader(reader_ref) {
    int64_t vocab_size = 1050; 
    int64_t embed_size = 64;
    int64_t hidden_size = 128;
    int64_t num_layers = 2;

    this->model = Net(vocab_size, embed_size, hidden_size, num_layers);

    std::cout << "PREDICTOR -- NUM FILES: " << reader.getNumFiles() << std::endl;
}