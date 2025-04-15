#include "../include/Predictor.h"
#include "../include/LSTMTorch.h"

Predictor::Predictor(const int DICTIONARY_SIZE) {

    wavReader dataReader = wavReader(DICTIONARY_SIZE);

    this->num_files = dataReader.getNumFiles();
    this->vocab_size = dataReader.getNumTokens();
    this->number_tokens_stream == dataReader.getStreamSize();

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

void Predictor::trainModel(int num_epochs, float learning_rate) {

    int batches_per_epoch = 3000; //(this->number_tokens_stream * this->train_split) / (this->BATCH_SIZE * this->SEQUENCE_LENGTH);
    

    torch::Device device(torch::kCPU);

    torch::optim::Adam optimiser(this->model->parameters(), torch::optim::AdamOptions(learning_rate));
    torch::nn::CrossEntropyLoss criterion;

    float loss_value, perplexity;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (int batch_idx = 0; batch_idx < batches_per_epoch; ++batch_idx) {
            std::pair<torch::Tensor, torch::Tensor> batch = this->generateBatch();
            torch::Tensor inputs = std::get<0>(batch);
            torch::Tensor targets = std::get<1>(batch);

            LSTMStates states = this->model->initHidden(static_cast<int64_t>(this->BATCH_SIZE));
            LSTMOutput output = this->model->forward(inputs, states);
            torch::Tensor logits = std::get<0>(output);
            logits = logits.view({-1, this->vocab_size});            // [BATCH_SIZE * SEQ_LEN, vocab_size]
            torch::Tensor flat_targets = targets.view(-1);          // [BATCH_SIZE * SEQ_LEN]

            torch::Tensor loss = criterion(logits, flat_targets);

            optimiser.zero_grad();
            loss.backward();
            optimiser.step();

            loss_value = loss.item<float>();
            perplexity = std::exp(loss_value);
        }
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs
                  << " - Loss: " << loss_value
                  << " - Perplexity: " << perplexity << std::endl;  
    }
}