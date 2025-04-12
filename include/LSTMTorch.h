#ifndef LSTMTORCH_H
#define LSTMTORCH_H

#include <iostream>
#include <torch/torch.h>

class NetImpl : public torch::nn::Module {

    private: 
        torch::nn::Embedding embedding{nullptr};
        torch::nn::LSTM lstm{nullptr};
        torch::nn::Linear fc{nullptr};

    public:

        NetImpl() {}

        NetImpl(int64_t vocab_size, int64_t embed_size, int64_t hidden_size, int64_t number_layers) {
            embedding = register_module("embedding", torch::nn::Embedding(vocab_size, embed_size));
            lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(embed_size, hidden_size).num_layers(number_layers).batch_first(true)));
            fc = register_module("fc", torch::nn::Linear(hidden_size, vocab_size));

            std::cout << "INIT LSTM!" << std::endl;
        }

        torch::Tensor forward(torch::Tensor x) {
            x = embedding(x);
            auto lstm_out = lstm->forward(x);
            auto output = std::get<0>(lstm_out);
            return fc(output);
        }
};

TORCH_MODULE(Net);

#endif //LSTMTORCH_H