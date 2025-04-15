#include "../include/LSTMTorch.h"

torch::Tensor NetImpl::forward(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto hidden = initHidden(batch_size);
    return std::get<0>(forward(x, hidden));
}

LSTMOutput NetImpl::forward(torch::Tensor x, LSTMStates hidden) {
    x =  this->embedding(x);    // shape: [batch, seq_len] → [batch, seq_len, embed_size]
    LSTMOutput  lstm_out = this->lstm->forward(x, hidden);
    torch::Tensor   output = std::get<0>(lstm_out);
    LSTMStates  next_hidden = std::get<1>(lstm_out);

    torch::Tensor logits = this->fc(output);    // shape: [batch, seq_len, vocab_size]

    return std::make_tuple(logits, next_hidden);
}

LSTMStates NetImpl::initHidden(int64_t batch_size) {
    torch::Tensor h0 = torch::zeros({lstm->options.num_layers(), batch_size, lstm->options.hidden_size()});
    torch::Tensor c0 = torch::zeros({lstm->options.num_layers(), batch_size, lstm->options.hidden_size()});

    return std::make_tuple(h0, c0);
}