#include "../include/arithmeticCode.h"
#include <iostream>


arithmeticCode::arithmeticCode(std::unique_ptr<std::vector<std::pair<int, double>>> sortedProb) {

    double cumulative = 0.0;

    for (const auto& [symbol, probability] : *sortedProb) {
        cumulative += probability;
        cumulative_probabilities.emplace_back(symbol, cumulative);
        std::cout << std::fixed << std::setprecision(16)  // 16 decimal places
              << "Symbol: " << symbol << " | Cumulative: " << cumulative << std::endl;
    }
}

uint64_t arithmeticCode::encode(const std::vector<int>& symbols) {  
    uint64_t low = 0;
    uint64_t high = SCALE;
    uint64_t range;

    for (int i = 0; i < symbols.size(); ++i) {
        int symbol = symbols[i];

        double cumulative_low = 0.0;
        double cumulative_high = 0.0;

        // Find cumulative_low and cumulative_high for the current symbol
        for (const auto& [sym, cumulative] : cumulative_probabilities) {
            if (sym == symbol) {
                cumulative_high = cumulative;
                break;
            }
            cumulative_low = cumulative;
        }

        range = high - low;
        high = low + static_cast<uint64_t>(range * cumulative_high);
        low = low + static_cast<uint64_t>(range * cumulative_low);

        std::cout << "Iteration: " << i << " | Symbol: " << symbol 
                  << " | Low: " << low << " | High: " << high 
                  << " | Diff: " << (high - low) << std::endl;
    }

    return (low + high) / 2;
}

std::vector<int> arithmeticCode::decode(uint64_t code, int num_symbols) {
    uint64_t low = 0;
    uint64_t high = SCALE;
    uint64_t range;
    std::vector<int> decoded_symbols;
    double cumulative_low;
    double cumulative_high;

    for (int i = 0; i < num_symbols; ++i) {
        range = high - low;
        double fraction = static_cast<double>(code - low) / range;
        
        int symbol = 0;
        cumulative_low = 0.0;
        cumulative_high = 0.0;

        for (int j = 0; j < cumulative_probabilities.size(); ++j) {
            double cumulative_low = (j == 0) ? 0.0 : cumulative_probabilities[j - 1].second;
            double cumulative_high = cumulative_probabilities[j].second;

            if (fraction >= cumulative_low && fraction < cumulative_high) {
                symbol = cumulative_probabilities[j].first;
                high = low + static_cast<uint64_t>(range * cumulative_high);
                low = low + static_cast<uint64_t>(range * cumulative_low);
                break;
            }
        }

        decoded_symbols.push_back(symbol);

        std::cout << "Iteration: " << i << " | Symbol: " << symbol 
                    << " | Low: " << low << " | High: " << high 
                    << " | Diff: " << high - low
                    << " | Fraction: " << fraction << std::endl;


        if (low == high) {
            std::cerr << "Warning: Range collapsed at iteration " << i << std::endl;
            break;
        }

    }

    return decoded_symbols; 

}

