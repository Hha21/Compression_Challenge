// Main.cpp

#include <iostream>
#include <string>
#include <cstdlib>
#include <cassert>

#include "../include/wavReader.h"
#include "../include/Predictor.h"

int main (int argc, char** argv) {

    int N = 1023;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: ./bpe [-N <dictionary_size>]\n"
                      << "Options:\n"
                      << "  -N <int>        Set dictionary size (default: 1023)\n"
                      << "  -h, --help      Print this message\n";
            return 0;
        }

        if ((arg == "-N" || arg == "--dictionary_size") && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        }
    }

    assert((N >= 1023) && "ERROR: DICTIONARY SIZE MUST BE >= NUMBER OF SINGLE SYMBOLS (1023)");

    Predictor predictor = Predictor(N);

    predictor.train();
    
    return 0;
}   