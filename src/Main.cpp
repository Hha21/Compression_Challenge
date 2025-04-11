// Main.cpp

#include <cassert>
#include <iostream>
#include <boost/program_options.hpp>

#include "../include/wavReader.h"
#include "../include/Predictor.h"

namespace po = boost::program_options;

int main (int argc, char** argv) {

    po::options_description opts("Allowed Options");
    opts.add_options()
        ("help,h", "Print help message")
        ("dictionary_size,N", po::value<int>()->default_value(1023), "Set dictionary size");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << opts << std::endl;
        return 0;
    }

    int N = vm["dictionary_size"].as<int>();

    assert((N >= 1023) && "ERROR: DICTIONARY SIZE MUST BE >= NUMBER OF SINGLE SYMBOLS (1023)");

    wavReader dataReader = wavReader(N);
    Predictor predictor = Predictor(dataReader);

    return 0;
}   