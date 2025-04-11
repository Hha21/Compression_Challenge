#include "../include/Predictor.h"

Predictor::Predictor(wavReader& reader_ref) : reader(reader_ref) {
    std::cout << "PREDICTOR -- NUM FILES: " << reader.getNumFiles() << std::endl;
}