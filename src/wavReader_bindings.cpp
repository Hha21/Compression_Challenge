#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/wavReader.h"

namespace py = pybind11;

PYBIND11_MODULE(neuralink, m) {
    m.doc() = "Neuralink compression tools";

    py::class_<wavReader>(m, "WavReader")
        .def(py::init<int>(), py::arg("dictionary_size"))
        .def("getNumFiles", &wavReader::getNumFiles)
        .def("getNumTokens", &wavReader::getNumTokens)
        .def("getStreamSize", &wavReader::getStreamSize)
        .def("getTokenStream",
             [](wavReader &self, int idx) {
                 // return a copy of the vector
                 return std::vector<int>(self.getTokenStream(idx));
             },
             py::arg("index"),
             "Return the BPE-encoded token ID stream for file `index`");
}

