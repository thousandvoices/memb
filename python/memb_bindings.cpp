#include "builder.h"
#include "reader.h"
#include "compression_strategy.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(_memb, m) {
    py::class_<memb::Builder>(m, "Builder")
        .def(py::init<size_t, const std::string&, size_t>())
        .def(
            "add_word",
            [](memb::Builder& builder, const std::string& word, py::array_t<float, py::array::c_style> values)
            {
                auto valuesBuffer = values.request();
                if (valuesBuffer.ndim != 1) {
                    throw std::runtime_error("Word vector must be 1-dimensional");
                }

                float* valuesPointer = reinterpret_cast<float*>(valuesBuffer.ptr);

                builder.addWord(
                    word,
                    std::vector<float>(valuesPointer, valuesPointer + valuesBuffer.shape[0]));
            })
        .def(
            "save",
            [](memb::Builder& builder, const std::string& filename)
            {
                builder.save(filename);
            });

    py::class_<memb::Reader>(m, "Reader")
        .def(py::init<std::string, size_t>())
        .def(
            "dim",
            [](memb::Reader& reader)
            {
                return reader.dim();
            })
        .def(
            "word_embedding",
            [](memb::Reader& reader, const std::string& word)
            {
                py::array_t<float> result(reader.dim());
                auto buffer = result.request();
                reader.wordEmbeddingToBuffer(word, reinterpret_cast<float*>(buffer.ptr));

                return result;
            })
        .def(
            "batch_embedding",
            [](memb::Reader& reader, const std::vector<std::string>& words)
            {
                py::array_t<float> result({words.size(), reader.dim()});
                auto buffer = result.request();
                reader.batchEmbeddingToBuffer(words, reinterpret_cast<float*>(buffer.ptr));

                return result;
            })
        .def(
            "keys",
            [](memb::Reader& reader)
            {
                return reader.keys();
            });

    m.def("available_compression_strategies", &memb::availableCompressionStrategies);
}
