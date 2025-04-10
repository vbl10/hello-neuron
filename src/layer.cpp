#include "layer.h"

Layer::Layer(const std::vector<Value>& in, unsigned int nout, const std::string& label)
{
    auto prefix = label.empty() ? "" : label + "_";
    for (int i = 0; i < nout; i++)
        neurons.push_back(Neuron(in, prefix + "N" + std::to_string(i + 1)));

    for (auto n : neurons)
        outs.push_back(n());
}

const std::vector<Value>& Layer::operator()() const
{
    for (auto n : neurons) n();
    return outs;
}