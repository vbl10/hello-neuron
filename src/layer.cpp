#include "layer.h"

Layer::Layer(unsigned int nin, unsigned int nout, const std::string& label)
{
    auto prefix = label.empty() ? "" : label + "_";
    for (int i = 0; i < nout; i++)
        neurons.push_back(Neuron(nin, prefix + "N" + std::to_string(i + 1)));
}

std::vector<Value> Layer::getParams() const
{
    std::vector<Value> params;
    for (const auto& n : neurons)
    {
        auto aux = n.getParams();
        params.insert(params.end(), aux.begin(), aux.end());
    }
    return params;
}
std::vector<Value> Layer::operator()(const std::vector<Value>& in) const
{
    std::vector<Value> outs;
    for (auto n : neurons) outs.push_back(n(in));
    return outs;
}