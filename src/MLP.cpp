#include "MLP.h"

MLP::MLP(unsigned int nin, const std::vector<unsigned int>& nouts, const std::string& label)
{
    auto prefix = (label.empty() ? "" : label + "_");

    layers.push_back(Layer(nin, nouts[0], prefix + "L1"));

    for (int i = 1; i < nouts.size(); i++)
        layers.push_back(
            Layer(
                layers[i - 1].neurons.size(), 
                nouts[i], 
                prefix + "L" + std::to_string(i + 1)
            )
        );    
}

std::vector<Value> MLP::getParams() const
{
    std::vector<Value> params;
    for (const auto& l : layers)
    {
        auto aux = l.getParams();
        params.insert(params.end(), aux.begin(), aux.end());
    }
    return params;
}
std::vector<Value> MLP::operator()(const std::vector<Value>& in) const
{
    std::vector<Value> out = in;
    for (int i = 0; i < layers.size(); i++)
        out = layers[i](out);
    return out;
}
