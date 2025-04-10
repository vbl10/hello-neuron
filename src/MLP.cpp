#include "MLP.h"

MLP::MLP(const std::vector<Value>& in, const std::vector<unsigned int>& nouts, const std::string& label)
{
    auto prefix = (label.empty() ? "" : label + "_");

    layers.push_back(Layer(in, nouts[0], prefix + "L1"));

    for (int i = 1; i < nouts.size(); i++)
        layers.push_back(
            Layer(
                layers[i - 1].outs, 
                nouts[i], 
                prefix + "L" + std::to_string(i + 1)
            )
        );
    
    outs = layers.back().outs;
}

const std::vector<Value>& MLP::operator()() const
{
    layers.back()();
    return outs;
}
