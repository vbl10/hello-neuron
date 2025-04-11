#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include <vector>

class Layer
{
public:
    std::vector<Neuron> neurons;

    Layer(unsigned int nin, unsigned int nout, const std::string& label = "");

    std::vector<Value> getParams() const;
    std::vector<Value> operator()(const std::vector<Value>& in) const;
};

#endif