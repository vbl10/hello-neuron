#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include <vector>

class Layer
{
public:
    std::vector<Neuron> neurons;
    std::vector<Value> outs;

    Layer(const std::vector<Value>& in, unsigned int nout, const std::string& label = "");

    const std::vector<Value>& operator()() const;
};

#endif