#ifndef NEURON_H
#define NEURON_H

#include "value.h"
#include <vector>

class Neuron
{
public:
    std::vector<Value> w;
    Value b;

    Neuron(unsigned int nin, const std::string& label = "");

    std::vector<Value> getParams() const;
    Value operator()(const std::vector<Value>& in) const;
};

#endif