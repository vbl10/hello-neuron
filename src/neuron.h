#ifndef NEURON_H
#define NEURON_H

#include "value.h"
#include <vector>

class Neuron
{
public:
    const std::vector<Value>& in;
    std::vector<Value> w;
    Value b;
    Value act;

    Neuron(const std::vector<Value>& in, const std::string& label = "");

    Value operator()() const;
};

#endif