#ifndef MLP_H
#define MLP_H

#include "layer.h"

class MLP
{
public:
    std::vector<Layer> layers;
    std::vector<Value> outs;

    MLP(
        const std::vector<Value>& in, 
        const std::vector<unsigned int>& nouts, 
        const std::string& label = ""
    );

    const std::vector<Value>& operator()() const;
};

#endif