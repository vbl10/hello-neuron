#ifndef MLP_H
#define MLP_H

#include "layer.h"

class MLP
{
public:
    std::vector<Layer> layers;

    MLP(
        unsigned int nin, 
        const std::vector<unsigned int>& nouts, 
        const std::string& label = ""
    );

    std::vector<Value> getParams() const;
    std::vector<Value> operator()(const std::vector<Value>& in) const;
};

#endif