#include "neuron.h"
#include <format>
#include <random>

std::random_device rd;
std::mt19937 gen(rd());
float rand1()
{
    static std::uniform_real_distribution dist(-1.0f, 1.0f);
    return dist(gen);
}

Neuron::Neuron(unsigned int nin, const std::string& label)
    :
    b(rand1(), (label.empty() ? "" : label + "_") + "b")
{
    auto prefix = (label.empty() ? "" : label + "_");
    for (int i = 0; i < nin; i++)
        w.push_back(Value(rand1(), prefix + "w" + std::to_string(i + 1)));
    
    
}

std::vector<Value> Neuron::getParams() const
{
    std::vector<Value> params = w;
    params.push_back(b);
    return params;
}
Value Neuron::operator()(const std::vector<Value>& in) const
{
    auto act = b;
    for (int i = 0; i < w.size(); i++)
        act = act + w[i] * in[i];
    act = act.tanh();
    
    return act;
}