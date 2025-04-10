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

Neuron::Neuron(const std::vector<Value>& in, const std::string& label)
    :
    in(in),
    b(rand1(), (label.empty() ? "" : label + "_") + "b")
{
    auto prefix = (label.empty() ? "" : label + "_");
    for (int i = 0; i < in.size(); i++)
        w.push_back(Value(rand1(), prefix + "w" + std::to_string(i + 1)));
    
    act = b;
    for (int i = 0; i < in.size(); i++)
        act = act + w[i] * in[i];
    act = act.tanh();
}

Value Neuron::operator()() const
{
    act->forward();
    return act;
}