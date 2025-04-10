#include "value.h"
#include <iostream>
#include <cmath>
#include <functional>
#include <set>
#include <format>

void Value::Core::printTree(int depth)
{
    printf("%*s%s[val: %.3f, grad: %.3f]: {", depth * 2, "", label.c_str(), val, grad);
    if (!prev.empty())
    {
        printf("\n");
        prev[0]->printTree(depth + 1);
    }
    for (int i = 1; i < prev.size(); i++)
    {
        printf(",\n");
        prev[i]->printTree(depth + 1);
    }
    if (!prev.empty())
        printf("\n%*s", depth * 2, "");
    printf("}");
    if (depth == 0)
        printf("\n");
}
void Value::Core::forward()
{
    for (auto p : buildTopo())
        p->_forward();
}
void Value::Core::_forward()
{
    switch (operation)
    {
    case OPERATION::ADD:
        val = prev[0]->val + prev[1]->val;
        break;
    case OPERATION::MULT:
        val = prev[0]->val * prev[1]->val;
        break;
    case OPERATION::POW:
        val = std::pow(prev[0]->val, prev[1]->val);
        break;
    }
}
void Value::Core::backward()
{
    std::vector<Core*> topo(buildTopo());

    for (const auto &node : topo)
        node->grad = 0.0f;

    grad = 1.0f;
    for (auto it = topo.rbegin(); it != topo.rend(); it++)
        (*it)->_backward();
}
void Value::Core::_backward()
{
    switch (operation)
    {
    case OPERATION::ADD:
        prev[0]->grad += grad;
        prev[1]->grad += grad;
        break;
    case OPERATION::MULT:
        prev[0]->grad += prev[1]->val * grad;
        prev[1]->grad += prev[0]->val * grad;
        break;
    case OPERATION::POW:
        prev[0]->grad += prev[1]->val * std::pow(prev[0]->val, prev[1]->val - 1.0f) * grad;
        prev[1]->grad += std::log(prev[0]->val) * std::pow(prev[0]->val, prev[1]->val) * grad;
        break;
    }
}
std::vector<Value::Core*> Value::Core::buildTopo()
{
    std::vector<Core*> topo;
    std::set<Core*> visited;
    std::function<void(Core*)> build_topo = [&build_topo, &topo, &visited](Core* node) -> void
    {
        if (!visited.count(node))
        {
            visited.insert(node);
            for (const auto &child : node->prev)
            {
                build_topo(child.get());
            }
            topo.push_back(node);
        }
    };
    build_topo(this);
    return topo;
}

Value::Value(float val, const std::string &label)
    : core(std::make_shared<Core>(val, label))
{
}
Value::Value(float val)
    : core(std::make_shared<Core>(val, std::format("{:.3f}", val)))
{
}
Value::Core *Value::operator->() const
{
    return &(*core);
}
Value Value::operator+(const Value &rhs) const
{
    Value out(core->val + rhs->val, "(" + core->label + "+" + rhs->label + ")");
    out->operation = OPERATION::ADD;
    out->prev.push_back(core);
    out->prev.push_back(rhs.core);
    return out;
}
Value Value::operator*(const Value &rhs) const
{
    Value out(core->val * rhs->val, "(" + core->label + "*" + rhs->label + ")");
    out->operation = OPERATION::MULT;
    out->prev.push_back(core);
    out->prev.push_back(rhs.core);
    return out;
}
Value Value::pow(const Value &rhs) const
{
    Value out(std::pow(core->val, rhs->val), "(" + core->label + "^" + rhs->label + ")");
    out->operation = OPERATION::POW;
    out->prev.push_back(core);
    out->prev.push_back(rhs.core);
    return out;
}

Value operator+(float lhs, const Value& rhs) 
{
    return rhs + lhs;
}
Value operator*(float lhs, const Value& rhs) 
{
    return rhs * lhs;
}