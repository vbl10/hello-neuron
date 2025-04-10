#include "MLP.h"

int main()
{
    std::vector<Value> in({
        Value(0.0f, "in1"),
        Value(0.0f, "in2"),
        Value(0.0f, "in3")
    });
    MLP mlp(in, {4, 4, 1});

    mlp.outs[0]->backward();
    mlp.outs[0]->print();

    return 0;
}