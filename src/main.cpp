#include <iostream>
#include "MLP.h"

int main()
{
    std::vector<std::vector<Value>> xs({
        {Value(2.0f), Value(3.0f), Value(-1.0f)},
        {Value(3.0f), Value(-1.0f), Value(0.5f)},
        {Value(0.5f), Value(1.0f), Value(1.0f)},
        {Value(1.0f), Value(1.0f), Value(-1.0f)}
    });
    std::vector<Value> ys = {
        Value(1.0f), 
        Value(-1.0f), 
        Value(-1.0f), 
        Value(1.0f)
    };

    MLP mlp(3, {4, 4, 1});

    auto mlpParams = mlp.getParams();

    for (int i = 0; i < 30; i++)
    {
        Value loss;
        for (int j = 0; j < xs.size(); j++)
            loss = loss + (ys[j] - mlp(xs[j])[0]).pow(2);

        std::cout << "Loss: " << loss->val << std::endl;
        
        loss->backward();
        for (auto& p : mlpParams)
        {
            p->val += -0.05f * p->grad;
        }
    }
    Value loss;
    for (int j = 0; j < xs.size(); j++)
        loss = loss + (ys[j] - mlp(xs[j])[0]).pow(2);
    
    std::cout << "Loss: " << loss->val << std::endl;

    std::cout << "MLP:\n";
    for (int i = 0; i < mlpParams.size(); i++)
        std::cout << "  " << mlpParams[i]->val << std::endl;

    return 0;
}