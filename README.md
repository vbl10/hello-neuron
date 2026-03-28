# Neural Network Autograd Engine (C++)

Projeto autodidata focado na implementação de uma *autograd engine* em C++ para compreender conceitos fundamentais de redes neurais e cálculo automático de gradientes.

## Objetivos

- Entender o funcionamento interno de redes neurais
- Implementar cálculo de gradientes via *backpropagation*
- Construir um sistema de diferenciação automática (*autograd*)
- Trabalhar com estruturas de dados e grafos computacionais em C++

## Conceitos abordados

- Grafos computacionais
- Diferenciação automática (modo reverso)
- Backpropagation
- Funções de ativação
- Operações matriciais básicas
- Otimização (gradiente descendente)

## Funcionalidades

- Construção dinâmica de grafo computacional
- Suporte a operações básicas com rastreamento de gradientes
- Execução automática do *backward pass*
- Implementação simples de camadas neurais

## Exemplo

```cpp
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

    for (int i = 0; i < 100; i++)
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

    std::cout << "MLP Params:\n";
    for (int i = 0; i < mlpParams.size(); i++)
        std::cout << "  " << mlpParams[i]->val << std::endl;

    std::cout << "Test:\n";
    for (int i = 0; i < xs.size(); i++)
    {
        std::cout << "  ys: " << ys[i]->val << " | ypred: " << mlp(xs[i])[0]->val << std::endl;
    }

    return 0;
}
```
