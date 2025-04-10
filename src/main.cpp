#include "value.h"

int main()
{
    Value a(0.5f, "a"), b(1.0f, "b"), c(3.0f, "c");
    Value d = c.pow(2.0f*(a + b));
    d->backward();
    d->printTree();

    return 0;
}