#include <memory>
#include <vector>
#include <string>

class Value
{
private:
    enum class OPERATION 
    {
        NONE,
        ADD,
        MULT,
        POW
    };

    class Core
    {
        friend Value;
    public:
        float val, grad = 0.0f;
        std::string label;

    private:
        OPERATION operation = OPERATION::NONE;
        std::vector<std::shared_ptr<Core>> prev;

    public:
        Core(float val, const std::string& label) :val(val), label(label) {}
        void printTree(int depth = 0);

        //forward changes in values through network
        void forward();
        //backpropagate gradients through network
        void backward();
    private:
        //topological order of previous nodes
        std::vector<Core*> buildTopo();
        void _backward();
        void _forward();
    };
    std::shared_ptr<Core> core;

public:
    Value(float val, const std::string& label);
    Value(float val);

    Core* operator->() const;

    Value operator+(const Value& rhs) const;
    Value operator*(const Value& rhs) const;
    Value pow(const Value& rhs) const;

    Value operator+(float rhs) const { return operator+(Value(rhs)); }
    Value operator*(float rhs) const { return operator*(Value(rhs)); }
    Value pow(float rhs) const { return pow(Value(rhs)); }
};

Value operator+(float lhs, const Value& rhs);
Value operator*(float lhs, const Value& rhs);