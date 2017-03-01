#include "UserMatrixMultiplicationOp.h"

void main()
{
    auto device = DeviceDescriptor::CPUDevice();
    size_t outDim = 15;
    size_t inDim = 10;
    auto W = Parameter(NDShape({ outDim, inDim }), DataType::Float, GlorotUniformInitializer(), device);
    auto x = InputVariable(NDShape({ inDim }), DataType::Float, { Axis::DefaultBatchAxis() });
    auto builtInTimes = Times(W, x, L"BuiltInTimes");
    auto userDefinedTimes = UserTimesFunction::Create(W, x, L"UserDefinedTimes");

    size_t batchSize = 3;
    std::vector<float> inputData(inDim * batchSize);
    for (size_t i = 0; i < inputData.size(); ++i)
        inputData[i] = (float)rand() / RAND_MAX;

    auto inputDataValue = Value::CreateBatch(x.Shape(), inputData, device);
    std::unordered_map<Variable, ValuePtr> outputValues = { { builtInTimes->Output(), nullptr} };
    builtInTimes->Forward({ {x, inputDataValue} }, outputValues, device);
}
