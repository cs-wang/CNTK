//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKLibrary.h"

using namespace CNTK;

class UserTimesFunction final : public Function
{
public:
    static FunctionPtr Create(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return AsComposite(std::shared_ptr<Function>(new UserTimesFunction(leftOperand, rightOperand, name), [](Function* ptr) { delete ptr; }));
    }

private:
    BackPropStatePtr Forward(const std::vector<ValuePtr>& inputValues,
                             std::unordered_map<Variable, ValuePtr>& outputs,
                             const DeviceDescriptor& computeDevice,
                             const std::unordered_set<Variable>& /*outputsToRetainBackwardStateFor*/) override
    {
        auto leftOperandData = inputValues[0]->Data();
        auto rightOperandData = inputValues[1]->Data();

        auto numOutRows = leftOperandData->Shape()[0];
        auto K = leftOperandData->Shape()[1];
        auto numOutCols = rightOperandData->Shape()[1];

        assert(K == rightOperandData->Shape()[0]);
        auto& outputValue = outputs[this->Output()];
        if (outputValue == nullptr)
            outputValue = std::shared_ptr<Value>(new Value(std::shared_ptr<NDArrayView>(new NDArrayView(DataType::Float, NDShape({ numOutRows , numOutCols }), computeDevice), [](NDArrayView* ptr) { delete ptr; })), [](Value* ptr) { delete ptr; });

        auto outputData = outputs[this->Output()]->Data();
        outputData->SetValue(0.0f);

        // The operands values are in column major layout
        auto FlattenedIndex = [](size_t rowIdx, size_t colIdx, const NDShape& matrixShape) {
            return (colIdx * matrixShape[0]) + rowIdx;
        };

        auto leftBuffer = leftOperandData->DataBuffer<float>();
        auto rightBuffer = rightOperandData->DataBuffer<float>();
        auto outBuffer = outputData->WritableDataBuffer<float>();
        for (size_t j = 0; j < numOutCols; ++j)
            for (size_t k = 0; k < K; ++k)
                for (size_t i = 0; i < numOutRows; ++i)
                    outBuffer[FlattenedIndex(i, j, outputData->Shape())] += leftBuffer[FlattenedIndex(i, k, leftOperandData->Shape())] * rightBuffer[FlattenedIndex(k, j, rightOperandData->Shape())];

        return nullptr;
    }

    const std::wstring& OpName() const override
    {
        static const std::wstring opName = L"UserTimesOp";
        return opName;
    }

    Dictionary Serialize() const override { NOT_IMPLEMENTED; }
    size_t CurrentVersion() const override { NOT_IMPLEMENTED; }

    void InferOutputs(std::vector<Variable>& outputs) override
    {
        auto leftOperand = Inputs()[0];
        auto rightOperand = Inputs()[1];

        if (leftOperand.Shape().Rank() != 2)
            std::runtime_error("Left operand must be 2D");

        if (rightOperand.Shape().Rank() != 1)
            std::runtime_error("Right operand must be 1D");

        if (!leftOperand.DynamicAxes().empty())
            std::runtime_error("Left operand must not have dynamic axes (i.e. should not be minibatch data, but be a Parameter of fixed size)");

        outputs.push_back(OutputVariable(NDShape({ leftOperand.Shape()[0] }), leftOperand.GetDataType(), rightOperand.DynamicAxes()));
    }

    UserTimesFunction(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
        : Function({ leftOperand, rightOperand }, Dictionary(), name)
    {
    }
};

