#pragma once

#include <memory>
#include <vector>

#include <tensorflow/lite/delegates/utils/simple_delegate.h>
#include <tensorflow/lite/builtin_ops.h>

#include <tensorflow/lite/kernels/internal/tensor_ctypes.h>
#include <tensorflow/lite/kernels/kernel_util.h>
#include <tensorflow/lite/tools/delegates/delegate_provider.h>

namespace tflite
{
class EigenDelegateKernel : public SimpleDelegateKernelInterface
{
public:
    TfLiteStatus Init(TfLiteContext* context, const TfLiteDelegateParams* params) override;

    TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override;

    TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override;
private:
    TfLiteStatus add(TfLiteContext *context, int nodeid);
    TfLiteStatus fullyConnected(TfLiteContext *context, int nodeid);

    std::vector<std::vector<int>> inputs_, outputs_;
    std::vector<int> builtin_code_;
    std::vector<TfLiteNode *> nodes_;
    std::vector<std::unique_ptr<std::vector<float>>> intermediatedata;
};

class EigenDelegate : public SimpleDelegateInterface
{
public:
    explicit EigenDelegate(const SimpleDelegateInterface::Options &options);

    bool IsNodeSupportedByDelegate(const TfLiteRegistration *registration, const TfLiteNode *node, TfLiteContext *context) const;

    TfLiteStatus Initialize(TfLiteContext *context) override;

    const char *Name() const override;

    std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface() override;

    SimpleDelegateInterface::Options DelegateOptions() const {return options;};
private:
    const SimpleDelegateInterface::Options options;
};
}

tflite::SimpleDelegateInterface::Options TfLiteEigenDelegateOptionsDefault();

TfLiteDelegate *TfLiteEigenDelegateCreate(const tflite::SimpleDelegateInterface::Options *options);

void TfLiteEigenDelegateDelete(TfLiteDelegate *delegate);

TfLiteDelegate *CreateEigenDelegateFromOptions(char **options_keys, char **options_values, size_t num_options);

extern "C"
{
TFL_CAPI_EXPORT TfLiteDelegate *tflite_plugin_create_delegate(char** options_keys, char** options_values, size_t num_options, void (*report_error)(const char*));
TFL_CAPI_EXPORT void tflite_plugin_destroy_delegate(TfLiteDelegate *delegate);
}
