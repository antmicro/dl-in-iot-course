#include "eigen-delegate.hpp"
#include <memory>

#include <Eigen/Core>
#include <tensorflow/lite/c/builtin_op_data.h>

#define DEBUG

using namespace tflite::tools;

namespace tflite
{

EigenDelegate::EigenDelegate(
    const SimpleDelegateInterface::Options &options
) : options(options)
{}

bool EigenDelegate::IsNodeSupportedByDelegate(
        const TfLiteRegistration *registration,
        const TfLiteNode *node,
        TfLiteContext *context) const
{
    switch (registration->builtin_code)
    {
        // TODO uncomment
        // case kTfLiteBuiltinAdd:
        case kTfLiteBuiltinFullyConnected:
            break;
        default:
            printf("Skipped builtin code %d\n", registration->builtin_code);
            return false;
    }
    // Only support float32
    for (int i = 0; i < node->inputs->size; i++)
    {
        auto &tensor = context->tensors[node->inputs->data[i]];
        if (tensor.type != kTfLiteFloat32)
        {
            printf("Skipped tensor type %d for %d (%s,%d)\n",
                tensor.type,
                i,
                registration->custom_name == NULL ? "" : registration->custom_name,
                registration->builtin_code
            );
            return false;
        }
    }
    return true;
}

TfLiteStatus EigenDelegate::Initialize(TfLiteContext *context)
{
    return kTfLiteOk;
}

const char *EigenDelegate::Name() const
{
    static constexpr char kName[] = "EigenDelegate";
    return kName;
}

std::unique_ptr<SimpleDelegateKernelInterface> EigenDelegate::CreateDelegateKernelInterface()
{
    return std::make_unique<EigenDelegateKernel>();
}

TfLiteStatus EigenDelegateKernel::Init(TfLiteContext* context, const TfLiteDelegateParams* params)
{
    // Save index to all nodes which are part of this delegate.
    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);
    nodes_.resize(params->nodes_to_replace->size);
    intermediatedata.clear();
    for (int i = 0; i < params->nodes_to_replace->size; i++)
    {
        const int node_index = params->nodes_to_replace->data[i];
        // Get this node information.
        TfLiteNode* delegated_node = nullptr;
        TfLiteRegistration* delegated_node_registration = nullptr;
        TF_LITE_ENSURE_EQ(
            context,
            context->GetNodeAndRegistration(
                context,
                node_index,
                &delegated_node,
                &delegated_node_registration
            ),
            kTfLiteOk
        );
        for (int j = 0; j < delegated_node->inputs->size; j++)
        {
            inputs_[i].push_back(delegated_node->inputs->data[j]);
        }
        for (int j = 0; j < delegated_node->outputs->size; j++)
        {
            outputs_[i].push_back(delegated_node->outputs->data[j]);
        }
        builtin_code_[i] = delegated_node_registration->builtin_code;
        nodes_[i] = delegated_node;
    }
#ifdef DEBUG
    printf("BEFORE ALLOCATION (tensor id, data address, dims address)\n");
    for (int i = 0; i < context->tensors_size; i++)
    {
        printf("tensor %d:  %x %x\n",
            i,
            context->tensors[i].data,
            context->tensors[i].dims
        );
    }
#endif // DEBUG
    for (auto &out : outputs_)
    {
        for (auto &outid : out)
        {
            if (GetTensorData<float *>(&context->tensors[outid]) == NULL)
            {
                intermediatedata.push_back(
                    std::make_unique<std::vector<float>>(
                        NumElements(&context->tensors[outid])
                    )
                );
                context->tensors[outid].data.f = intermediatedata.back()->data();
            }
        }
    }
#ifdef DEBUG
    printf("AFTER ALLOCATION (tensor id, data address, dims address)\n");
    for (int i = 0; i < context->tensors_size; i++)
    {
        printf("tensor %d:  %x %x\n",
            i,
            context->tensors[i].data,
            context->tensors[i].dims
        );
    }
#endif // DEBUG
    return kTfLiteOk;
}

TfLiteStatus EigenDelegateKernel::Prepare(TfLiteContext* context, TfLiteNode* node)
{
    return kTfLiteOk;
}

TfLiteStatus EigenDelegateKernel::Eval(TfLiteContext* context, TfLiteNode* node)
{
    TfLiteStatus ret = kTfLiteOk;
    for (int i = 0; i < inputs_.size(); i++)
    {
        switch (builtin_code_[i])
        {
            // TODO uncomment
            // case kTfLiteBuiltinAdd:
            //     ret = add(context, i);
            //     break;
            case kTfLiteBuiltinFullyConnected:
                ret = fullyConnected(context, i);
                break;
            default:
                ret = kTfLiteDelegateError;
        }
        if (ret != kTfLiteOk)
        {
            break;
        }
    }
    return ret;
}

TfLiteStatus EigenDelegateKernel::add(TfLiteContext *context, int nodeid)
{
    // TODO implement using Eigen routines
    return kTfLiteOk;
}

TfLiteStatus EigenDelegateKernel::fullyConnected(TfLiteContext *context, int nodeid)
{
    auto &tfinput = context->tensors[inputs_[nodeid][0]];
    auto &tfweights = context->tensors[inputs_[nodeid][1]];
    auto &tfbias = context->tensors[inputs_[nodeid][2]];
    auto &tfoutput = context->tensors[outputs_[nodeid][0]];

    auto ws = GetTensorShape(&tfweights);

    Eigen::Map<Eigen::VectorXf> input(
        GetTensorData<float>(&tfinput), NumElements(&tfinput)
    );

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> weights(
        GetTensorData<float>(&tfweights),
        ws.Dims(0),
        ws.Dims(1)
    );

    Eigen::Map<Eigen::VectorXf> bias(
        GetTensorData<float>(&tfbias),
        NumElements(&tfbias)
    );

    Eigen::Map<Eigen::VectorXf> output(
        GetTensorData<float>(&tfoutput),
        NumElements(&tfoutput)
    );

    output = weights * input + bias;

    TfLiteFullyConnectedParams *params = static_cast<TfLiteFullyConnectedParams *>(
        nodes_[nodeid]->builtin_data
    );

    switch (params->activation)
    {
        case TfLiteFusedActivation::kTfLiteActNone:
            break;
        case TfLiteFusedActivation::kTfLiteActRelu:
            output = output.cwiseMax(0);
            break;
        default:
            return kTfLiteDelegateError;
    }
    return kTfLiteOk;
}

} // namespace tflite

tflite::SimpleDelegateInterface::Options TfLiteEigenDelegateOptionsDefault() {
    tflite::SimpleDelegateInterface::Options options = {0};
    return options;
}

TfLiteDelegate *TfLiteEigenDelegateCreate(const tflite::SimpleDelegateInterface::Options* options) {
    std::unique_ptr<tflite::EigenDelegate> custom(
        new tflite::EigenDelegate(
            options ? *options : TfLiteEigenDelegateOptionsDefault()
        )
    );
    return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(custom));
}

void TfLiteEigenDelegateDelete(TfLiteDelegate* delegate) {
    tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}

TfLiteDelegate *CreateEigenDelegateFromOptions(char **options_keys, char **options_values, size_t num_options)
{
    return TfLiteEigenDelegateCreate(NULL);
}

TfLiteDelegate *tflite_plugin_create_delegate(char** options_keys, char** options_values, size_t num_options, void (*report_error)(const char*))
{
	return CreateEigenDelegateFromOptions(options_keys, options_values, num_options);
}

void tflite_plugin_destroy_delegate(TfLiteDelegate *delegate)
{
	TfLiteEigenDelegateDelete(delegate);
}
