#include <torch/extension.h>

#include <vector>

// CUDA forward declaration

std::vector<torch::Tensor> gcnn_cuda_forward(
    torch::Tensor input,
    torch::Tensor filters,
    int in_channels,
    int out_channels,
    int in_trans,
    int out_trans,
    int filter_size,
    torch::Tensor ind1,
    torch::Tensor ind2,
    torch::Tensor ind3);

// NOTE: TORCH_CHECK is used in the new versions of PyTorch
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> gcnn_cuda_forward(
    torch::Tensor input,
    torch::Tensor filters,
    int in_channels,
    int out_channels,
    int in_trans,
    int out_trans,
    int filter_size,
    torch::Tensor ind1,
    torch::Tensor ind2,
    torch::Tensor ind3){

    CHECK_INPUT(input);
    CHECK_INPUT(filter);
    CHECK_INPUT(ind1);
    CHECK_INPUT(ind2);
    CHECK_INPUT(ind3);

  return gcnn_cuda_forward(input, filters, in_channels, out_channels, in_trans, filter_size, ind1, ind2, ind3);
}
