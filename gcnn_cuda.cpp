#include <torch/extension.h>

#include <vector>

// CUDA forward declaration

torch::Tensor gcnn_cuda_forward(
    torch::Tensor filters,
    int in_channels,
    int out_channels,
    int in_trans,
    int out_trans,
    int filter_size,
    torch::Tensor ind1,
    torch::Tensor ind2,
    torch::Tensor ind3,
    torch::Tensor filters_transformed);

// NOTE: TORCH_CHECK is used in the new versions of PyTorch
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor gcnn_forward(
    torch::Tensor filters,
    int in_channels,
    int out_channels,
    int in_trans,
    int out_trans,
    int filter_size,
    torch::Tensor ind1,
    torch::Tensor ind2,
    torch::Tensor ind3){


    torch::Tensor filters_transformed = torch::zeros({out_channels * out_trans, in_channels * in_trans, filter_size, filter_size});
    //CHECK_INPUT(input);
    CHECK_INPUT(filters);
    //CHECK_INPUT(in_channels);
    //CHECK_INPUT(out_channels);
    //CHECK_INPUT(in_trans);
    //CHECK_INPUT(out_trans);
    //CHECK_INPUT(filter_size);
    CHECK_INPUT(ind1);
    CHECK_INPUT(ind2);
    CHECK_INPUT(ind3);
    CHECK_INPUT(filters_transformed);

  return gcnn_cuda_forward(filters, in_channels, out_channels, in_trans, out_trans, filter_size, ind1, ind2, ind3, filters_transformed);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gcnn_forward, "GCNN forward (CUDA)");
  //m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}
