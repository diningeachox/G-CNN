#include <torch/extension.h>

#include <iostream>
#include <vector>

// CUDA forward declarations
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

torch::Tensor gmaxpool_cuda_forward(
    torch::Tensor input,
    torch::Tensor output);

torch::Tensor gcnn_cuda_backward(
    int out_channels, int out_trans, int in_channels, int in_trans, int filter_size,
    torch::Tensor ind1,
    torch::Tensor ind2,
    torch::Tensor ind3,
    torch::Tensor grad_filters,
    torch::Tensor grad_filters_trans);

// NOTE: TORCH_CHECK is used in the new versions of PyTorch
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
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

    //Tensor to write new values to (also move the tensor to GPU)
<<<<<<< HEAD
    torch::Tensor filters_transformed = torch::zeros({out_channels * out_trans, in_channels * in_trans, filter_size, filter_size}, torch::TensorOptions().dtype(torch::kF32).layout(torch::kStrided).device(torch::kCUDA));
=======
    torch::Tensor filters_transformed = torch::zeros({out_channels * out_trans, in_channels * in_trans, filter_size, filter_size}, torch::TensorOptions().device(torch::kCUDA).requires_grad(true));
>>>>>>> bd4210ffd9be09bef281ef6282592c56a1de600d

    //Check if input tensors are on the GPU
    CHECK_INPUT(filters);
    CHECK_INPUT(ind1);
    CHECK_INPUT(ind2);
    CHECK_INPUT(ind3);
    CHECK_INPUT(filters_transformed);

    return gcnn_cuda_forward(filters, in_channels, out_channels, in_trans, out_trans, filter_size, ind1, ind2, ind3, filters_transformed);
}

torch::Tensor gmaxpool_forward(torch::Tensor input){
    //Tensor to write new values to (also move the tensor to GPU)
    torch::Tensor output = torch::zeros({input.size(0), input.size(1) / 4, input.size(2), input.size(3)}, torch::TensorOptions().device(torch::kCUDA).requires_grad(true));

    //Check if input tensors are on the GPU
    CHECK_INPUT(input);

    return gmaxpool_cuda_forward(input, output);
}

torch::Tensor gcnn_backward(
    int out_channels, int out_trans, int in_channels, int in_trans, int filter_size,
    torch::Tensor ind1,
    torch::Tensor ind2,
    torch::Tensor ind3,
    torch::Tensor grad_filters_trans
){
<<<<<<< HEAD
    torch::Tensor grad_filters = torch::zeros({out_channels, in_channels, in_trans, filter_size, filter_size}, torch::TensorOptions().dtype(torch::kF32).layout(torch::kStrided).device(torch::kCUDA));
=======
    torch::Tensor grad_filters = torch::zeros({out_channels, in_channels, in_trans, filter_size, filter_size}, torch::TensorOptions().device(torch::kCUDA).requires_grad(true));
>>>>>>> bd4210ffd9be09bef281ef6282592c56a1de600d

    //Check if input tensors are on the GPU
    CHECK_INPUT(grad_filters_trans);
    CHECK_INPUT(ind1);
    CHECK_INPUT(ind2);
    CHECK_INPUT(ind3);

    return gcnn_cuda_backward(out_channels, out_trans, in_channels, in_trans, filter_size, ind1, ind2, ind3, grad_filters, grad_filters_trans);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gmaxpool_forward", &gmaxpool_forward, "GMaxPool forward (CUDA)");
    m.def("forward", &gcnn_forward, "GCNN forward (CUDA)");
    m.def("backward", &gcnn_backward, "GCNN backward (CUDA)");
}
