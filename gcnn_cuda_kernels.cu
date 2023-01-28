#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}
/*
template <typename scalar_t>
__global__ void gcnn_cuda_forward_kernel(
    const scalar_t* __restrict__ gates,
    const scalar_t* __restrict__ old_cell,
    scalar_t* __restrict__ new_h,
    scalar_t* __restrict__ new_cell,
    scalar_t* __restrict__ input_gate,
    scalar_t* __restrict__ output_gate,
    scalar_t* __restrict__ candidate_cell,
    size_t state_size) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * state_size + column;
  const int gates_row = blockIdx.y * (state_size * 3);
  if (column < state_size) {
    input_gate[index] = sigmoid(gates[gates_row + column]);
    output_gate[index] = sigmoid(gates[gates_row + state_size + column]);
    candidate_cell[index] = elu(gates[gates_row + 2 * state_size + column]);
    new_cell[index] =
        old_cell[index] + candidate_cell[index] * input_gate[index];
    new_h[index] = tanh(new_cell[index]) * output_gate[index];
  }
}
*/
namespace {
template <typename scalar_t>
__global__ void gcnn_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> ind1,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> ind2,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> ind3,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> filter,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> filters_transformed,
    int out_trans, int in_trans, int filter_size) {

  // column index
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int s_prime = 0; s_prime < out_trans; s_prime++){
      for (int s = 0; s < in_trans; s++){
          for (int u = 0; u < filter_size; u++){
              for (int v = 0; v < filter_size; v++){
                  _s = ind1[s_prime, s, u, v].item()
                  _u = ind2[s_prime, s, u, v].item()
                  _v = ind3[s_prime, s, u, v].item()
                  filters_transformed[row * s_prime, col * s, u, v] = filters[row, col, _s, _u, _v]
              }
          }
      }
  }
}
} //namespace

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

    //Allocate thread blocks
    //const int threads = 1024;
    dim3 threads_per_block(16, 16);
    const dim3 blocks((in_channels + threads_per_block.x - 1) / threads_per_block.x,
                      (out_channels + threads_per_block.y - 1) / threads_per_block.y);

    AT_DISPATCH_FLOATING_TYPES(gates.type(), "gcnn_forward_cuda", ([&] {
    gcnn_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
        ind1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        ind2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        ind3.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        filter.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        filter_transformed.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        int out_trans, int in_trans, int filter_size);
    }));
}
