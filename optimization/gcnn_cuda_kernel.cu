#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <initializer_list>
#include <vector>
#include <stdio.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

//template <typename scalar_t>
__global__ void gcnn_cuda_forward_kernel(
    torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> ind1,
    torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> ind2,
    torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> ind3,
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> filters,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> filters_transformed,
    int in_channels, int out_channels, int out_trans, int in_trans, int filter_size) {

  // column index
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < out_channels && col < in_channels){
      for (int s_prime = 0; s_prime < out_trans; s_prime++){
          for (int s = 0; s < in_trans; s++){
              for (int u = 0; u < filter_size; u++){
                  for (int v = 0; v < filter_size; v++){
                      int _s = ind1[s_prime][s][u][v];
                      int _u = ind2[s_prime][s][u][v];
                      int _v = ind3[s_prime][s][u][v];
                      filters_transformed[row * s_prime][col * s][u][v] = filters[row][col][_s][_u][_v];
                  }
              }
          }
      }
  }
}


__global__ void gmaxpool_cuda_forward_kernel(
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> output){

    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < input.size(0); b++){
        for (int i = 0; i < input.size(0); i++){

            float m = std::max(input[b][i][u][v], input[b][i + output.size(0)][u][v]);
            m = std::max(m, input[b][i + output.size(0) * 2][u][v]);
            m = std::max(m, input[b][i + output.size(0) * 3][u][v]);
            output[b][i][u][v] = m;

        }
    }
}

__global__ void gcnn_cuda_backward_kernel(
    torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> ind1,
    torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> ind2,
    torch::PackedTensorAccessor32<int,4,torch::RestrictPtrTraits> ind3,
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> grad_filters,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> grad_filters_trans,
    int in_channels, int out_channels, int out_trans, int in_trans, int filter_size) {

  // column index
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < out_channels && col < in_channels){
      for (int s_prime = 0; s_prime < out_trans; s_prime++){
          for (int s = 0; s < in_trans; s++){
              for (int u = 0; u < filter_size; u++){
                  for (int v = 0; v < filter_size; v++){
                      int _s = ind1[s_prime][s][u][v];
                      int _u = ind2[s_prime][s][u][v];
                      int _v = ind3[s_prime][s][u][v];
                      __syncthreads(); //Stop race conditions
                      grad_filters[row][col][_s][_u][_v] += grad_filters_trans[row * out_trans + s_prime][col * in_trans + s][u][v];
                  }
              }
          }
      }
  }
}

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
    torch::Tensor filters_transformed) {

    //Allocate thread blocks
    dim3 threads_per_block(16, 16);
    int blocks_x = (in_channels + threads_per_block.x - 1) / threads_per_block.x;
    int blocks_y = (out_channels + threads_per_block.y - 1) / threads_per_block.y;
    const dim3 blocks(blocks_x, blocks_y);

    gcnn_cuda_forward_kernel<<<blocks, threads_per_block>>>(
        ind1.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
        ind2.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
        ind3.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
        filters.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
        filters_transformed.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        in_channels, out_channels, out_trans, in_trans, filter_size);

    /*
    AT_DISPATCH_FLOATING_TYPES(filters.type(), "gcnn_forward_cuda", ([&] {
        gcnn_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            ind1.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
            ind2.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
            ind3.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
            filters.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
            filters_transformed.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
            out_trans, in_trans, filter_size);
    }));
    */

    return filters_transformed;
}

torch::Tensor gmaxpool_cuda_forward(torch::Tensor input, torch::Tensor output){
    //Allocate thread blocks
    dim3 threads_per_block(16, 16);
    int blocks_x = (input.size(3) + threads_per_block.x - 1) / threads_per_block.x;
    int blocks_y = (input.size(2) + threads_per_block.y - 1) / threads_per_block.y;
    const dim3 blocks(blocks_x, blocks_y);

    gmaxpool_cuda_forward_kernel<<<blocks, threads_per_block>>>(
        input.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        output.packed_accessor32<float,4,torch::RestrictPtrTraits>());

    return output;
}

torch::Tensor gcnn_cuda_backward(
    int out_channels, int out_trans, int in_channels, int in_trans, int filter_size,
    torch::Tensor ind1,
    torch::Tensor ind2,
    torch::Tensor ind3,
    torch::Tensor grad_filters,
    torch::Tensor grad_filters_trans) {

    //Allocate thread blocks
    dim3 threads_per_block(16, 16);
    int blocks_x = (in_channels + threads_per_block.x - 1) / threads_per_block.x;
    int blocks_y = (out_channels + threads_per_block.y - 1) / threads_per_block.y;
    const dim3 blocks(blocks_x, blocks_y);

    gcnn_cuda_backward_kernel<<<blocks, threads_per_block>>>(
        ind1.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
        ind2.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
        ind3.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
        grad_filters.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
        grad_filters_trans.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        in_channels, out_channels, out_trans, in_trans, filter_size);

    /*
    AT_DISPATCH_FLOATING_TYPES(filters.type(), "gcnn_forward_cuda", ([&] {
        gcnn_cuda_forward_kernel<scalar_t><<<blocks, threads_per_block>>>(
            ind1.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
            ind2.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
            ind3.packed_accessor32<int,4,torch::RestrictPtrTraits>(),
            filters.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
            filters_transformed.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
            out_trans, in_trans, filter_size);
    }));
    */

    return grad_filters;
}
