#include <torch/extension.h>
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <vector>
#include <math.h>       /* pow, sin, atan, cos */

const float pi = 3.1415926;

torch::Tensor d_sigmoid(torch::Tensor z) {
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
}

/*
    Converts a tuple (s, u, v) s - rotation, u - row, v - col, into a matrix group element
*/
torch::Tensor gcnn_group_element(int s, int u, int v, int m){

    float m_data[3][3] =
    {
        {pow(-1.0, m) * cos(s * pi / 2.), 1.0 * pow(-1.0, m+1) * sin(s * pi / 2.), u},
        {sin(s * pi / 2.), cos(s * pi / 2.), v},
        {0., 0., 1.}
    };
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor matrix = torch::from_blob(m_data, {3, 3}, options);

    return matrix;
}

std::vector<int> gcnn_group_element_inverse(torch::Tensor matrix){
    int angle = 0;
    if (matrix[0][0].item<float>() != 0){
        angle = (int) (atan(matrix[1][0].item<float>() / matrix[0][0].item<float>()) / pi * 2.0);
    }
    else {
        if (matrix[1][0].item<float>() > 0){
            angle = 1;
        }
        else {
            angle = 3;
        }
    }

    return {angle, static_cast<int>(matrix[0][2].item<float>()), static_cast<int>(matrix[1][2].item<float>())};
}

std::vector<int> calc_indices(int s_prime, int s, int u, int v, int ref_prime, int ref){
    torch::Tensor first = gcnn_group_element(s_prime, 0, 0, ref_prime);
    torch::Tensor second = gcnn_group_element(s, u, v, ref);
    torch::Tensor transform = torch::mm(torch::inverse(first), second);
    return gcnn_group_element_inverse(transform);
}


torch::Tensor transform_filter(int out_channels, int out_trans, int in_channels, int in_trans, int filter_size,
                                            torch::Tensor ind1, torch::Tensor ind2, torch::Tensor ind3,
                                            torch::Tensor filters){
    /*****
    Transformation rule: F+[i, s′, j, s, u, v] = F[i, j, s¯,  u¯,  v¯]
    Transformations are stored as [0, pi/2, pi, 3pi/2] CCW in p4,
                                [0, pi/2, pi, 3pi/2, r, r * pi/2, r * pi, r * 3pi/2] CCW with reflections in p4m
    *****/

    torch::Tensor filters_transformed = torch::zeros({out_channels * out_trans, in_channels * in_trans, filter_size, filter_size});

    for (int i = 0; i < out_channels; i++){
        for (int j = 0; j < in_channels; j++){
            for (int s_prime = 0; s_prime < out_trans; s_prime++){
                for (int s = 0; s < in_trans; s++){
                    for (int u = 0; u < filter_size; u++){
                        for (int v = 0; v < filter_size; v++){

                            int _s = ind1[s_prime][s][u][v].item<int>();
                            int _u = ind2[s_prime][s][u][v].item<int>();
                            int _v = ind3[s_prime][s][u][v].item<int>();
                            filters_transformed[i * s_prime][j * s][u][v] = filters[i][j][_s][_u][_v];
                        }
                    }
                }
            }
        }
    }

    return filters_transformed;
}

torch::Tensor gcnn_pool_forward(torch::Tensor x) {

    auto out = torch::zeros({x.sizes()[0], x.sizes()[1] / 4, x.sizes()[2], x.sizes()[3]}, torch::kF32);
    /***********
        Transformation rule: F+[i, s′, j, s, u, v] = F[i, j, s¯,  u¯,  v¯]
        Transformations are stored as [0, pi/2, pi, 3pi/2] CCW in p4,
                                    [0, pi/2, pi, 3pi/2, r, r * pi/2, r * pi, r * 3pi/2] CCW with reflections in p4m
    ***********/
    for (int b = 0; b < x.sizes()[0]; b++){
        for (int i = 0; i < x.sizes()[0]; i++){
            for (int u = 0; u < x.sizes()[2]; u++){
                for (int v = 0; v < x.sizes()[3]; v++){
                    out[b][i][u][v] = std::max({x[b][i][u][v].item<float>(),
                                                x[b][i + x.sizes()[0]][u][v].item<float>(),
                                                x[b][i + x.sizes()[0] * 2][u][v].item<float>(),
                                                x[b][i + x.sizes()[0] * 3][u][v].item<float>()
                                            });
                }
            }
        }
    }
    return out;

}

std::vector<at::Tensor> gcnn_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor filters,
    torch::Tensor filters_transformed,
    torch::Tensor ind1,
    torch::Tensor ind2,
    torch::Tensor ind3
){
    return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("calc_indices", &calc_indices, "Calculate Indices");
  m.def("transform_filter", &transform_filter, "Transform the filters equivariantly");
  m.def("gmaxpool_forward", &gcnn_pool_forward, "GMaxPool forward");
  //m.def("group_element", &gcnn_group_element, "GCNN group element");
  //m.def("group_element_inverse", &gcnn_group_element_inverse, "GCNN group element inverse");
  //m.def("forward", &gcnn_forward, "GCNN forward");
  m.def("backward", &gcnn_backward, "GCNN backward");
}
