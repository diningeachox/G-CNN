import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
import time

'''
    Converts a tuple (s, u, v) s - rotation, u - row, v - col, into a matrix group element
'''
def group_element(s, u, v, m=0, device='cpu'):
    m_data = [[pow(-1, m) * math.cos(s * math.pi / 2), pow(-1, m+1) * math.sin(s * math.pi / 2), u],
                [math.sin(s * math.pi / 2), math.cos(s * math.pi / 2), v],
                [0., 0., 1.]]
    matrix = torch.tensor(m_data, device=device)
    return matrix

def group_element_inverse(matrix):
    if (matrix[0][0] != 0):
        angle = math.atan(matrix[1][0] / matrix[0][0]) / (math.pi / 2)
    else:
        if (matrix[1][0] > 0):
            angle = 1
        elif (matrix[1][0] < 0):
            angle = 3
    return (int(angle), int(matrix[0][2]), int(matrix[1][2]))

#Custom forward and backward functions
class CustomForwardBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
          with torch.enable_grad():
              output = ctx.g3.forward(input)
              ctx.save_for_backward(input, output)
          return ctx.f3.forward(input)
    @staticmethod
    def backward(ctx, grad_output):
          input, output = ctx.saved_tensors
          output.backward(grad_output, retain_graph=True)
          return input.grad

class GConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=1, stride=1, in_transformations=1, out_transformations=4, group="p4", device='cpu'):
        super().__init__()
        #Filters stored as a K_l x K_{l-1} x S_{l_1} x n x n
        self.filters = torch.randn(out_channels, in_channels, in_transformations, filter_size, filter_size)

        #Filter transformations stored as a K_l x S_l x K_{l-1} x S_{l_1} x n x n
        self.filters_transformed = torch.zeros(out_channels, out_transformations, in_channels, in_transformations, filter_size, filter_size)


        '''
        Transformation rule: F+[i, s′, j, s, u, v] = F[i, j, s¯,  u¯,  v¯]
        Transformations are stored as [0, pi/2, pi, 3pi/2] CCW in p4,
                                    [0, pi/2, pi, 3pi/2, r, r * pi/2, r * pi, r * 3pi/2] CCW with reflections in p4m
        '''
        #Loop over all relevent elements
        m = (group == "p4m") #Add extra reflection parameter if group is p4m
        for i, j, s, s_prime, u, v in itertools.product(range(out_channels),
                                                        range(in_channels),
                                                        range(in_transformations),
                                                        range(out_transformations),
                                                        range(filter_size),
                                                        range(filter_size)):
            ref_prime = (s_prime > 3)
            ref = (s > 3)
            _s, _u, _v = group_element_inverse(torch.inverse(group_element(s_prime, 0, 0, m=ref_prime, device=device)) * group_element(s, u, v, m=ref, device=device))

            self.filters_transformed[i, s_prime, j, s, u, v] = self.filters[i, j, _s, _u, _v]

        #Reshape into normal convolutional filter
        self.filters_transformed = torch.reshape(self.filters_transformed, (out_channels * out_transformations, in_channels * in_transformations, filter_size, filter_size))
        self.filters_transformed.requires_grad = True
        #self.conv = nn.Conv2d(in_channels * transformations, out_channels * transformations, kernel_size=filter_size, stride=stride)

        #Register the transformed filters as trainable parameters
        self.register_parameter(name='filter_trans', param=torch.nn.Parameter(self.filters_transformed))

    def forward(self, x):

        x = F.conv2d(x, self.filters_transformed) #Convolve with the transformed filters
        return x

'''
Here we implement coset pooling over H and subsample over cosets gH, where
H = The 4 rotations around the origin if G = p4

'''
class GMaxPool2d(nn.Module):
    def __init__(self, group="p4", device='cpu'):
        super().__init__()
        self.group = group

    def forward(self, x):
        # x is of shape [channels x |H|, width, height]
        # out should be of shape [channels, width, height]

        if self.group == "p4":
            out = torch.zeros(x.shape[0], int(x.shape[1] / 4), x.shape[2], x.shape[3])
            for b, i, u, v in itertools.product(range(x.shape[0]),
                                                range(out.shape[0]),
                                                range(x.shape[2]),
                                                range(x.shape[3])):
                out[b, i, u, v] = max(x[b, i, u, v], x[b, i + out.shape[0], u, v], x[b, i + out.shape[0] * 2, u, v], x[b, i + out.shape[0] * 3, u, v])
        return out

if __name__ == "__main__":
    #Test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(1, 3, 23, 23, requires_grad = True).to(device)
    g_conv = GConv2d(3, 10, filter_size=3, device=device).to(device) #first layer with p4 group
    g_pool = GMaxPool2d(device=device).to(device)
    optimizer = torch.optim.Adam(g_conv.parameters(), lr=1e-4)

    start = time.time()
    y = g_conv(x)

    end = time.time()

    print(y.shape)
    y = g_pool(y)
    print(y.shape)

    print(f"Forward time: {end - start} s")

    target = torch.zeros_like(y)
    loss = nn.MSELoss()(y, target)


    print(loss.item())
    start = time.time()
    loss.backward()
    optimizer.step()
    end = time.time()
    print(f"Backward time: {end - start} s")
