import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
import time

'''
    Converts a tuple (s, u, v) s - rotation, u - row, v - col, into a matrix group element
'''
def group_element(s, u, v, m=0):
    m_data = [[pow(-1, m) * math.cos(s * math.pi / 2), pow(-1, m+1) * math.sin(s * math.pi / 2), u],
                [math.sin(s * math.pi / 2), math.cos(s * math.pi / 2), v],
                [0., 0., 1.]]
    matrix = torch.tensor(m_data)
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
class GConvFunctions(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filters, in_channels, out_channels, in_trans, out_trans, filter_size, device='cpu'):
        #ctx.save_for_backward(input, filters)

        #Save dimensions to ctx
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels
        ctx.in_trans = in_trans
        ctx.out_trans = out_trans
        ctx.filter_size = filter_size

        filters_transformed = torch.zeros(out_channels, out_trans, in_channels, in_trans, filter_size, filter_size, device=device)
        '''
        Transformation rule: F+[i, s′, j, s, u, v] = F[i, j, s¯,  u¯,  v¯]
        Transformations are stored as [0, pi/2, pi, 3pi/2] CCW in p4,
                                    [0, pi/2, pi, 3pi/2, r, r * pi/2, r * pi, r * 3pi/2] CCW with reflections in p4m
        '''
        for i, s_prime, j, s, u, v in itertools.product(range(out_channels),
                                                        range(out_trans),
                                                        range(in_channels),
                                                        range(in_trans),
                                                        range(filter_size),
                                                        range(filter_size)):
            ref_prime = (s_prime > 3)
            ref = (s > 3)
            _s, _u, _v = group_element_inverse(torch.inverse(group_element(s_prime, 0, 0, m=ref_prime)) * group_element(s, u, v, m=ref))

            filters_transformed[i, s_prime, j, s, u, v] = filters[i, j, _s, _u, _v]
        filters_transformed = torch.reshape(filters_transformed, (out_channels * out_trans, in_channels * in_trans, filter_size, filter_size))
        ctx.save_for_backward(input, filters, filters_transformed)
        return F.conv2d(input, filters_transformed)

    @staticmethod
    def backward(ctx, grad_output):
        print('Custom backward called!')
        input, filters, filters_transformed = ctx.saved_tensors

        print(grad_output.shape)
        print(input.shape)
        print(filters.shape)

        #Calculate dw = x * grad_output channel by channel
        '''
        dw = torch.zeros_like(filters_transformed)
        for i in range(dw.shape[1]): #in_channels
            for j in range(dw.shape[0]): #out_channels
                #print(input[:, i, :, :].shape, grad_output[:, j, :, :].shape)
                dw[] += F.conv2d(input[:, i, :, :].unsqueeze(1), grad_output[:, j, :, :].unsqueeze(1))
        print(dw.shape)
        '''

        grad_input = None
        grad_filters = torch.zeros_like(filters)

        grad_input = torch.nn.grad.conv2d_input(input.shape, filters_transformed, grad_output)
        grad_filters_trans = torch.nn.grad.conv2d_weight(input, filters_transformed.shape, grad_output)
        print(grad_filters_trans.shape)
        #grad_output.backward(filters)
        for i, s_prime, j, s, u, v in itertools.product(range(ctx.out_channels),
                                                        range(ctx.out_trans),
                                                        range(ctx.in_channels),
                                                        range(ctx.in_trans),
                                                        range(ctx.filter_size),
                                                        range(ctx.filter_size)):
            ref_prime = (s_prime > 3)
            ref = (s > 3)
            _s, _u, _v = group_element_inverse(torch.inverse(group_element(s_prime, 0, 0, m=ref_prime)) * group_element(s, u, v, m=ref))

            #Update grad_filters according to grad_output
            grad_filters[i, j, _s, _u, _v] += grad_filters_trans[i * ctx.out_trans + s_prime, j * ctx.in_trans + s, u, v]


        #7 parameters in forward() so need to pad the number of fields returned
        return grad_input, grad_filters, None, None, None, None, None

class GConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=1, stride=1, in_transformations=1, out_transformations=4, group="p4", device='cpu'):
        super().__init__()
        #Filters stored as a K_l x K_{l-1} x S_{l_1} x n x n
        self.filters = torch.randn(out_channels, in_channels, in_transformations, filter_size, filter_size, device=device)
        self.filters.requires_grad = True
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_trans = in_transformations
        self.out_trans = out_transformations
        self.filter_size = filter_size
        #Filter transformations stored as a K_l x S_l x K_{l-1} x S_{l_1} x n x n


        #Register the transformed filters as trainable parameters
        self.register_parameter(name='filter', param=torch.nn.Parameter(self.filters))

    def forward(self, x):
        #return F.conv2d()
        return GConvFunctions.apply(x, self.filters, self.in_channels, self.out_channels, self.in_trans, self.out_trans, self.filter_size)



'''
Here we implement coset pooling over H and subsample over cosets gH, where
H = The 4 rotations around the origin if G = p4

'''
class GMaxPool2d(nn.Module):
    def __init__(self, group="p4"):
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
    print(device)
    x = torch.randn(1, 3, 23, 23, requires_grad = True).to(device)
    g_conv = GConv2d(3, 10, filter_size=3).to(device) #first layer with p4 group
    g_pool = GMaxPool2d().to(device)
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
    end = time.time()

    optimizer.step()
    print(f"Backward time: {end - start} s")
