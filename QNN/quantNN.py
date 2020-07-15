import torch
from torch import nn

class QuantValue(nn.Module):
    """
    Quantization
    """
    def __init__(self, N, m):
        super(QuantValue, self).__init__()
        self.N = N
        self.m = m
        self.quant = QuantValue_F.apply

    def forward(self, x):
        return self.quant(x, self.N, self.m)
    
    def extra_repr(self):
        s = ('N = %d, m = %d'%(self.N, self.m))
        return s

class QuantValue_F(torch.autograd.Function):
    """
    res = clamp(round(input/pow(2,-m)) * pow(2, -m), -pow(2, N-1), pow(2, N-1) - 1)
    """

    @staticmethod 
    def forward(ctx, inputs, N, m):
        Q = pow(2, N - 1) - 1
        delt = pow(2, - m)
        M = (inputs.to(torch.float32)/delt).round().clamp(-Q-1,Q)
        ctx.save_for_backward((((M <= -Q-1).to(torch.float) + (M >= Q).to(torch.float)) == 0).to(torch.float), M)
        return delt*M
    
    @staticmethod
    def backward(ctx, g):
        mask, M = ctx.saved_tensors
        right = (mask == 0).to(torch.float) * (M > 0).to(torch.float32)
        left  = (mask == 0).to(torch.float) * (M < 0).to(torch.float32)
        # return g * mask + ((g * left).abs() - (g * right).abs()) * 1e-4, None, None
        return g * mask, None, None


class QLinear(nn.Module):
    """
        Quantized Linear
    """
    def __init__(self,  N, m, in_features, out_features, bias = True, is_print = False):
        super(QLinear, self).__init__()
        self.linear   = nn.Linear(in_features, out_features, bias)
        #self.linear.weight.data.normal_(0, 0.05 * pow(2, N-m))
        self.quant    = QuantValue(N,m)
        self.is_print = is_print 

    def forward(self, x):
        res = self.quant(self.linear(x))
        if self.is_print:
            print(res.abs().max())
        return res

class QConv2d(nn.Module):
    """
        Quantized Conv2d
    """
    def __init__(self, N, m, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', is_print = False):
        super(QConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        #self.conv.weight.data.normal_(0, 1e-3 * pow(2, N-m))
        self.quant  = QuantValue(N,m)
        self.is_print = is_print


    def forward(self, x):
        res = self.quant(self.conv(x))
        if self.is_print:
            print(res.abs().max())
        return res


if __name__ == "__main__":
    conv = QConv2d(8,8,1,1,1)
    model = QLinear(8, 8, 1,2)
    a = torch.Tensor([0.125]).requires_grad_()
    optimizer = torch.optim.Adam(model.parameters())
    for _ in range(10):
        optimizer.zero_grad()
        c = conv(a.view(1,1,1,1))
        res = model(c).sum()
        print(res.item())
        res.backward()
        optimizer.step()
        