import torch
import torch.nn as nn


class MyMultiply(nn.Module):
    def __init__(self):
        super(MyMultiply, self).__init__()

    @staticmethod
    def forward(a, b):
        return a*b

def backward_hook(module, grad_input, grad_output):
    print('module:', module)
    print('grad_input:', grad_input)
    print('grad_output:', grad_output)


def main():
    my_multiply = MyMultiply()
    my_multiply.register_backward_hook(backward_hook)

    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)

    c = my_multiply(a, b)

    c.backward()

if __name__ == '__main__':
    main()