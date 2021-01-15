import torch
# forward           # backward (normal)     #  backward (inplace operation)
# a(2)  b(3)        # a(3) b(2)             # a(300)b(200)     
#  \   /            #  \   /                #  \   /         
#    *              #    *                  #    *           
#    |              #    |                  #    |           
#    c(6)  d(4)     #    c(1) d(100)        #    c(100)d(100)          
#     \   /         #     \   /             #     \   /      
#       +           #       +               #       +      
#       |           #       |               #       |     
#       e(10)       #       e(1)            #       e(1)     

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a * b

c.retain_grad()

d = torch.tensor(4.0, requires_grad=True)


def d_hook(grad):
    # grad *= 100 # <= inplace operation
    return grad * 100 # <= normal


d.register_hook(d_hook)
d.retain_grad()

e = c + d

e.backward()

print('d:', d._grad) 
print('c:', c._grad)
print('b:', b._grad)
print('a:', a._grad)

# inplace operation case
# d: tensor(100.)
# c: tensor(100.)
# b: tensor(200.)
# a: tensor(300.)
# normal case
# d: tensor(100.)
# c: tensor(1.)
# b: tensor(2.)
# a: tensor(3.)