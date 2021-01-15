import torch
# forward           # backward
# a(2)  b(3)        # a(30) b(20)        
#  \   /            #  \   /    
#    *              #    *
#    |              #    |
#    c(6)  d(4)     #    c(10) d(112)        
#     \   /         #     \   /    
#       *           #       *    
#       |           #       |    
#       e(24)       #       e(2)        

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

a.register_hook(lambda grad: print('a:', grad))
b.register_hook(lambda grad: print('b:', grad))

c = a*b

def c_hook(grad):
    print(grad)
    return grad+2

c.register_hook(c_hook)
c.register_hook(lambda grad: print('c:', grad))
c.retain_grad() # retain_grad() : leaf node (a,b)에서는 gradient가 자동으로 저장되지만 intermediate node에서는 자동으로 저장되지 않는다. gradient를 저장하기 위한 함수.

d = torch.tensor(4.0, requires_grad=True)

d.register_hook(lambda grad: grad + 100)
d.register_hook(lambda grad: print('d:', grad))

e = c*d

e.retain_grad()
e.register_hook(lambda grad: grad * 2)
e.retain_grad() # 두번째는 영향을 주지 않는다.

e.backward()

print(c._grad) # c.retain_grad() 유무에 따라 값을 출력하거나 None을 출력.
print(d._grad)
print(e._grad) # 두번째 e.retain_grad()는 영향을 주지 않으므로 (2.)가 아니라 (1.)을 출력
