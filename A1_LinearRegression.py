import torch

x = torch.Tensor([[1],[2]])
y = torch.Tensor([[1],[1]])
print('x: {}'.format(x))

# Form the X matrix
X = torch.cat((x, torch.ones_like(x)),1)
print('X: {}'.format(X))

# Solution 1
##############################
## Fill in the arguments
##############################
res1 = torch.linalg.lstsq(X,y)
print('Solution 1: {}'.format(res1.solution))

# Solution 2
XTX = torch.matmul(torch.transpose(X, 0, 1), X)
XTy = torch.matmul(torch.transpose(X, 0, 1), y)

print('X^TX: {}'.format(XTX))
print('X^Ty: {}'.format(XTy))

##############################
## How to compute l and r?
## Dimensions: l (2x2); r (2x1)
##############################
l = XTX
r = XTy
res2 = torch.linalg.solve(l,r)
print('Solution 2: {}'.format(res2))

# Solution 3
##############################
## What is l and r?
## Dimensions: l (2x2); r (2x1)
##############################
l = XTX
r = XTy
res3 = torch.matmul(torch.linalg.inv(l),r)
print("Solution 3: {}".format(res3))

