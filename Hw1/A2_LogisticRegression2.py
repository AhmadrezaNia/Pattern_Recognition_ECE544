import torch
import torch.optim as optim

torch.manual_seed(1)
X = torch.Tensor([[-1, 1, 2],[1, 1, 1]])
y = torch.Tensor([-1, 1, 1])
w = torch.Tensor([[0.1],[0.1]]) #initialization
w.requires_grad = True
alpha = 1

optimizer = optim.SGD([w], lr=alpha)
optimizer.zero_grad()

for iter in range(100): # play with the number of iterations

    tmp = torch.exp(torch.matmul(torch.transpose(w,0,1),X)*(-y))
    ##############################
    ## loss is the same as f in A2_LogisticRegression.py
    ## Dimensions: loss (scalar)
    ##############################
    loss = torch.mean(torch.log(torch.ones_like(tmp))+tmp)

    loss.backward()
    print("Loss: {:.6f}; ||g||: {:.6f}".format(loss, torch.norm(w.grad)))

    ##############################
    ## Use two functions within the optimizer instance to perform the update step
    ##############################
    optimizer.step()
    optimizer.zero_grad()

print('Solution: {}'.format(w))
