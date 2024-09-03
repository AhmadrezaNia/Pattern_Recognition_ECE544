import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(1)
X = torch.tensor([[1, 0, 0, -1],[0, 1, 0, -1]], dtype=torch.float32)
y = torch.tensor([1, 1, -1, -1], dtype=torch.float32)
# initialize w and b
w = torch.tensor([0.0, 0.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
alpha = 0.001
C = 1

optimizer = optim.SGD([w,b], lr=alpha, weight_decay=0)
optimizer.zero_grad()

grads = []
losses = []

for iter in range(10000):
    if iter==0:
        print('1-y(w^T*x+b): {}'.format(1 - y*(torch.matmul(X.T, w)+b)))
    ##############################
    ## Complete this line which is our cost function
    ## Dimensions: loss (scalar)
    ##############################

    New_term = 1 - y*(torch.matmul(X.T, w)+b)
    loss = C/2 * torch.matmul(w, w) + torch.sum(torch.clamp(New_term, min=0))
    loss.backward()
    gn = torch.norm(w.grad)**2 + torch.norm(b.grad)**2
    print("Iter: %d; Loss: %f; ||Grad||: %f" % (iter, loss, gn))
    optimizer.step()
    optimizer.zero_grad()

    losses.append(loss.item())
    grads.append(gn)

    print('w: {}'.format(w.data))
    print('b: {}'.format(b.data))

plt.figure()
plt.plot(losses, label='Loss')
plt.title('Loss')
plt.show()
plt.figure()
plt.plot(grads, label='Grad')
plt.title('||Grad||')
plt.show()