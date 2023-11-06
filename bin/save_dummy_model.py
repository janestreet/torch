import torch

# This script can be run to generate dummy.pt, used in tensor_tools as an example model
# to demonstrate initializing a module on any device.


class DummyModel(torch.nn.Module):
    def __init__(self, a):
        super(DummyModel, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(a))

    def forward(self, x):
        return self.a * torch.matmul(x, x)


model = Bar(3.0)
torch.jit.script(model).save("bar.pt")
