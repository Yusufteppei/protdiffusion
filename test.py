import torch
import numpy as np
from rfdiffusion.geometry.frames import rigid_from_3_points

a = torch.tensor([-4, -3, 4], dtype=torch.float64)
b = torch.tensor([3, 4, 3], dtype=torch.float64)
print(f"a, b - {a}, {b}")

a_ = a / torch.linalg.norm(a)
b_ = b / torch.linalg.norm(b)
print(f"a_, b_ - {a_}, {b_}")

e1 = a_
print(f"e1 - {e1}")

x1 = torch.dot(e1, b_)*e1
print(f"x1 - {x1}")
e2 = (b_ - x1)
print(f"e2 - {e2}")
print(torch.dot(e1, e2))
e3 = torch.cross(e1, e2)
print(f"e3 - {e3}")
print(f"e3 . e2 = {torch.dot(e3, e2)}")


N = torch.Tensor([12, 13, 5])
C = torch.Tensor([17, 0, 2])
CA = torch.Tensor([11, 3, 9])

X = rigid_from_3_points(N=N, CA=CA, C=C)
print(torch.dot(X[2], X[1]))