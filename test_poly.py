import numpy as np
import polytope as pc
try:
    print("Polytope version:", pc.__version__)
except:
    print("No version")

# Test H -> V
A = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]]) # Box [-1, 1]^2
b = np.array([1, 1, 1, 1])
p = pc.Polytope(A, b)
print("Created Polytope")

try:
    v = pc.extreme(p)
    print("Vertices from pc.extreme:", v)
except Exception as e:
    print("pc.extreme failed:", e)

# Test V -> H
V = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
try:
    p2 = pc.qhull(V)
    print("H-rep from pc.qhull:", p2)
    print("A:", p2.A)
    print("b:", p2.b)
except Exception as e:
    print("pc.qhull failed:", e)
