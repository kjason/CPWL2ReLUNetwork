"""
Created on Friday Sep 30 2022

@author: Kuan-Lin Chen

K.-L. Chen, H. Garudadri, and B. D. Rao. Improved Bounds on Neural Complexity for
Representing Piecewise Linear Functions. In Advances in Neural Information Processing
Systems, 2022.
"""
from typing import List
import math
import torch
from scipy.optimize import linprog
from cpwl import Pieces, ObservableFunction, LinearComponents

torch.set_default_dtype(torch.double)


def find_all_distinct_linear_components(f: ObservableFunction, X: Pieces) -> LinearComponents:
    n = f.n
    lc = LinearComponents(f.n)
    q = X.q
    for i in range(q):
        x_0 = X.interior_point[i]
        y_0 = f.value(x_0)
        S = X.epsilon[i] * torch.eye(n)
        x = x_0.unsqueeze(1) + S
        z = f.value(x.transpose(0, 1)) - y_0
        a = torch.matmul((1.0/X.epsilon[i]) * torch.eye(n), z.reshape(-1, 1))
        b = y_0 - torch.matmul(a.transpose(0, 1), x_0.unsqueeze(1))
        lc.add(a, b)
    return lc


class ReLUNet(torch.nn.Module):
    def __init__(self, k: List[int]) -> None:
        super(ReLUNet, self).__init__()
        if len(k) < 2 or min(k) < 1:
            raise ValueError("Invalid k")
        self.k = k
        self.l = len(k) - 1
        self.net = torch.nn.ModuleList()
        for i in range(self.l):
            self.net.append(torch.nn.Linear(
                in_features=k[i], out_features=k[i+1], bias=True))

    def set_param(self, weight: torch.Tensor, bias: torch.Tensor, l: int, requires_grad: bool = False) -> None:
        self.net[l].weight = torch.nn.parameter.Parameter(
            weight, requires_grad=requires_grad)
        self.net[l].bias = torch.nn.parameter.Parameter(
            bias, requires_grad=requires_grad)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for i in range(self.l-1):
            z = torch.nn.functional.relu(self.net[i](z))
        return self.net[-1](z)

    def forward(self, x: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        if requires_grad is True:
            return self.value(x)
        else:
            with torch.no_grad():
                return self.value(x)


class ExtremeAffineReLUNet(ReLUNet):
    A = torch.Tensor([[-1, 1], [1, 0], [-1, 0]])
    B = torch.Tensor([[1, 1, -1]])
    C = torch.Tensor([[1], [-1]])

    def __init__(self, lc: LinearComponents, min_affine: bool) -> None:
        if lc.k < 1:
            raise ValueError(
                "Invalid linear components, lc.k must be larger than 0")
        self.min_affine = min_affine
        l = math.ceil(math.log2(lc.k))+1
        self.l = l
        k = [lc.n]
        c = [lc.k]
        for i in range(1, l):
            if c[i-1] % 2 == 0:
                c.append(c[i-1] // 2)
                k.append(3*c[-1])
            else:
                c.append((c[i-1]+1) // 2)
                k.append(3*c[-1]-1)
        k.append(1)
        super(ExtremeAffineReLUNet, self).__init__(k)
        W_1 = lc.get_A().transpose(0, 1)
        b_1 = lc.get_b().transpose(0, 1)
        self.set_param(W_1, b_1.reshape(-1), 0)
        if l > 1:
            if c[0] % 2 == 0:
                tmp = torch.block_diag(*((self.A,) * c[1]))
                W_1 = torch.matmul(tmp, W_1)
                b_1 = torch.matmul(tmp, b_1)
            else:
                tmp = torch.block_diag(torch.block_diag(
                    *((self.A,) * (c[1]-1))), self.C)
                W_1 = torch.matmul(tmp, W_1)
                b_1 = torch.matmul(tmp, b_1)
            self.set_param(W_1, b_1.reshape(-1), 0)
            self.set_param(self.B, torch.Tensor([0.0]), -1)
        if l > 2:
            for i in range(2, l):
                if c[i-1] % 2 == 0:
                    T = torch.block_diag(*((self.A,) * c[i]))
                else:
                    T = torch.block_diag(torch.block_diag(
                        *((self.A,) * (c[i]-1))), self.C)
                if c[i-2] % 2 == 0:
                    W_i = torch.matmul(
                        T, torch.block_diag(*((self.B,) * c[i-1])))
                else:
                    W_i = torch.matmul(T, torch.block_diag(torch.block_diag(
                        *((self.B,) * (c[i-1]-1))), self.C.transpose(0, 1)))
                b_i = torch.zeros(k[i], 1)
                self.set_param(W_i, b_i.reshape(-1), i-1)
        if min_affine is True:
            self.set_param(-self.net[0].weight, -self.net[0].bias, 0)
            self.set_param(-self.net[-1].weight, -self.net[-1].bias, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(x).reshape(-1)


class IdentityReLUNet(ReLUNet):
    A = torch.Tensor([[1], [-1]])
    B = torch.Tensor([[1, -1]])
    C = torch.Tensor([[1, -1], [-1, 1]])

    def __init__(self, n: int, l: int) -> None:
        if n < 1 or l < 1:
            raise ValueError("Invalid n or l")
        self.l = l
        self.n = n
        k = [n]
        b = []
        for i in range(1, l):
            k.append(2*n)
            b.append(torch.zeros(k[-1]))
        k.append(n)
        b.append(torch.zeros(n))
        super(IdentityReLUNet, self).__init__(k)
        if l == 1:
            self.set_param(torch.eye(n), b[0], 0)
        else:
            self.set_param(torch.block_diag(*((self.A,) * k[0])), b[0], 0)
            self.set_param(torch.block_diag(*((self.B,) * k[-1])), b[-1], -1)
        if l > 2:
            for i in range(1, l-1):
                self.set_param(torch.block_diag(*((self.C,) * n)), b[i], i)


class CompositedReLUNet(ReLUNet):
    def __init__(self, g_1: ReLUNet, g_2: ReLUNet) -> None:
        super(CompositedReLUNet, self).__init__(g_1.k[:-1]+g_2.k[1:])
        l_1 = g_1.l - 1
        for i in range(self.l):
            if i < l_1:
                self.set_param(g_1.net[i].weight, g_1.net[i].bias, i)
            elif i == l_1:
                self.set_param(torch.matmul(g_2.net[0].weight, g_1.net[i].weight), torch.matmul(
                    g_2.net[0].weight, g_1.net[i].bias)+g_2.net[0].bias, i)
            else:
                self.set_param(g_2.net[i-l_1].weight, g_2.net[i-l_1].bias, i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(x).squeeze()


class ConcatenatedReLUNet(ReLUNet):
    def __init__(self, g: List[ReLUNet]) -> None:
        len_g = len(g)
        l = max([g[j].l for j in range(len_g)])
        for j in range(len_g):
            if g[j].l < l:
                m = g[j].k[-1]
                g_c = IdentityReLUNet(m, l-g[j].l+1)
                g[j] = CompositedReLUNet(g[j], g_c)
        k = [g[0].k[0], sum([g[j].k[1] for j in range(len_g)])]
        for i in range(2, l+1):
            k.append(sum([g[j].k[i] for j in range(len_g)]))
        super(ConcatenatedReLUNet, self).__init__(k)
        self.set_param(torch.cat([g[j].net[0].weight for j in range(
            len_g)], dim=0), torch.cat([g[j].net[0].bias for j in range(len_g)], dim=0), 0)
        for i in range(1, l):
            self.set_param(torch.block_diag(*[g[j].net[i].weight for j in range(
                len_g)]), torch.cat([g[j].net[i].bias for j in range(len_g)], dim=0), i)


def CPWLReLUNet(
    f: ObservableFunction,
    p: Pieces,
    eps: float = 1e-6
) -> ConcatenatedReLUNet:
    lc = find_all_distinct_linear_components(f, p)
    q = p.q
    n = f.n
    k = lc.k
    A = []
    vi = []
    identity_lc = LinearComponents(q)
    for i in range(q):
        A.append(LinearComponents(n))
        interior_point = p.interior_point[i]
        # just check 1 point here to obtain the linear component of the unknown CPWL function on this piece
        # the algorithm needs to check n+1 affinely independent points in the worst-case scenario, which is unlikely
        min_value, idx = torch.min(torch.abs(torch.Tensor(
            [lc.value(interior_point, j) - f.value(interior_point) for j in range(k)])), dim=0)
        if min_value >= eps:
            raise ValueError(
                f"Invalid linear componeents, min_value={min_value:.2e} is larger than the specified eps={eps:.2e}")
        a_f, b_f = lc.get_a_b(idx)
        for j in range(k):
            if j == idx:
                A[-1].add(*lc.get_a_b(j))
            else:
                a, b = lc.get_a_b(j)
                c = (a - a_f).tolist()
                bounds = [(None, None) for _ in range(n)]
                result = linprog(c=c, A_ub=p.A[i].tolist(
                ), b_ub=p.b[i].tolist(), bounds=bounds)
                if result.success and result.fun + b - b_f + eps >= 0:
                    A[-1].add(*lc.get_a_b(j))
        vi.append(ExtremeAffineReLUNet(A[-1], True))
        s = torch.zeros(q)
        s[i] = 1.0
        identity_lc.add(s, torch.Tensor([0]))
    v = ConcatenatedReLUNet(vi)
    u = ExtremeAffineReLUNet(identity_lc, False)
    return CompositedReLUNet(v, u), k
