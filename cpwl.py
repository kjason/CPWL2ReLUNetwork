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


class Pieces:
    def __init__(self) -> None:
        self.interior_point = []
        self.epsilon = []
        self.A = []
        self.b = []
        self.pattern = []
        self.q = 0

    def add(self, c: torch.Tensor, eps: float, A: torch.Tensor, b: torch.Tensor, p: int) -> None:
        self.interior_point.append(c)
        self.epsilon.append(eps)
        self.A.append(A)
        self.b.append(b)
        self.pattern.append(p)
        self.q = self.q + 1

    def __str__(self) -> str:
        s = "---Pieces---\n"
        for i in range(self.q):
            s = s + \
                f"#{i+1}: pattern id #{self.pattern[i]} has epsilon {self.epsilon[i]:.3f} at: {self.interior_point[i]}\n"
        return s[:-1]


class LinearComponents:
    def __init__(self, n: int, eps: float = 1e-6) -> None:
        self.n = n
        self.eps = eps
        self.k = 0
        self.__lc = []

    def __is_unique(self, x: torch.Tensor) -> bool:
        if len(self.__lc) == 0:
            return True
        else:
            return not True in (torch.mean(torch.abs(torch.cat(self.__lc, dim=1) - x), dim=0) < self.eps).tolist()

    def add(self, a: torch.Tensor, b: torch.Tensor) -> None:
        a = a.squeeze()
        b = b.squeeze()
        if len(a.shape) == 0:
            a = a.unsqueeze(0)
        if a.shape[0] != self.n:
            raise ValueError(
                "Dimension of a does not match the input dimension n")
        if len(b.shape) != 0:
            raise ValueError(
                "Invalid bias, b should be a scalar and its shape cannot be [0]")
        lc = torch.cat((a, b.unsqueeze(0)), dim=0).unsqueeze(1)
        if self.__is_unique(lc):
            self.__lc.append(lc)
            self.k = self.k + 1

    def get_A(self) -> torch.Tensor:
        lc = torch.cat(self.__lc, dim=1)
        return lc[:-1, :]

    def get_b(self) -> torch.Tensor:
        lc = torch.cat(self.__lc, dim=1)
        return lc[[-1], :]

    def value(self, x: torch.Tensor, i: int) -> torch.Tensor:
        return (torch.matmul(self.__lc[i][:-1].transpose(0, 1), x) + self.__lc[i][-1]).squeeze()

    def all_value(self, x: torch.Tensor) -> torch.Tensor:
        lc = torch.cat(self.__lc, dim=1)
        return (torch.matmul(lc[:-1, :].transpose(0, 1), x) + lc[[-1], :].transpose(0, 1)).squeeze()

    def get_a_b(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.__lc[i][:-1], self.__lc[i][-1]

    def __str__(self) -> str:
        s = "---LinearComponents---\n"
        for i in range(self.k):
            s = s + \
                f"#{i+1}: linear component has lc={self.__lc[i].squeeze()}\n"
        return s + f"The number of linear components is given by k={self.k}"


class PiecesInfo:
    def __init__(
        self,
        pieces: Pieces,
        pattern_curr: List,
        pattern_prev: List,
        sample_size: int,
        std_scale: float,
        std_growth_rate: float,
        growth_rate: int,
        interior_margin: float,
        min_margin: float
    ) -> None:
        self.pieces = pieces
        self.valid = pattern_curr == pattern_prev
        self.pattern_curr = pattern_curr
        self.pattern_prev = pattern_prev
        self.sample_size = sample_size
        self.std_scale = std_scale
        self.growth_rate = growth_rate
        self.std_growth_rate = std_growth_rate
        self.interior_margin = interior_margin
        self.min_margin = min_margin

    def __str__(self) -> str:
        s = self.pieces.__str__()
        if self.valid is True:
            s = s + \
                f"\n{self.pieces.q} pieces are verified with sample_size=({self.sample_size//self.growth_rate},{self.sample_size}), std_scale=({self.std_scale/self.std_growth_rate},{self.std_scale}), interior_margin={self.interior_margin}, and min_margin={self.min_margin}"
        else:
            s = s + \
                f"\n{self.pieces.q} pieces tested by sample_size={self.sample_size}, std_scale={self.std_scale}, interior_margin={self.interior_margin}, and min_margin={self.min_margin} are invalid. len(pattern_curr)={len(self.pattern_curr)}, len(pattern_prev)={len(self.pattern_prev)}"
        return s


class RandomCPWL:
    def __init__(
        self,
        n: int,
        w: int
    ) -> None:
        if n < 1 or w < 0:
            raise ValueError(
                "invalid n or w, n and w must be larger than or equal to 1")
        self.n = n  # input dimension
        self.w = w  # number of hidden neurons, if w=0, then the CPWL function is a linear function
        """
        Randomly generate a CPWL function using a simple 1-layer ReLU network
        """
        if self.w == 0:
            self.a = math.sqrt(1.0/n) * torch.randn(1, n)
            self.c = math.sqrt(1.0/n) * torch.randn(1)
        else:
            self.W = math.sqrt(1.0/n) * torch.randn(w, n)
            self.b = math.sqrt(1.0/n) * torch.randn(w, 1)
            self.a = math.sqrt(2.0/w) * torch.randn(1, w)
            self.c = math.sqrt(2.0/w) * torch.randn(1)

    def preact(self, x: torch.Tensor) -> torch.Tensor:
        if self.w == 0:
            raise ValueError("Preact should not be used with w=0")
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        return torch.matmul(self.W, x) + self.b

    def value(self, x: torch.Tensor) -> torch.Tensor:
        if self.w == 0:
            if len(x.shape) == 1:
                x = x.unsqueeze(1)
            return (torch.matmul(self.a, x) + self.c).squeeze()
        else:
            y = self.preact(x)
            relu_y = torch.nn.functional.relu(y)
            z = torch.matmul(self.a, relu_y)
            return (z + self.c).squeeze()


class ObservableFunction:
    def __init__(self, f: RandomCPWL) -> None:
        self.__f = f
        self.n = f.n

    def value(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            return self.__f.value(x.transpose(0, 1))
        else:
            return self.__f.value(x)


def get_pieces_per_sample_size(
    f: RandomCPWL,
    sample_size: int = 2**16,
    std_scale: float = 200.0,
    interior_margin: float = 1e-8,
    min_margin: float = 1e-7
) -> Pieces:
    P = Pieces()
    if f.w == 0:
        P.add(torch.zeros(f.n), 1.0, torch.zeros(1, f.n), torch.zeros(1, 1), 0)
        return P
    random_input = std_scale * (torch.rand(f.n, sample_size) - 0.5)
    distance = torch.abs(torch.matmul(f.W, random_input)+f.b) / \
        torch.linalg.vector_norm(f.W, dim=1).unsqueeze(1)
    min_distance = torch.min(distance, dim=0)[0]
    act_pattern = torch.sum((f.preact(random_input) > 0) * torch.Tensor(
        [2**i for i in range(f.w)]).unsqueeze(1), dim=0).to(torch.int32)
    unique_pattern = torch.unique(act_pattern)
    for i in unique_pattern:
        g = act_pattern == i
        max_distance, index = torch.max(min_distance[g], dim=0)
        center = random_input[:, g]
        center = center[:, index]
        epsilon = max_distance - interior_margin
        if epsilon > min_margin:
            preact = f.preact(center) > 0
            s = ((preact == False) * -1 + (preact == True) * 1)
            P.add(center, epsilon.cpu().item(), -f.W * s, f.b * s, i)
    return P


def get_pieces(
    f: RandomCPWL,
    init_sample_size: int = 2**16,
    max_sample_size: int = 2**24,
    std_scale: float = 200.0,
    std_growth_rate: float = 1.0,
    sample_size_growth_rate: int = 4,
    interior_margin: float = 1e-8,
    min_margin: float = 1e-7
) -> PiecesInfo:
    pattern_prev = []
    sample_size = init_sample_size
    normalized_max_sample_size = max_sample_size / sample_size_growth_rate
    X = get_pieces_per_sample_size(
        f, sample_size, std_scale, interior_margin, min_margin)
    pattern_curr = X.pattern
    while pattern_curr != pattern_prev and sample_size <= normalized_max_sample_size:
        sample_size = sample_size * sample_size_growth_rate
        std_scale = std_scale * std_growth_rate
        X = get_pieces_per_sample_size(
            f, sample_size, std_scale, interior_margin, min_margin)
        pattern_prev = pattern_curr
        pattern_curr = X.pattern
    piece_info = PiecesInfo(X, pattern_curr, pattern_prev, sample_size,
                            std_scale, std_growth_rate, sample_size_growth_rate, interior_margin, min_margin)
    return piece_info
