"""
Created on Friday Sep 30 2022

@author: Kuan-Lin Chen

K.-L. Chen, H. Garudadri, and B. D. Rao. Improved Bounds on Neural Complexity for
Representing Piecewise Linear Functions. In Advances in Neural Information Processing
Systems, 2022.
"""
import time
import csv
import math
import torch
from cpwl import RandomCPWL, ObservableFunction, get_pieces
from relu_net import CPWLReLUNet


def thm2_bound(k: int, q: int):  # the upper bound given by Theorem 2
    log2k = math.ceil(math.log2(k))
    log2q = math.ceil(math.log2(q))
    return (3 * (2 ** log2k) + 2 * log2k - 3) * q + 3 * (2 ** log2q) - 2 * log2k - 3


def find_a_relu_net(
    n: int,
    w: int,
    seed: int,
    d: int = 100,
    batch_size: int = 10000,
    init_sample_size: int = 2**18,
    max_sample_size: int = 2**24,
    sample_size_growth_rate: int = 4,
    max_allow_error: float = 1e-10,
    min_allow_sdr: float = 150.0
):
    torch.manual_seed(seed)
    # generate a random CPWL function
    random_cpwl = RandomCPWL(n=n, w=w)
    # find all the pieces
    piece_info = get_pieces(f=random_cpwl, init_sample_size=init_sample_size,
                            max_sample_size=max_sample_size, sample_size_growth_rate=sample_size_growth_rate)
    if piece_info.valid is False:
        print((f"piece_info.valid is False, get_pieces with sample_size={piece_info.sample_size} and "
               f"std_scale={piece_info.std_scale} may not cover the whole space."))
        return False, None, None
    # Algorithm 1 only needs to know the pieces and the output value of the function when feeding an input
    f = ObservableFunction(random_cpwl)
    # measure the running time of Algorithm 1
    tic = time.time()
    cpwlnet, k = CPWLReLUNet(f, piece_info.pieces)
    toc = time.time()
    # find the maximum error and minimum SDR between the given CPWL function and the ReLU network found by Alrogithm 1
    x = 2 * d * (torch.rand(batch_size, n) - 0.5)
    y1 = f.value(x)
    y2 = cpwlnet(x)
    e = torch.abs(y2 - y1)
    max_error = torch.max(e).item()
    min_sdr = torch.min(
        10 * torch.log10((torch.abs(y1)**2) / (torch.abs(e)**2))).item()
    success_precision = max_error < max_allow_error and min_sdr > min_allow_sdr
    # check the upper bound of the number of hidden neurons given by Theorem 2
    # num_neurons is the exact number of hidden neurons in the ReLU network found by Algorithm 1
    num_neurons = sum(cpwlnet.k[1:-1])
    # thm2_bound computes the upper bound given by Theorem 2
    upper_bound = thm2_bound(k, piece_info.pieces.q)
    success_bound = num_neurons <= upper_bound
    if success_precision is False:
        print((
            f"The specified constraints are not satisfied, max_error={max_error:.3e} and min_sdr={min_sdr:.3e}, "
            f"it is likely that the pieces found by get_pieces with sample_size={piece_info.sample_size} and "
            f"std_scale={piece_info.std_scale} do not cover the whole space, try to use a larger init_sample_size "
            f"or sample_size_growth_rate."))
    if success_bound is False:
        print(f"{num_neurons} > {upper_bound}, something is wrong")
    return success_precision and success_bound, piece_info.pieces.q, toc - tic


if __name__ == "__main__":
    result_filename = f"cpwl_relu_net_result_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    num_seeds = 50  # number of random seeds to estimate the average running time
    n_list = [1, 10, 100]  # input dimension of a random CPWL function
    # every element in w_list creates a random CPWL function with a different number of pieces
    w_list = [[0, 1, 3, 7, 15, 31], range(6), range(6)]
    # header of the csv file
    header = ['n', 'q', 'avg_elapsed', f'num_seeds={num_seeds}']
    # creat a csv file to save the result
    with open(result_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    # estimate the running time of Algorithm 1 under different numbers of pieces and input dimensions
    for i, n in enumerate(n_list):
        for w in w_list[i]:
            success_list = []
            elapsed_list = []
            for seed in range(num_seeds + 1):
                success, q, elapsed = find_a_relu_net(n=n, w=w, seed=seed)
                success_list.append(success)
                elapsed_list.append(elapsed)
                if success is False:
                    break
            if False not in success_list:  # only estimate the running time when all trials are successful
                avg_elapsed = sum(elapsed_list[1:]) / (len(elapsed_list) - 1)
                result = [n, q, avg_elapsed]
                print(f"n={n}, q={q}, avg_elapsed={avg_elapsed:.6f} seconds")
                # save the result to a csv file
                with open(result_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(result)
