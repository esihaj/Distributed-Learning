import numpy as np
from mpi4py import MPI
import polynomial_code
import logging
LARGE_PRIME_NUMBER = 65537

# A = np.arange(2*3).reshape([2, 3])
# B = np.ones([2, 1])
# m = 3
# n = 1
# N = 4

A = np.arange(1, 1+4*4).reshape([4, 4])
B = np.ones([4, 2])
B[:, 1] = B[:, 1] * 2
m = 4
n = 1
N = 5

if __name__ == '__main__':
    # print ("rank: ", MPI.COMM_WORLD.Get_rank())
    logging.basicConfig(format='%(message)s', level=logging.WARN)

    if MPI.COMM_WORLD.Get_rank() == 0:
        logging.info("A:\n" + str(A))
        logging.info("B:\n" + str(B))
    p_code = polynomial_code.PolynomialCoder(A, B, m, n, None, LARGE_PRIME_NUMBER, N, MPI.COMM_WORLD)
    p_code.polynomial_code()

    if MPI.COMM_WORLD.Get_rank() == 0:
        logging.info("np.matmul:\n " + str(np.matmul(A.T, B)))
