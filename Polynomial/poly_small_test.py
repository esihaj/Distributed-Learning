import numpy as np
from mpi4py import MPI
import polynomial_code
LARGE_PRIME_NUMBER = 65537

A = np.arange(1, 1+2*3).reshape([2, 3])
B = np.ones([2, 1])

# A = np.arange(1, 1+4*4).reshape([4, 4])
# # B = np.arange(100, 100+2*4).reshape([4, 2])
# B = np.ones([4, 2])
# B[:, 1] = B[:, 1] * 2

if __name__ == '__main__':
    # print ("rank: ", MPI.COMM_WORLD.Get_rank())
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("A", A)
        print ("B", B)
    p_code = polynomial_code.PolynomialCoder(A, B, 3, 1, None, LARGE_PRIME_NUMBER, 4, MPI.COMM_WORLD)
    p_code.polynomial_code()

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("np.matmul")
        print (np.matmul(A.T, B))