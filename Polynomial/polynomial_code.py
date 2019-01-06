from mpi4py import MPI
import numpy as np
import random
import threading
import time
from scipy.interpolate import lagrange
import logging

def loop():
    t = time.time()
    while time.time() < t + 60:
        a = 1 + 1


class PolynomialCoder:

    def __init__(self, A, B, m, n, buffer, F, N, comm):
        # Buffer to put the answer
        self.buffer = buffer
        # Change to True for more accurate timing, sacrificing performance
        self.barrier = False
        # Change to True to imitate straggler effects
        self.straggling = False
        self.comm = comm
        self.N = N
        self.A = A
        self.B = B
        self.s = A.shape[0]
        self.r = A.shape[1]
        self.t = B.shape[1]
        self.m = m
        self.n = n
        self.var = [i+1 for i in range(N+1)] + [3]
        logging.debug("var:\n" + str(self.var))
        # self.zero_padding_matrices()
        self.F = F

    def zero_padding_matrices(self):
        s = self.s
        r = self.r
        m = self.m
        n = self.n
        A = self.A
        B = self.B
        t = self.t
        new_r = r + m - r % m
        A_pad = np.zeros((s, new_r))
        A_pad[:, :s] = A
        new_t = t + n - t % n
        B_pad = np.zeros((s, new_t))
        B_pad[:, :s] = B
        self.A = A_pad
        self.B = B_pad
        self.r = new_r
        self.t = new_t

    def data_send(self):

        comm = self.comm
        A = self.A
        B = self.B
        var = self.var
        F = self.F
        n = self.n
        m = self.m
        N = self.N

        # Decide and broadcast chosen straggler
        straggler = random.randint(1, N)
        for i in range(N):
            comm.send(straggler, dest=i + 1, tag=7)

        # Split matrices
        Ap = np.hsplit(A, m)
        Bp = np.hsplit(B, n)

        # Encode the matrices
        Aenc = [sum([Ap[j] * (pow(var[i], j, F)) for j in range(m)]) % F for i in range(N)]
        Benc = [sum([Bp[j] * (pow(var[i], j * m, F)) for j in range(n)]) % F for i in range(N)]
        Benc = np.asarray(Benc, dtype=np.int)
        # Aenc = [sum([Ap[j] * (pow(var[i], j)) for j in range(m)]) for i in range(N)]
        # Benc = [sum([Bp[j] * (pow(var[i], j * m)) for j in range(n)])for i in range(N)]

        logging.debug("Aenc:\n" + str(Aenc))
        logging.debug("Benc:\n" + str(Benc))

        # Start requests to send
        request_A = [None] * N
        request_B = [None] * N
        self.bp_start = time.time()
        for i in range(N):
            request_A[i] = comm.Isend([Aenc[i], MPI.INT], dest=i + 1, tag=15)
            request_B[i] = comm.Isend([Benc[i], MPI.INT], dest=i + 1, tag=29)
        MPI.Request.Waitall(request_A)
        MPI.Request.Waitall(request_B)

        # Optionally wait for all workers to receive their submatrices, for more accurate timing
        if self.barrier:
            comm.Barrier()

        self.bp_sent = time.time()
        logging.info("Time spent sending all messages is: %f" % (self.bp_sent - self.bp_start))

    def reducer(self):

        comm = self.comm
        var = self.var
        F = self.F
        r = self.r
        n = self.n
        m = self.m
        t = self.t
        N = self.N

        # Initialize return dictionary
        return_dict = []
        for i in range(N):
            return_dict.append(np.zeros((int(r / m), int(t / n)), dtype=np.int_))

        # Start requests to receive
        request_C = [None] * N
        for i in range(N):
            request_C[i] = comm.Irecv([return_dict[i], MPI.INT], source=i + 1, tag=42)

        return_C = [None] * N
        list = []
        # Wait for the mn fastest workers
        for i in range(m * n):
            j = MPI.Request.Waitany(request_C)
            list.append(j)
            return_C[j] = return_dict[j]

        self.bp_received = time.time()
        logging.info("Time spent waiting for %d workers %s is: %f" % (
            m * n, ",".join(map(str, [x + 1 for x in list])), (self.bp_received - self.bp_sent)))

        logging.debug("return C: " + str(return_C))
        base_indices = []
        for j in range(n):
            for i in range(m):
                base_indices.append([i*int(r/m), j*int(t/n)])
        base_indices = tuple(reversed(base_indices))
        logging.debug("base_indices:\n" + str(base_indices))
        # Lagrange polynomial interpolation
        # coeffs = np.zeros((int(r / m), int(t / n), m * n))
        # list is 0 based but our workers, and Aenc, Benc matrices are 1 based
        # so we need to convert the list
        recv_var = tuple(map(lambda x: x+1, list))
        coeffs = np.zeros((r, t))
        for i in range(int(r / m)):
            for j in range(int(t / n)):
                f_z = []
                for k in range(m * n):
                    f_z.append(return_C[list[k]][i][j])
                lagrange_interpolate = lagrange(recv_var, f_z)
                logging.debug("list: %s,\nf_z: %s,\nlag: %s" %
                              tuple(map(lambda x: str(x), [list, f_z, lagrange_interpolate])))
                for index, lag_coef in enumerate(lagrange_interpolate):
                    coeffs[i+base_indices[index][0]][j+base_indices[index][1]] = lag_coef
        logging.debug("coeffs: " + str(coeffs.shape))

        logging.info("coeffs:\n" + str(coeffs))

        # TODO: Create C by combibing C_tildes

        self.bp_done = time.time()
        logging.info("Time spent decoding is: %f" % (self.bp_done - self.bp_received))

    def mapper(self):

        comm = self.comm
        F = self.F
        s = self.s
        n = self.n
        m = self.m
        r = self.r
        t = self.t
        # Receive straggler information from the master
        straggler = comm.recv(source=0, tag=7)


        # Receive split input matrices from the master
        Ai = np.empty_like(np.matrix([[0] * int(r / m) for i in range(s)]))
        Bi = np.empty_like(np.matrix([[0] * int(t / n) for i in range(s)]))
        receive_A = comm.Irecv(Ai, source=0, tag=15)
        receive_B = comm.Irecv(Bi, source=0, tag=29)

        receive_A.wait()
        receive_B.wait()

        if self.barrier:
            comm.Barrier()

        self.wbp_received = time.time()

        # Start a separate thread to mimic background computation tasks if this is a straggler
        if self.straggling:
            if straggler == comm.rank:
                t = threading.Thread(target=loop)
                t.start()

        Ci = (Ai.getT() * Bi) % F
        logging.debug("r[" + str(comm.Get_rank()) + "] A:\n" + str(Ai)
                      + "\nB:\n" + str(Bi)
                      + "\nC:\n" + str(Ci))

        # print("Ci["+ str(comm.Get_rank()) +"]", Ci )
        self.wbp_done = time.time()
        logging.info("Worker %d computing takes: %f\n" % (comm.Get_rank(), self.wbp_done - self.wbp_received))

        sC = comm.Isend(Ci, dest=0, tag=42)
        sC.Wait()

    def polynomial_code(self):
        comm = self.comm
        if comm.Get_rank()  == 0:
            self.data_send()
            self.reducer()
        else:
            self.mapper()
