import taichi as ti
from enum import Enum
import numpy as np

class PreConditioner(Enum):
    NonePC = 0
    Jacobi = 1
    MultiG = 2

# multigrid preconditioned conjugate gradient method
@ti.data_oriented
class PCG_Solver:

    def __init__(self , n , dim = 2 , max_iter = 400 , real = ti.f32 , preconditioner = PreConditioner.NonePC ):

        self.max_iter = max_iter
        assert isinstance(preconditioner , PreConditioner)
        self.preconditioner = preconditioner

        # grid parameters
        self.N = n

        self.n_mg_levels = 4
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 50
        self.dim = dim

        self.N_ext = self.N // 2  # number of ext cells set so that that total grid size is still power of 2
        self.N_tot = 2 * self.N

        # setup sparse simulation data arrays
        self.r = [ti.var(dt=real) for _ in range(self.n_mg_levels)]  # residual
        self.z = [ti.var(dt=real) for _ in range(self.n_mg_levels)]  # M^-1 self.r
        self.x = ti.var(dt=real)  # solution
        self.p = ti.var(dt=real)  # conjugate gradient
        self.Ap = ti.var(dt=real)  # matrix-vector product
        self.alpha = ti.var(dt=real)  # step size
        self.beta = ti.var(dt=real)  # step size
        self.sum = ti.var(dt=real)  # storage for reductions

        indices = ti.ijk if self.dim == 3 else ti.ij
        self.grid = ti.root.pointer(indices, [self.N_tot // 4]).dense(
            indices, 4).place(self.x, self.p, self.Ap)

        for l in range(self.n_mg_levels):
            self.grid = ti.root.pointer(indices,
                                        [self.N_tot // (4 * 2**l)]).dense(
                                            indices,
                                            4).place(self.r[l], self.z[l])

        self.indices = indices
        ti.root.place(self.alpha, self.beta, self.sum)
        # print(self.z[0].shape)

        # self.shape_range = self.x.shape
        # for i in range(len(self.x.shape)) :
        #     self.shape_range[i] = (0 , self.shape_range[i])

    def set_A(self , A):
        self.A = A

    @ti.kernel
    # def init(self , r0 : ti.template() , x0 : ti.template()):
    #     for I in ti.grouped(r0):
    #         if r0[I] != 0.0 :
    #             self.r[0][I] = r0[I]
    #             self.z[0][I] = 0.0
    #             self.Ap[I] = 0.0
    #             self.p[I] = 0.0
    #             self.x[I] = 0.0
    def init(self , x0 :ti.template()):
        for I in ti.grouped(x0):
            self.x[I] = x0[I]

    # @ti.func
    # def sample(self , m , I):
    #     dim = ti.static(len(m.shape))
    #     index = list((0,)*dim )
    #     for i in ti.static(range(dim)):
    #         index[i] = max(0, min(m.shape[i] - 1, int(I[i])))
    #     return m[index]

    @ti.func
    def neighbor_sum(self, x , I):
        ret = 0.0
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            # ret += self.sample(x , I + offset) + self.sample(x , I - offset)
            ret += x[I+offset] + x[I - offset]
        return ret

    @ti.kernel
    def multiply_by_A(self , v : ti.template() , res : ti.template()):
        # for I in ti.grouped(ti.ndrange(*v.shape)):
        #     res[I] = self.neighbor_sum(v, I) - (2 * self.dim) * v[I]  
        self.A.multiply(v , res)

    # @ti.kernel
    # def compute_Ap(self):
    #     for I in ti.grouped(self.Ap):
    #         self.Ap[I] = self.neighbor_sum(self.p, I) - (2 * self.dim) * self.p[I]

    @ti.kernel
    def sub(self , r : ti.template() , b:ti.template()):
        for I in ti.grouped(b):
            r[I] = b[I] - r[I]

    @ti.kernel
    def reduce(self, p : ti.template() , q : ti.template() ) -> ti.f32:
        self.sum[None] = 0
        for I in ti.grouped(p):
            self.sum[None] += p[I] * q[I]
        return self.sum[None]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.Ap):
            self.r[0][I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.kernel
    def restrict(self, l : ti.template()):
        for I in ti.grouped(self.r[l]):
            res = self.r[l][I] - (2.0 * self.dim * self.z[l][I] -
                                  self.neighbor_sum(self.z[l], I))
            self.r[l + 1][I // 2] += res * 0.5

    @ti.kernel
    def prolongate(self, l : ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] = self.z[l + 1][I // 2]

    @ti.kernel
    def smooth(self, l : ti.template() , phase : ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (I.sum()) & 1 == phase:
                self.z[l][I] = (self.r[l][I] + self.neighbor_sum(
                    self.z[l], I)) / (2.0 * self.dim)

    @ti.kernel
    def jacobi_precondition(self):
        wi = ti.static(1.0 / self.dim)
        for I in ti.grouped(self.z[0]):
            self.z[0][I] = self.r[0][I] * wi 

    def apply_preconditioner(self):
        if self.preconditioner == PreConditioner.NonePC :
            self.z[0].copy_from(self.r[0])
        elif self.preconditioner == PreConditioner.Jacobi :
            self.jacobi_precondition()
        else :
            self.z[0].fill(0)
            for l in range(self.n_mg_levels - 1):
                for _ in range(self.pre_and_post_smoothing << l):
                    self.smooth(l, 0)
                    self.smooth(l, 1)
                self.z[l + 1].fill(0)
                self.r[l + 1].fill(0)
                self.restrict(l)

            for _ in range(self.bottom_smoothing):
                self.smooth(self.n_mg_levels - 1, 0)
                self.smooth(self.n_mg_levels - 1, 1)

            for l in reversed(range(self.n_mg_levels - 1)):
                self.prolongate(l)
                for _ in range(self.pre_and_post_smoothing << l):
                    self.smooth(l, 1)
                    self.smooth(l, 0)

    def solve(self ,  r0 , x0 ):

        self.init(x0)
        self.multiply_by_A(x0 , self.r[0])      # r[0] = A * x0
        self.sub(self.r[0] , r0)                # r[0] = b - r[0]
        
        # self.init(r0 , x0)

        self.reduce(self.r[0], self.r[0])
        initial_rTr = self.sum[None]

        # self.r = b - Ax = b    since self.x = 0
        # self.p = self.r = self.r + 0 self.p
        self.apply_preconditioner()

        self.update_p()

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]

        # CG
        for _ in range(self.max_iter):
            # self.alpha = rTr / pTAp
            # self.compute_Ap()
            self.multiply_by_A(self.p , self.Ap)
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / pAp

            # self.x = self.x + self.alpha self.p
            self.update_x()

            # self.r = self.r - self.alpha self.Ap
            self.update_r()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]
            if rTr < initial_rTr * 1.0e-12:
                break

            # self.z = M^-1 self.r
            self.apply_preconditioner()

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / old_zTr
            # self.p = self.z + self.beta self.p

            self.update_p()
            old_zTr = new_zTr

            # print(f'residual={rTr}')

### ================== test PCG ==============================

@ti.data_oriented
class MatA :
    
    def __init__(self):
        pass

    @ti.func
    def multiply(self, v, res) : 
        d = ti.static(len(v.shape))
        for I in ti.grouped(ti.ndrange(*v.shape)) :
            res[I] = self.neighbor_sum(v, I) - (2 * d ) * v[I]  

    @ti.func
    def neighbor_sum(self, x , I):
        ret = 0.0
        dim = ti.static(len(x.shape))
        for i in ti.static(range(dim)):
            offset = ti.Vector.unit(dim, i)
            # ret += self.sample(x , I + offset) + self.sample(x , I - offset)
            ret+= x[I + offset] + x[I - offset]
        return ret

    @ti.func
    def sample(self , m , I):
        dim = ti.static(len(m.shape))
        index = list((0,)*dim )
        for i in ti.static(range(dim)):
            index[i] = max(0, min(m.shape[i] - 1, int(I[i])))
        return m[index]

@ti.kernel
def init_r0(solver : ti.template() , r0 :ti.template()):
    for I in ti.grouped(
            ti.ndrange(*(
                (solver.N_ext, solver.N_tot - solver.N_ext), ) * solver.dim)):
        r0[I] = 1.0
        for k in ti.static(range(solver.dim)):
            r0[I] *= ti.sin(2.0 * np.pi * (I[k] - solver.N_ext) *
                                    2.0 / solver.N_tot)

def init(solver , r0 , x0):
    r0.fill(0.0)
    x0.fill(0.0)
    init_r0(solver ,r0)

def test_mgpcg():
    import time

    r0 = ti.var(dt = ti.f32 , shape = (256 , 256 , 256 ))
    x0 = ti.var(dt = ti.f32 , shape = (256 , 256 , 256 ))
    solver = PCG_Solver(n = 128 , dim = 3 ,preconditioner=PreConditioner.MultiG)
    init(solver , r0 , x0)
    solver.set_A(MatA())

    t = time.time()
    solver.solve(r0, x0)
    print(f'Solver time: {time.time() - t:.3f} s')

if __name__ == "__main__":
    ti.init(kernel_profiler = True)
    test_mgpcg()
    ti.kernel_profiler_print()
    ti.core.print_stat()