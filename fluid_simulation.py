import taichi as ti
import numpy as np
from enum import Enum

ti.init(arch=ti.gpu)

#resolution
resolution = 512

color = (np.random.rand(3) * 0.7) + 0.3

@ti.data_oriented
class Pair :
    def __init__ (self , curr, next):
        self.curr = curr
        self.next = next
    
    def swap(self):
        self.curr , self.next = self.next , self.curr

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

        # self.N_ext = self.N // 2  # number of ext cells set so that that total grid size is still power of 2
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

        ti.root.place(self.alpha, self.beta, self.sum)

    @ti.kernel
    def init(self , r0 : ti.template()):
        for I in ti.grouped(r0):
            self.r[0][I] = r0[I]
            self.z[0][I] = 0.0
            self.Ap[I] = 0.0
            self.p[I] = 0.0
            self.x[I] = 0.0
        
    @ti.func
    def neighbor_sum(self, x , I):
        ret = 0.0
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            ret += x[I + offset] + x[I - offset]
        return ret

    # TODO need seperated
    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            self.Ap[I] = self.neighbor_sum(self.p, I) - (2 * self.dim) * self.p[I]  

    @ti.kernel
    def reduce(self, p : ti.template() , q : ti.template() ):
        self.sum[None] = 0
        for I in ti.grouped(p):
            self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
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

    def solve(self , r0):
        self.init(r0)
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
            self.compute_Ap()
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

@ti.data_oriented
class Smoke_Solver2D:

    def __init__(self , res  ,use_mgpcg = False , use_dye = True):

        self.res = res

        ### physical grid
        # pressure field
        self._pressure_curr = ti.var(dt = ti.f32 , shape=(res , res))
        self._pressure_next = ti.var(dt = ti.f32 , shape=(res , res))
        # velocity field
        self._velocity_curr = ti.Vector(2, dt=ti.f32 , shape=(res , res))
        self._velocity_next = ti.Vector(2, dt=ti.f32 , shape=(res , res))
        # velocity divergence field
        self.velocity_div = ti.var(dt=ti.f32 , shape=(res , res))
        # dyeing field
        self._dye_curr = ti.Vector(3, dt=ti.f32 , shape=(res , res))
        self._dye_next = ti.Vector(3, dt=ti.f32 , shape=(res , res))

        # color buff
        self.color = ti.Vector(3, dt=ti.f32 , shape=(res , res))
        # dye color
        self.dcolor = ti.Vector(list(np.random.rand(3) * 0.7 + 0.3) )
        # smoke source
        self.source_x = res / 2
        self.source_y = 0
        # emit direction
        self.direction = ti.Vector([0.0 , 1.0])

        ### constant physical value
        # time step / delta t
        self.dt = 0.03
        # delta x
        self.dx = 1.0
        # dyeing brightness decay
        self.dye_decay = 0.98
        # force
        self.f_strength = 10000.0

        ### solver param
        # RK-order
        self.RK = 3
        # use_BFECC
        # self.use_BFECC = False
        # linear solver scheme
        self.use_MGPCG = use_mgpcg # false then use jacobi
        # rendering option
        self.use_dye = use_dye # false then render vel_div field

        ### swap pair
        self.velocity = Pair(self._velocity_curr , self._velocity_next)
        self.pressure = Pair(self._pressure_curr , self._pressure_next)
        self.dyeing   = Pair(self._dye_curr , self._dye_next)

        ### MGPCG usage
        if ti.static(self.use_MGPCG) :
            self.pcg = PCG_Solver(n = res // 2 , dim = 2 , max_iter = 20 ,preconditioner = PreConditioner.NonePC)

    ### ================ Advection ===========================

    @ti.func    
    def lerp(self , v1 , v2 , frac):
        return v1 + frac * (v2 - v1)

    @ti.func
    def sample(self , field , u , v):
        i = max(0, min(self.res - 1, int(u)))
        j = max(0, min(self.res - 1, int(v)))
        return field[i, j]

    @ti.func
    def bilinear_interpolate(self , field , u ,v) :
        s, t = u - 0.5, v - 0.5
        iu, iv = int(s), int(t)
        fu, fv = s - iu, t - iv
        a = self.sample(field, iu + 0.5, iv + 0.5)
        b = self.sample(field, iu + 1.5, iv + 0.5)
        c = self.sample(field, iu + 0.5, iv + 1.5)
        d = self.sample(field, iu + 1.5, iv + 1.5)
        return self.lerp(self.lerp(a, b, fu), self.lerp(c, d, fu), fv)

    @ti.func
    def backtrace(self , vf , u,v , dt):
        p = ti.Vector([u,v]) + 0.5
        if ti.static(self.RK == 1) :
            p -= dt * vf[u,v]  #RK1
        elif ti.static(self.RK == 2):
            mid = p - 0.5 * dt * vf[u,v]
            p -= dt * self.sample(vf, mid[0] , mid[1])
        elif ti.static(self.RK == 3) :
            v1 = vf[u,v]
            p1 = p - 0.5 * dt * v1
            v2 = self.sample(vf , p1[0] , p1[1])
            p2 = p - 0.75 * dt * v
            v3 = self.sample(vf , p2[0] , p2[1])
            p -= dt * ( 2/9 * v1 + 1/3 * v2 + 4/9 * v3 )
        else :
            ti.static_print(f"unsupported order for RK{self.RK}")
        return p

    @ti.kernel
    # @ti.func
    def advection(self , vf: ti.template() , field : ti.template() , next_field : ti.template()):
        self.semi_lagrange(vf, field , next_field , self.dt)

    @ti.func
    def semi_lagrange(self , vf , field , next_field , dt):
        for i , j in vf  : 
            p = self.backtrace( vf , i , j , dt)
            next_field[i,j] = self.bilinear_interpolate(field , p[0],  p[1])

    # @ti.func
    # def BFECC(self):
    #    pass

    ### ================ External Force Effect ===========================

    @ti.kernel
    # @ti.func
    def external_force(self , vf : ti.template()):
        f_strenght_dt = ti.static(self.f_strength * self.dt)
        force_r = ti.static(self.res / 3.0)
        inv_force_r = ti.static (1.0 / force_r)
        sx , sy = ti.static( self.source_x , self.source_y )

        # solve smoke source
        for i , j in vf :
            dx , dy = i + 0.5 - sx , j + 0.5 - sy
            d2 = dx * dx + dy * dy
            momentum = self.direction * f_strenght_dt * ti.exp( -d2 * inv_force_r)
            v = vf[i,j]
            vf[i,j] = v + momentum

    ### ================ Projection ===========================

    @ti.kernel
    def divergence_vel(self , field:ti.template()):
        half_inv_dx = ti.static(0.5 / self.dx)
        for i , j in field:
            vl = self.sample(field, i - 1, j)[0]
            vr = self.sample(field, i + 1, j)[0]
            vb = self.sample(field, i, j - 1)[1]
            vt = self.sample(field, i, j + 1)[1]
            vc = self.sample(field, i, j)
            # edge check
            if i == 0:
                vl = -vc[0]
            elif i == self.res - 1:
                vr = -vc[0]
            if j == 0:
                vb = -vc[1]
            elif j == self.res - 1:
                vt = -vc[1]
            # div_v
            div = (vr - vl + vt - vb) * half_inv_dx
            self.velocity_div[i, j] = div

    # @ti.kernel
    # @ti.func
    def projection(self , v_cur : ti.template(), p : ti.template() ):
        self.divergence_vel(v_cur)
        if ti.static(self.use_MGPCG):
            self.MGPCG(p)
        else :
            self.jacobi(p)

    def jacobi(self , p) :
        # jacobi iteration
        for _ in ti.static(range(30)):
            self.jacobi_step(p.curr , p.next)
            p.swap()

    @ti.kernel
    def jacobi_step(self , p_cur:ti.template() , p_nxt:ti.template()):
        dx_sqr = ti.static(self.dx * self.dx)
        for i , j in p_cur :
            pl = self.sample(p_cur , i - 1 , j)
            pr = self.sample(p_cur , i + 1 , j)
            pt = self.sample(p_cur , i , j + 1)
            pb = self.sample(p_cur , i , j - 1)
            p_nxt[i,j] = 0.25 *  (pl + pr + pt + pb - dx_sqr * self.velocity_div[i,j]) 
    
    def MGPCG(self, p):
        self.pcg.solve(self.velocity_div)
        p.curr.copy_from(self.pcg.x)
        return 

    ### ================ Rendering ===========================

    @ti.kernel
    # @ti.func
    def update_v(self , vf : ti.template() , pf : ti.template()):
        half_inv_dx = ti.static( 0.5 / self.dx )
        for i,j in vf :
            pl = self.sample(pf, i - 1, j)
            pr = self.sample(pf, i + 1, j)
            pb = self.sample(pf, i, j - 1)
            pt = self.sample(pf, i, j + 1)
            vf[i, j] = self.sample(vf, i, j) - half_inv_dx * ti.Vector([pr - pl, pt - pb])

    @ti.kernel
    # @ti.func
    def update_dye(self , dyef : ti.template()):
        inv_dye_denom = ti.static(4.0 / (self.res / 15.0)**2)
        sx , sy = ti.static(self.source_x , self.source_y)
        w = ti.Vector([1.0,1.0,1.0])
        for i , j in dyef :
            dx , dy = i + 0.5 - sx , j + 0.5 - sy
            d2 = dx * dx + dy * dy
            dc = dyef[i,j]
            dc += ti.exp(-d2 * inv_dye_denom ) * self.dcolor
            dc *= self.dye_decay
            dyef[i,j] = min (dc , w)
    
    @ti.func
    def render_vel_div(self):
        sf = ti.static(self.velocity_div)
        for i , j in self.color :
            s= abs(sf[i,j])
            self.color[i,j] = ti.Vector([s, s * 0.25 , 0.2])

    @ti.func
    def render_dyeing(self):
        vf = ti.static(self.dyeing.curr)
        for i , j in vf : 
            v = vf[i,j]
            self.color [i,j] = ti.Vector([abs(v[0]) , abs(v[1]) , abs(v[2])])  

    @ti.kernel
    def render(self) :
        if ti.static(self.use_dye) :
            self.render_dyeing()
        else :
            self.render_vel_div()

    # @ti.kernel
    # For each solving step
    def step(self):
        # caculate advection for velocity field and pressure field
        if ti.static(self.use_dye) :
            self.advection(self.velocity.curr , self.velocity.curr , self.velocity.next)
            self.advection(self.velocity.curr , self.dyeing.curr , self.dyeing.next)
        else :
            self.advection(self.velocity.curr , self.velocity.curr , self.velocity.next)
        
        self.velocity.swap()
        self.dyeing.swap()

        # external force
        self.external_force(self.velocity.curr )
        # projection
        self.projection(self.velocity.curr , self.pressure ) 
        # update velocity field
        self.update_v(self.velocity.curr , self.pressure.curr)
        # update dyeing field
        if ti.static(self.use_dye):
            self.update_dye(self.dyeing.curr)

    def reset(self):
        self.velocity.curr.fill([0,0])
        self.pressure.curr.fill(0.0)
        self.dyeing.curr.fill([0,0,0])
        self.color.fill([0,0,0])

def main(gui):
    # smk = Smoke_Solver2D(res = resolution)
    smk = Smoke_Solver2D(res = resolution , use_mgpcg= True)
    smk.reset()
    while gui.running :
        # gui.get_event(ti.GUI.PRESS)
        # solve
        smk.step()
        # rendering
        smk.render()
        # display
        gui.set_image(smk.color)
        gui.show()

if __name__ == "__main__":
    gui = ti.GUI("Smoke Simulation" , res= resolution)
    main(gui)