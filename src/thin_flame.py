import taichi as ti
from utils.tools import Pair
from pcg_method import PCG_Solver
import numpy as np

@ti.data_oriented
class Thin_Flame:
    def __init__(self , resolution = 512 ) :
        shape = (resolution , resolution)

        self._sd_cur = ti.var(dt = ti.f32 , shape= shape)
        self._sd_nxt = ti.var(dt = ti.f32 , shape= shape)        
        self._velocity_cur = ti.Vector(2 ,dt = ti.f32 , shape= shape)
        self._velocity_nxt = ti.Vector(2 ,dt = ti.f32 , shape= shape)
        self._pressure_cur = ti.var(dt = ti.f32 , shape= shape)
        self._pressure_nxt = ti.var(dt = ti.f32 , shape= shape)

        self.velocity_div = ti.var(dt = ti.f32 , shape= shape)
        self.pixel = ti.Vector( 3 , dt = ti.f32 , shape= shape)

        self.density_burnt = 1.2
        self.density_fuel = 1.0

        self.sign_dis = Pair(self._sd_cur , self._sd_nxt)
        self.pressure = Pair(self._pressure_cur , self._pressure_nxt)
        self.velocity = Pair(self._velocity_cur , self._velocity_nxt)

        self.resolution = resolution

        self.RK = 3
        self.dx = 1.0
        self.dt = 0.04
        self.speed = 1.0 # 0.5 m/s 
        self.direction = ti.Vector([0.0 , 1.0])
        self.source_pos_x = range(int(resolution /2) - 10 ,int( resolution/2) + 10)
        self.out_momentum = ti.Vector([0.0, 5000.0])
        self.dcolor = ti.Vector(list(np.random.rand(3) * 0.7 + 0.3) )

        self.clamp_sampler = Thin_Flame.Clamping_Sampler(resolution)
        self.extra_sampler = Thin_Flame.Extrapolation_Sampler(resolution)

    @ti.data_oriented
    class Clamping_Sampler:
        def __init__(self , res ):
            self.resolution = res
        @ti.func
        def sample(self , field , u , v):
            i = max(0, min(self.resolution - 1, int(u)))
            j = max(0, min(self.resolution - 1, int(v)))
            return field[i, j]    

    @ti.data_oriented
    class Extrapolation_Sampler:
        def __init__(self , res):
            self.resolution = res
        @ti.func
        def sample(self, field , u , v):
            i = max(0, min(self.resolution - 1, int(u)))
            j = max(0, min(self.resolution - 1, int(v)))
            return field[i,j] - ti.abs(v - j)  if j != int(v) and j < 0 else field[i,j]

    @ti.func
    def density(self , u , v):
        return self.density_burnt if self.sign_dis.curr[u ,v] <= 0 else self.density_fuel

    @ti.func
    def lerp(self , v1 , v2 , frac):
        return v1 + frac * (v2 - v1)

    @ti.func
    def sample(self , field , u , v):
        i = max(0, min(self.resolution - 1, int(u)))
        j = max(0, min(self.resolution - 1, int(v)))
        return field[i, j]

    @ti.func
    def bilinear_interpolate(self , field , u ,v , sampler) :
        s, t = u - 0.5, v - 0.5
        iu, iv = int(s), int(t)
        fu, fv = s - iu, t - iv
        a = sampler.sample(field, iu + 0.5, iv + 0.5)
        b = sampler.sample(field, iu + 1.5, iv + 0.5)
        c = sampler.sample(field, iu + 0.5, iv + 1.5)
        d = sampler.sample(field, iu + 1.5, iv + 1.5)
        return self.lerp(self.lerp(a, b, fu), self.lerp(c, d, fu), fv)

    @ti.func
    def backtrace(self , vf , u,v , dt ):
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

    @ti.func
    def semi_lagrange(self , vf , field , next_field , dt , sampler):
        for i , j in vf  : 
            p = self.backtrace( vf , i , j , dt )
            next_field[i,j] = self.bilinear_interpolate(field , p[0],  p[1] , sampler)

    @ti.kernel
    # @ti.func
    def advection(self , vf: ti.template() , field : ti.template() , sampler: ti.template()):
        self.semi_lagrange(vf, field.curr , field.next, self.dt , sampler)

    @ti.kernel
    def momentum(self, vf : ti.template()):
        # TODO effect velocity by density div on flame front
        # for i , j in  self.sign_dis.curr:
        #     vf[i , j] = vf
        
        # inv_r = ti.static(4.0 / self.resolution)
        res = ti.static(int(self.resolution/2))
        # source
        for i , j in ti.ndrange((res - 10 , res  + 10 ) , (0 , 20)):
            # dir_v = ti.Vector([ res - i, 30]).normalized()
            vf[i,j] += self.dt * self.out_momentum 
        # for j in range(self.resolution - 1):
        #     for i in ti.static(self.source_pos_x) :
        #         vf[i, j] += self.dt * self.out_momentum * ti.exp( - j * inv_r)

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
            elif i == self.resolution - 1:
                vr = -vc[0]
            if j == 0:
                vb = -vc[1]
            elif j == self.resolution - 1:
                vt = -vc[1]
            # div_v
            div = (vr - vl + vt - vb) * half_inv_dx
            self.velocity_div[i, j] = div

    # @ti.kernel
    def projection(self , v_cur : ti.template(), p : ti.template() ):
        self.divergence_vel(v_cur)
        self.jacobi(p)

    # @ti.kernel
    def jacobi(self , p:ti.template()) :
        for _ in ti.static(range(200)):
            self.jacobi_step(p.curr , p.next)
            p.swap()

    @ti.kernel
    def jacobi_step(self , p_cur:ti.template() , p_nxt:ti.template()):
        dx_sqr = ti.static(self.dx * self.dx)
        for i , j in p_cur :
            # pl = p_cur[i - 1 , j]
            # pr = p_cur[i + 1 , j]
            # pt = p_cur[i , j + 1]
            # pb = p_cur[i , j - 1]
            pl = self.sample(p_cur , i - 1 , j)#p_cur[i-1, j]#self.sample(p_cur , i - 1 , j)
            pr = self.sample(p_cur , i + 1 , j)#p_cur[i+1 ,j]#
            pt = self.sample(p_cur , i , j + 1)#p_cur[i, j+1]#
            pb = self.sample(p_cur , i , j - 1)#p_cur[i ,j-1]#
            p_nxt[i,j] = 0.25 *  (pl + pr + pt + pb - dx_sqr * self.velocity_div[i,j]) 

    @ti.kernel 
    def update_v(self , vf : ti.template() , pf:ti.template()):
        half_inv_dx = ti.static( 0.5 / self.dx )
        for i,j in vf :
            pl = self.sample(pf, i - 1, j)
            pr = self.sample(pf, i + 1, j)
            pb = self.sample(pf, i, j - 1)
            pt = self.sample(pf, i, j + 1)
            vf[i, j] = self.sample(vf, i, j) - half_inv_dx * ti.Vector([pr - pl, pt - pb])

    @ti.kernel
    def update_distance(self, sdf : ti.template() , vf : ti.template()):
        # inv_r = ti.static(4.0 / (self.resolution / 20.0)**2)
        res = ti.static(int(self.resolution/2))
        
        for i ,j in ti.ndrange((res - 10 , res + 10) , (0 , 20)) : 
            # dx , dy = self.resolution / 2 - i , j
            # d2 = dx * dx + dy * dy
            sdf[i , j] = -1.0 #ti.exp(- d2 * inv_r) * -10.0

        for i, j in sdf:
            # dx , dy = self.resolution / 2 - i , j
            # d2 = dx * dx + dy * dy
            # sdf[i , j] -= ti.exp(- d2 * inv_r) * 10.0
            # sdf[i , j] += self
            sdf[i , j] += self.dt * self.speed #(self.speed - vf[i,j].norm())


    @ti.kernel
    def init_level_set(self):
        sdf = ti.static(self.sign_dis.curr)
        inv_r = ti.static(4.0 / (self.resolution / 20.0)**2)
        for i, j in sdf:
            dx , dy = self.resolution / 2 - i , j
            d2 = dx * dx + dy * dy
            sdf[i , 0] = ti.exp(- d2 * inv_r) * 10.0

    # @ti.kernel
    def init(self):
        self.velocity.curr.fill([0.0,0.0])
        self.pressure.curr.fill(0.0)
        self.sign_dis.curr.fill(10.0)
        self.pixel.fill([0.0 , 0.0 , 0.0 ])
        self.init_level_set()
            
    @ti.kernel
    def render(self , sdf: ti.template()):
        zero = ti.Vector([0.0 , 0.0 , 0.0])
        for indices in ti.grouped(sdf):
            self.pixel[indices] = self.dcolor * ti.exp(1.0/ (sdf[indices] - 0.01)) if sdf[indices] < 0.0 else zero


    def step(self):
        # advection 
        self.advection(self.velocity.curr , self.velocity , self.clamp_sampler)
        self.advection(self.velocity.curr , self.sign_dis , self.clamp_sampler)
        self.velocity.swap()
        self.sign_dis.swap()

        # externel force 
        self.momentum(self.velocity.curr)
        # projection
        self.projection(self.velocity.curr , self.pressure)
        # update
        self.update_v(self.velocity.curr , self.pressure.curr)
        self.update_distance(self.sign_dis.curr , self.velocity.curr)

        self.render(self.sign_dis.curr)

def main():
    resolution = 512
    ti.init(arch=ti.gpu , kernel_profiler = True)
    gui = ti.GUI("Thin Flame" , res= resolution)

    solver = Thin_Flame(resolution)
    solver.init()
    while gui.running:
        solver.step()
        gui.set_image(solver.pixel)
        gui.show()

if __name__ == '__main__':
    main()