import taichi as ti
import numpy as np
from pcg_method import PreConditioner , PCG_Solver
from utils.tools import Pair

#resolution
resolution = 512


@ti.data_oriented
class Smoke_Solver2D:

    @ti.data_oriented
    class MatA :
        
        def __init__(self):
            pass

        @ti.kernel
        def multiply(self, v : ti.template(), res : ti.template()) : 
            d = ti.static(len(v.shape))
            for I in ti.grouped(ti.ndrange(*v.shape)) :
                res[I] = self.neighbor_sum(v, I) - (2 * d ) * v[I]  

        @ti.func
        def neighbor_sum(self, x , I):
            ret = 0.0
            dim = ti.static(len(x.shape))
            for i in ti.static(range(dim)):
                offset = ti.Vector.unit(dim, i)
                ret += self.sample(x , I + offset) + self.sample(x , I - offset)
                # ret+= x[I + offset] + x[I - offset]
            return ret

        @ti.func
        def sample(self , m , I):
            dim = ti.static(len(m.shape))
            index = list((0,)*dim )
            for i in ti.static(range(dim)):
                index[i] = max(0, min(m.shape[i] - 1, int(I[i])))
            return m[index]

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
        self.f_strength = 2000.0

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

        ### Linear solver setting 
        if ti.static(self.use_MGPCG) :
            self.pcg = PCG_Solver(n = res , dim = 2 , max_iter = 20 ,preconditioner = PreConditioner.Nothing)
            self.pcg.set_A(Smoke_Solver2D.MatA())
        
        self.jacobi_max_iter = 30

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
            p2 = p - 0.75 * dt * v1
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
        f_gravity_dt = ti.static(9.8 * self.dt)

        # solve smoke source
        for i , j in vf :
            dx , dy = i + 0.5 - sx , j + 0.5 - sy
            d2 = dx * dx + dy * dy
            momentum = self.direction * f_strenght_dt * ti.exp( -d2 * inv_force_r) - f_gravity_dt
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
        for _ in ti.static(range(self.jacobi_max_iter)):
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
        # set p.curr as x0 , at the beginning of CG iteration
        self.pcg.solve(self.velocity_div , p.curr)
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
        inv_dye_denom = ti.static(4.0 / (self.res / 20.0)**2)
        sx , sy = ti.static(self.source_x , self.source_y)
        for i , j in dyef :
            dx , dy = i + 0.5 - sx , j + 0.5 - sy
            d2 = dx * dx + dy * dy
            dc = dyef[i,j]
            dc += ti.exp(-d2 * inv_dye_denom ) * self.dcolor
            dc *= self.dye_decay
            dyef[i,j] = min (dc , self.dcolor)
    
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

### ====================== GUI Display ===========================

class GIF_Maker:

    def __init__(self):
        self.saving_gif = False
        self.video_mng = ti.VideoManager("./fluid" , framerate=24)
        self.frame_cnt = 0

    def change(self):
        self.saving_gif = not self.saving_gif
        self.frame_cnt = 0
        if self.saving_gif :
            print("starting making gif...")
        else :
            print("cancel .")
            self.video_mng.clean_frames()

    def set_img(self, img):
        if not self.saving_gif :
            return 

        if self.frame_cnt < 72 :
            self.video_mng.write_frame(img.to_numpy())
            self.frame_cnt += 1
        elif self.frame_cnt == 72 :
            print("saving gif...")
            self.video_mng.make_video(mp4 = False)
            self.video_mng.clean_frames()
            self.frame_cnt = 0
            self.saving_gif = False

            import os
            import time
            t = time.strftime("%m-%d_%H:%M:%S", time.localtime())
            os.rename("./fluid/video.gif" , f"./fluid/{t}.gif ")

def Smoke_Solver_With_Jocabi():
    return Smoke_Solver2D(res = resolution )

def Smoke_Solver_With_PCG():
    return Smoke_Solver2D(res = resolution , use_mgpcg= True)

def Smoke_Solver_With_Jocabi_iter(max_iter):
    smk = Smoke_Solver_With_Jocabi()
    smk.jacobi_max_iter = max_iter
    return smk

def Smoke_Solver_With_MGPCG():
    smk = Smoke_Solver_With_PCG()
    smk.pcg.preconditioner = PreConditioner.MultiGrid
    return smk

def main(gui):
    # smk = Smoke_Solver_With_Jocabi()
    smk = Smoke_Solver_With_Jocabi_iter(200)
    # smk = Smoke_Solver_With_PCG()
    # smk = Smoke_Solver_With_MGPCG()
    smk.reset()

    gif = GIF_Maker()
    while gui.running :

        if gui.get_event(ti.GUI.PRESS):
            if gui.is_pressed(ti.GUI.SPACE) : 
                gif.change()

        # solve
        smk.step()
        # rendering
        smk.render()

        # display
        gif.set_img(smk.color)
        gui.set_image(smk.color)
        gui.show()
    
if __name__ == "__main__":
    ti.init(arch=ti.gpu , kernel_profiler = True)
    gui = ti.GUI("Smoke Simulation" , res= resolution)
    main(gui)
    ti.kernel_profiler_print()
    # ti.core.print_profile_info()

