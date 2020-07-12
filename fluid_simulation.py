import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

#resolution
res = 500

color = (np.random.rand(3) * 0.7) + 0.3

class Pair :
    def __init__ (self , curr, next):
        self.curr = curr
        self.next = next
    
    def swap(self):
        self.curr , self.next = self.next , self.curr

# see taichi/example/stable_fluid.py
class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: currrent mouse xy
        # [4:7]: color
        mouse_data = np.array([0] * 8, dtype=np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            x , y = gui.get_cursor_pos()
            mxy = np.array([x,y], dtype=np.float32) * res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dark colors
                self.prev_color = color
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data

@ti.data_oriented
class Fluid_Solver:

    def __init__(self):

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

        ### constant physical value
        # time step / delta t
        self.dt = 0.03
        # delta x
        self.dx = 1.0
        # dyeing brightness decay
        self.dye_decay = 0.99
        # force
        self.f_strength = 10000.0
        # dye color
        self.dcolor = ti.Vector(list(np.random.rand(3) * 0.7 + 0.3) )
        
        ### solver param
        # RK-order
        self.RK = 1
        # use_BFECC
        # self.use_BFECC = False
        # linear solver scheme
        self.use_MGPCG = False # false then use jacobi
        # rendering option
        self.use_dye = False # false then render vel_div field

        ### swap pair
        self.velocity = Pair(self._velocity_curr , self._velocity_next)
        self.pressure = Pair(self._pressure_curr , self._pressure_next)
        self.dyeing   = Pair(self._dye_curr , self._dye_next)

    @ti.func    
    def lerp(self , v1 , v2 , frac):
        return v1 + frac * (v2 - v1)

    @ti.func
    def sample(self , field , u , v):
        i = max(0, min(res - 1, int(u)))
        j = max(0, min(res - 1, int(v)))
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

    @ti.kernel
    def add_impluse(self , vf : ti.template() , dyef : ti.template() , data : ti.ext_arr()):
        f_strenght_dt = ti.static(self.f_strength * self.dt)
        force_r = ti.static(res / 3.0)
        inv_force_r = ti.static (1.0 / force_r)
        inv_dye_denom = ti.static(4.0 / (res / 15.0)**2)

        for i , j in vf :
            dx , dy = i + 0.5 - data[2] , j + 0.5 - data[3] 
            mdir = ti.Vector([data[0] , data[1]])
            # mdir = ti.Vector([1,1]).normalized()
            d2 = dx * dx + dy * dy
            momentum = mdir * f_strenght_dt * ti.exp( -d2 * inv_force_r)
            v = vf[i,j]
            vf[i,j] = v + momentum

            if ti.static(self.use_dye) :
                dc = dyef[i,j]
                if mdir.norm() > 0.5 :
                    dc += ti.exp(-d2 * inv_dye_denom ) * self.dcolor
                dc *= self.dye_decay
                dyef[i,j] = dc

    @ti.func
    def divergence_vel(self , field):
        inv_dx = ti.static(1 / self.dx)
        for i , j in field:
            vl = self.sample(field, i - 1, j)[0]
            vr = self.sample(field, i + 1, j)[0]
            vb = self.sample(field, i, j - 1)[1]
            vt = self.sample(field, i, j + 1)[1]
            vc = self.sample(field, i, j)
            # edge check
            if i == 0:
                vl = -vc[0]
            elif i == res - 1:
                vr = -vc[0]
            if j == 0:
                vb = -vc[1]
            elif j == res - 1:
                vt = -vc[1]
            # div_v
            self.velocity_div[i, j] = (vr - vl + vt - vb) * 0.5 * inv_dx

    @ti.kernel
    def projection(self , v_cur : ti.template(), p : ti.template() ):
        self.divergence_vel(v_cur)
        if ti.static(self.use_MGPCG):
            self.MGPCG()
        else :
            # jacobi iteration
            for _ in ti.static(range(30)):
                self.jacobi(p.curr , p.next)
                p.swap()

    @ti.func
    def jacobi(self , p_cur , p_nxt):
        dx_sqr = self.dx * self.dx
        for i , j in p_cur :
            pl = self.sample(p_cur , i - 1 , j)
            pr = self.sample(p_cur , i + 1 , j)
            pt = self.sample(p_cur , i , j + 1)
            pb = self.sample(p_cur , i , j - 1)
            p_nxt[i,j] = 0.25 *  (pl + pr + pt + pb - dx_sqr * self.velocity_div[i,j]) 
    
    @ti.func    
    def MGPCG(self):
        # TODO
        return 

    @ti.kernel
    def update_v(self , vf : ti.template() , pf : ti.template()):
        half_inv_dx = ti.static( 0.5 / self.dx )
        for i,j in vf :
            pl = self.sample(pf, i - 1, j)
            pr = self.sample(pf, i + 1, j)
            pb = self.sample(pf, i, j - 1)
            pt = self.sample(pf, i, j + 1)
            vf[i, j] = self.sample(vf, i, j)  - half_inv_dx * ti.Vector([pr - pl, pt - pb])

    # see taichi/example/stable_fluid.py
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


    def step(self , mouse_data):
        # caculate advection for velocity field and pressure field
        if self.use_dye :
            self.advection(self.velocity.curr , self.dyeing.curr , self.dyeing.next)
        self.advection(self.velocity.curr , self.velocity.curr , self.velocity.next)
        self.velocity.swap()
        self.dyeing.swap()

        # external force
        self.add_impluse(self.velocity.curr , self.dyeing.curr , mouse_data )
        # projection
        self.projection(self.velocity.curr , self.pressure ) 
        # self.pressure.swap()
        # update velocity field
        self.update_v(self.velocity.curr , self.pressure.curr)

    def reset(self):
        self.velocity.curr.fill([0,0])
        self.pressure.curr.fill(0.0)
        self.dyeing.curr.fill([0,0,0])
        self.color.fill([0,0,0])

def main(gui):
    fls = Fluid_Solver()
    fls.reset()
    gen_mouse_data = MouseDataGen()
    while gui.running :
        gui.get_event(ti.GUI.PRESS)
        # event processing
        data = gen_mouse_data(gui)
        # solve
        fls.step(data)
        # rendering
        fls.render()
        # display
        gui.set_image(fls.color)
        gui.show()
            
if __name__ == "__main__":
    gui = ti.GUI("Eular Fluid Solver" , res= res)
    main(gui)
