import taichi as ti
import taichi_glsl as ts

ti.init(arch=ti.gpu)

(w , h) = (800,600)
pixels = ti.Vector(3 ,dt = ti.f32 , shape=(w,h))
iResolution = ts.vec(w,h)

scale = ti.var(ti.f32 , shape=())
camdis = ti.var(ti.f32 , shape=())
min_radius = ti.var(ti.f32 , shape=())
fix_radius = ti.var(ti.f32 , shape=())
dr = ti.var(ti.f32 , shape=())
ds = ti.var(ti.f32 , shape=())

scale[None] = -2.0 #6.0
camdis[None] = 8
min_radius[None] = -2
fix_radius[None] = 2

dr[None] = 0.001
ds[None] = 0.01

# camdis = 8

@ti.func
def plane(pos):
    return ts.length(ti.max(ti.abs(pos) - ts.vec(12.0 , 0.5 ,12.0) , 0.0))

@ti.func
def mandelbox(z):
    offset = z 
    dz = 1.0

    for _ in range(0,10):
        # box fold
        z = ts.clamp(z , -1.0 , 1.0) * 2.0 - z
        # ball fold
        r2 = ts.dot(z ,z )
        if r2 < min_radius[None] :
            tmp = (fix_radius[None] / min_radius[None])
            z *= tmp
            dz*= tmp
        elif r2 < fix_radius[None] :
            tmp = fix_radius[None] / r2
            z *= tmp
            dz*= tmp

        z = scale * z + offset
        dz= dz * ti.abs(scale) + 1.0
    
    return ts.length(z) / ti.abs(dz)

@ti.func
def scene(pos):
    #return ti.min(mandelbox(pos) , plane(pos - ts.vec(0.0 , -6.5 , 0.0)))
    return mandelbox(pos)

@ti.func
def raymarcher(ro , rd) :
    maxd = 60.0
    precis = 0.01
    h = precis * 2.0
    t = 0.0
    res = -1.0
    
    for _ in range(0,100):
        if h < precis or t > maxd :
            break
        h = scene(ro + rd * t)
        t += h * 1.0
    
    if t <= maxd :
        res = t
    return res
 
@ti.func
def background(rd):
    v = 1.0 + 1.2 * rd[1]
    return ts.vec(v,v,v)

@ti.func
def ambocc(pos , nor):
    occ = 0.0
    sca = 1.0
    for i in range(0,5):
        hr = 0.01 + 0.12 * float(i)/ 4.0
        aopos = nor * hr + pos 
        dd = scene(aopos)
        occ += -(dd-hr) * sca
        sca *= 0.95
    return ts.clamp(1.0 - 3.0 * occ , 0.0 , 1.0)

@ti.func
def light(lightdir , lightcol , tex , norm ,camdir):
    cosa = ti.pow(0.5 + 0.5 * norm.dot(-lightdir) , 2.0)
    cosr = ti.max((-camdir).dot(ts.reflect(lightdir , norm)) , -0.0)

    diffuse = cosa
    phong = ti.pow(cosr , 8.0)

    return lightcol * (tex * diffuse + phong)

@ti.func
def normal(pos) :
    eps = 0.005
    v1 = ts.vec3( 1.0,-1.0,-1.0)
    v2 = ts.vec3(-1.0,-1.0, 1.0)
    v3 = ts.vec3(-1.0, 1.0,-1.0)
    v4 = ts.vec3( 1.0, 1.0, 1.0)

    return ts.normalize(
        v1 * scene(pos + v1 * eps) + 
        v2 * scene(pos + v2 * eps) + 
        v3 * scene(pos + v3 * eps) + 
        v4 * scene(pos + v4 * eps))

@ti.func
def softray(ro , rd , hn):
    res = 1.0
    t = 0.0005
    h = 1.0
    for _ in range(0,40):
        h = scene(ro + rd * t)
        res = ti.min(res , hn * h / t)
        t += ts.clamp(h, 0.02 ,2.0)
    return ts.clamp(res , 0.0 , 1.0)

@ti.func
def material(pos , camdir):
    norm = normal(pos)

    d1 = - ts.normalize(ts.vec(5.0  , 10.0 , -20.0))
    d2 = - ts.normalize(ts.vec(-5   , 10.0 , 20.0))
    d3 = - ts.normalize(ts.vec(20   , 5.0  , -5.0))
    d4 = - ts.normalize(ts.vec(-20.0, 5.0  , 5.0))

    tex = ts.vec(0.2 , 0.2 , 0.2)
    if pos[1] > -5.95 :
        tex = ts.vec3(0.32,0.28,0.0)

    sha = 0.7 * softray(pos , - d1 , 32.0) + 0.3 * softray(pos , -d4 , 16.0)
    ao = ambocc(pos , norm)

    l1 = light(d1, ts.vec3(1.0,0.9,0.8), tex, norm, camdir)
    l2 = light(d2, ts.vec3(0.8,0.7,0.6), tex, norm, camdir)
    l3 = light(d3, ts.vec3(0.3,0.3,0.4), tex, norm, camdir)
    l4 = light(d4, ts.vec3(0.5,0.5,0.5), tex, norm, camdir)

    return 0.2 * ao + 0.8 * (l1 + l2 + l3 + l4) * sha
    #return 0.5 * ao + 0.5 * (l1 + l2 + l3 + l4) * sha

@ti.func
def render_ray(campos , camdir ) :
    col = ts.vec(0.0 , 0.0 , 0.0)
    dist = raymarcher(campos , camdir)
    if dist == -1.0 :
        col = background(camdir)
    else :
        inters = campos + dist * camdir
        col = material(inters , camdir)
    return col

@ti.func
def cal_look_at_mat(ro , ta , roll) :
    ww = (ta - ro ).normalized()
    uu = ts.cross(ww , ts.vec3(ts.sin(roll) , ts.cos(roll) , 0.0)).normalized()
    vv = ts.cross(uu ,ww).normalized()

    return ts.mat(
        [uu[0] , vv[0] , ww[0]] ,
        [uu[1] , vv[1] , ww[1]] ,
        [uu[2] , vv[2] , ww[2]] )

@ti.func
def main_image(t , i , j ):
    fragcoord = ts.vec(i,j)
    xy = (fragcoord - iResolution / 2.0) / max(iResolution[0] , iResolution[1])
    campos = ts.vec(camdis[None] * ts.cos(t / 5.0) , camdis * 0.5 , camdis[None] * ts.sin(t/5.0))
    camtar = ts.vec(0.0,0.0,0.0)

    camdir = (cal_look_at_mat(campos , camtar , 0.0) @ ts.vec3(xy[0] , xy[1] , 0.9)).normalized()

    return ti.pow(render_ray(campos , camdir) , ts.vec(1.0/2.2 , 1.0/2.2 , 1.0 /2.2))

@ti.kernel
def paint(t : ti.f32 ):
    global dr , ds

    scale[None] += ds[None]
    min_radius[None] += dr[None]
    fix_radius[None] -= dr[None]

    if scale[None] > 0 or scale[None] <-6 :
        ds = -ds
    if min_radius >= fix_radius or min_radius <= -3 or fix_radius >= 3 :
        dr = -dr

    for i , j in pixels :
        pixels[i,j] = main_image(t * 0.03 , i , j)

def main(export):

    resultdir = "./mandelbox_result"
    video_manager = ti.VideoManager(output_dir=resultdir , framerate=24,automatic_build=False)

    gui = ti.GUI("Mandelbox" ,(w,h))
    rng = range (50) if export == True else range(10000000)

    for ts in rng :

        while gui.get_event(ti.GUI.MOTION):
            if gui.event.key == ti.GUI.WHEEL:
                if gui.event.delta[1] > 0 :
                    camdis[None] = max (camdis[None] - 0.5 , 4)
                elif gui.event.delta[1] < 0 :
                    camdis[None] += 0.5

        paint(ts)
        gui.set_image(pixels)

        gui.text(f"t:{ts}" ,(0.02,0.99),color = 0x000000)
        gui.text(f"scale:{scale[None]:.2}" , (0.02 , 0.95) , color=0x000000)
        gui.text(f"min_r:{min_radius[None]:.2}",(0.02,0.90) , color=0x000000)
        gui.text(f"fix_r:{fix_radius[None]:.2}",(0.02,0.85) , color=0x000000)

        gui.show()
        if export == True:
            video_manager.write_frame(pixels.to_numpy())
    
    if export == True:
        print("Exporting gif result...")
        video_manager.make_video(gif=True,mp4=False)
        print("Finish.")

if __name__ == '__main__':
    main(export=False)