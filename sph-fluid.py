#!/usr/bin/env python3
"""
SPH 粒子流体模拟 — Python 版
零外部依赖（仅 tkinter，Python 内置）
"""

import tkinter as tk
import math
import random
import time

# ============ 配置 ============
W, H = 900, 640
CX, CY = W // 2, H // 2
BR = 300
GRAVITY = 600
H_RADIUS = 22
DT = 0.003
SUBSTEPS = 3
PARTICLE_R = 4
DAMPING = 0.3
MOUSE_RADIUS = 80
MOUSE_FORCE = 25000
NUM_PARTICLES = 300  # Python 比 JS 慢，减少粒子数保证帧率

# 流体属性
FLUID_VISC = 80
FLUID_REST_DENSITY = 1000
FLUID_STIFFNESS = 4000
FLUID_COLOR = (60, 140, 255)

# SPH 核函数常量
H2 = H_RADIUS * H_RADIUS
POLY6_COEFF = 315.0 / (64.0 * math.pi * H_RADIUS ** 9)
SPIKY_COEFF = -45.0 / (math.pi * H_RADIUS ** 6)
VISC_COEFF = 45.0 / (math.pi * H_RADIUS ** 6)

# ============ 粒子 ============
class Particle:
    __slots__ = ['x', 'y', 'vx', 'vy', 'fx', 'fy', 'density', 'pressure', 'mass']
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.fx = 0.0
        self.fy = 0.0
        self.density = 0.0
        self.pressure = 0.0
        self.mass = 1.0

particles = []

def create_particles():
    for _ in range(NUM_PARTICLES):
        angle = random.random() * math.pi * 2
        r = random.random() * BR * 0.6
        x = CX + math.cos(angle) * r * 0.7
        y = CY + math.sin(angle) * r * 0.7
        particles.append(Particle(x, y))

# ============ 空间哈希 ============
GRID_SIZE = H_RADIUS
grid_cols = math.ceil(W / GRID_SIZE)

def build_grid():
    grid = {}
    for i, p in enumerate(particles):
        key = int(p.x // GRID_SIZE) + int(p.y // GRID_SIZE) * grid_cols
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    return grid

def get_neighbors(grid, idx):
    p = particles[idx]
    gx = int(p.x // GRID_SIZE)
    gy = int(p.y // GRID_SIZE)
    result = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            key = (gx + dx) + (gy + dy) * grid_cols
            cell = grid.get(key)
            if cell:
                for j in cell:
                    if j != idx:
                        result.append(j)
    return result

# ============ SPH 核函数 ============
def poly6(r2):
    if r2 >= H2:
        return 0.0
    diff = H2 - r2
    return POLY6_COEFF * diff * diff * diff

def spiky_grad(r):
    if r >= H_RADIUS or r < 0.001:
        return 0.0
    diff = H_RADIUS - r
    return SPIKY_COEFF * diff * diff

def visc_laplacian(r):
    if r >= H_RADIUS:
        return 0.0
    return VISC_COEFF * (H_RADIUS - r)

# ============ 物理模拟 ============
def compute_density_pressure(grid):
    for i, pi in enumerate(particles):
        pi.density = 0.0
        neighbors = get_neighbors(grid, i)
        for j in neighbors:
            pj = particles[j]
            dx = pj.x - pi.x
            dy = pj.y - pi.y
            r2 = dx * dx + dy * dy
            pi.density += pj.mass * poly6(r2)
        pi.density += pi.mass * poly6(0.0)
        if pi.density < 1.0:
            pi.density = 1.0
        pi.pressure = FLUID_STIFFNESS * (pi.density - FLUID_REST_DENSITY)

def compute_forces(grid):
    for i, pi in enumerate(particles):
        pi.fx = 0.0
        pi.fy = GRAVITY * pi.density
        neighbors = get_neighbors(grid, i)
        for j in neighbors:
            pj = particles[j]
            dx = pj.x - pi.x
            dy = pj.y - pi.y
            r2 = dx * dx + dy * dy
            if r2 >= H2 or r2 < 0.0001:
                continue
            r = math.sqrt(r2)
            # 压力力
            pt = -pi.mass * (pi.pressure + pj.pressure) / (2.0 * pj.density)
            grad = spiky_grad(r)
            rx = dx / r
            ry = dy / r
            pi.fx += pt * grad * rx
            pi.fy += pt * grad * ry
            # 粘性力
            visc = FLUID_VISC * pj.mass * visc_laplacian(r) / pj.density
            pi.fx += visc * (pj.vx - pi.vx)
            pi.fy += visc * (pj.vy - pi.vy)

def integrate(dt):
    for p in particles:
        ax = p.fx / p.density
        ay = p.fy / p.density
        p.vx += ax * dt
        p.vy += ay * dt
        speed = math.sqrt(p.vx * p.vx + p.vy * p.vy)
        if speed > 800:
            p.vx = p.vx / speed * 800
            p.vy = p.vy / speed * 800
        p.x += p.vx * dt
        p.y += p.vy * dt

def enforce_boundary():
    for p in particles:
        dx = p.x - CX
        dy = p.y - CY
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > BR - PARTICLE_R:
            nx = dx / max(dist, 0.001)
            ny = dy / max(dist, 0.001)
            p.x = CX + nx * (BR - PARTICLE_R)
            p.y = CY + ny * (BR - PARTICLE_R)
            vn = p.vx * nx + p.vy * ny
            if vn > 0:
                p.vx -= (1 + DAMPING) * vn * nx
                p.vy -= (1 + DAMPING) * vn * ny

# ============ 鼠标交互 ============
mouse_x, mouse_y = 0, 0
mouse_down = False
prev_mx, prev_my = 0, 0

def apply_mouse_force():
    global mouse_x, mouse_y, prev_mx, prev_my
    if not mouse_down:
        return
    mdx = mouse_x - prev_mx
    mdy = mouse_y - prev_my
    m_speed = math.sqrt(mdx * mdx + mdy * mdy)
    for p in particles:
        dx = p.x - mouse_x
        dy = p.y - mouse_y
        dist2 = dx * dx + dy * dy
        if dist2 > MOUSE_RADIUS * MOUSE_RADIUS or dist2 < 1:
            continue
        dist = math.sqrt(dist2)
        falloff = 1.0 - dist / MOUSE_RADIUS
        inv_density = 1.0 / max(p.density, 1.0)
        # 拖拽力
        if m_speed > 0.5:
            drag = falloff * MOUSE_FORCE * inv_density * DT
            p.vx += mdx * drag
            p.vy += mdy * drag
        # 旋涡力
        nx = dx / dist
        ny = dy / dist
        vortex = falloff * 6000 * inv_density * DT
        p.vx += -ny * vortex
        p.vy += nx * vortex
        # 径向推力
        push = falloff * falloff * 3000 * inv_density * DT
        p.vx += nx * push
        p.vy += ny * push

# ============ 主循环 ============
def step():
    grid = build_grid()
    compute_density_pressure(grid)
    compute_forces(grid)
    apply_mouse_force()
    for _ in range(SUBSTEPS):
        integrate(DT)
        enforce_boundary()

# ============ GUI ============
class SPHApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SPH 粒子流体模拟")
        self.root.configure(bg='#111')
        self.root.resizable(False, False)

        self.bg_color = '#111111'

        # 控制面板
        ctrl = tk.Frame(self.root, bg='#0a0e1a', padx=10, pady=8)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        tk.Label(ctrl, text="⚙ 控制", fg='#88aaff', bg='#0a0e1a',
                 font=('Consolas', 11, 'bold')).pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(ctrl, text="粒子大小:", fg='#999', bg='#0a0e1a',
                 font=('Consolas', 10)).pack(side=tk.LEFT)
        self.size_var = tk.DoubleVar(value=4.0)
        self.size_label = tk.Label(ctrl, text="4.0", fg='#fff', bg='#0a0e1a',
                                   font=('Consolas', 10, 'bold'), width=4)
        self.size_label.pack(side=tk.LEFT)
        size_scale = tk.Scale(ctrl, from_=1, to=10, resolution=0.5, orient=tk.HORIZONTAL,
                              variable=self.size_var, command=self._on_size, length=100,
                              bg='#0a0e1a', fg='#999', highlightthickness=0, troughcolor='#222',
                              showvalue=False)
        size_scale.pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(ctrl, text="粘度:", fg='#999', bg='#0a0e1a',
                 font=('Consolas', 10)).pack(side=tk.LEFT)
        self.visc_var = tk.IntVar(value=80)
        self.visc_label = tk.Label(ctrl, text="80", fg='#fff', bg='#0a0e1a',
                                   font=('Consolas', 10, 'bold'), width=4)
        self.visc_label.pack(side=tk.LEFT)
        visc_scale = tk.Scale(ctrl, from_=5, to=600, resolution=5, orient=tk.HORIZONTAL,
                              variable=self.visc_var, command=self._on_visc, length=150,
                              bg='#0a0e1a', fg='#999', highlightthickness=0, troughcolor='#222',
                              showvalue=False)
        visc_scale.pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(ctrl, text="背景:", fg='#999', bg='#0a0e1a',
                 font=('Consolas', 10)).pack(side=tk.LEFT)
        bg_presets = [('#111111', '深黑'), ('#0a0a1e', '深蓝'), ('#1a0a0a', '深红'),
                      ('#0a1a0a', '深绿'), ('#1a1a1a', '炭灰'), ('#f0ece4', '米白')]
        for color, tip in bg_presets:
            btn = tk.Button(ctrl, bg=color, width=2, height=1, relief=tk.FLAT,
                            command=lambda c=color: self._set_bg(c))
            btn.pack(side=tk.LEFT, padx=1)

        tk.Label(ctrl, text="粒子:", fg='#999', bg='#0a0e1a',
                 font=('Consolas', 10)).pack(side=tk.LEFT)
        self.fps_label = tk.Label(ctrl, text="FPS: --", fg='#666', bg='#0a0e1a',
                                  font=('Consolas', 10))
        self.fps_label.pack(side=tk.RIGHT)

        # 画布
        self.canvas = tk.Canvas(self.root, width=W, height=H, bg=self.bg_color,
                                highlightthickness=0)
        self.canvas.pack()

        # 算法面板
        algo = tk.Frame(self.root, bg='#060812', pady=6, padx=10)
        algo.pack(side=tk.BOTTOM, fill=tk.X)
        self.algo_labels = {}
        steps = [
            ("① 初始化", "init"),
            ("② 近邻搜索", "grid"),
            ("③ 密度·压力", "density"),
            ("④ 压力·粘性力", "force"),
            ("⑤ 加速度·积分", "integrate"),
            ("⑥ 边界约束", "boundary"),
        ]
        for title, key in steps:
            f = tk.Frame(algo, bg='#141e3c', padx=8, pady=4, relief=tk.RIDGE, bd=1)
            f.pack(side=tk.LEFT, padx=4)
            tk.Label(f, text=title, fg='#6699ff', bg='#141e3c',
                     font=('Consolas', 9, 'bold')).pack(anchor=tk.W)
            lbl = tk.Label(f, text="—", fg='#7a8a9a', bg='#141e3c',
                           font=('Consolas', 8), justify=tk.LEFT, anchor=tk.W)
            lbl.pack(anchor=tk.W)
            self.algo_labels[key] = lbl

        # 鼠标事件
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

        # 统计
        self.frame_count = 0
        self.last_time = time.time()
        self.fps_val = 0
        self.active_step = -1

        # 创建粒子
        create_particles()

        # 开始渲染
        self._render()
        self.root.mainloop()

    def _on_size(self, val):
        global PARTICLE_R
        v = float(val)
        PARTICLE_R = v
        self.size_label.config(text=f"{v:.1f}")

    def _on_visc(self, val):
        global FLUID_VISC
        FLUID_VISC = int(val)
        self.visc_label.config(text=str(FLUID_VISC))

    def _set_bg(self, color):
        self.bg_color = color
        self.canvas.configure(bg=color)

    def _on_press(self, e):
        global mouse_down, mouse_x, mouse_y, prev_mx, prev_my
        mouse_down = True
        mouse_x = prev_mx = e.x
        mouse_y = prev_my = e.y

    def _on_drag(self, e):
        global mouse_x, mouse_y, prev_mx, prev_my
        prev_mx, prev_my = mouse_x, mouse_y
        mouse_x, mouse_y = e.x, e.y

    def _on_release(self, e):
        global mouse_down
        mouse_down = False

    def _update_algo(self):
        self.algo_labels["init"].config(
            text=f"N={NUM_PARTICLES} h={H_RADIUS}\ndt={DT} sub={SUBSTEPS}\nρ₀={FLUID_REST_DENSITY} μ={FLUID_VISC}")
        # 简化版统计 — 直接从粒子计算
        min_d = min(p.density for p in particles) if particles else 0
        max_d = max(p.density for p in particles) if particles else 0
        avg_d = sum(p.density for p in particles) / len(particles) if particles else 0
        min_p = min(p.pressure for p in particles) if particles else 0
        max_p = max(p.pressure for p in particles) if particles else 0
        max_v = max(math.sqrt(p.vx**2 + p.vy**2) for p in particles) if particles else 0
        self.algo_labels["density"].config(
            text=f"ρ=[{min_d:.0f},{max_d:.0f}]\nρ̄={avg_d:.0f} P=[{min_p:.0f},{max_p:.0f}]")
        self.algo_labels["integrate"].config(text=f"|v|max={max_v:.0f}")

    def _render(self):
        t0 = time.time()

        # 物理
        step()

        # 清屏
        self.canvas.delete('all')

        # 球体
        self.canvas.create_oval(CX - BR, CY - BR, CX + BR, CY + BR,
                                outline='#4a6080', width=3)

        # 粒子
        c = FLUID_COLOR
        for p in particles:
            speed = math.sqrt(p.vx * p.vx + p.vy * p.vy)
            brightness = min(1.0, 0.6 + speed / 400)
            r = int(c[0] * brightness)
            g = int(c[1] * brightness)
            b = int(c[2] * brightness)
            color = f'#{r:02x}{g:02x}{b:02x}'
            pr = PARTICLE_R
            self.canvas.create_oval(p.x - pr, p.y - pr, p.x + pr, p.y + pr,
                                    fill=color, outline='')

        # 鼠标指示
        if mouse_down:
            self.canvas.create_oval(mouse_x - MOUSE_RADIUS, mouse_y - MOUSE_RADIUS,
                                    mouse_x + MOUSE_RADIUS, mouse_y + MOUSE_RADIUS,
                                    outline='#ffffff22', width=1)

        # FPS
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_time
        if elapsed > 0.5:
            self.fps_val = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = now
            self.fps_label.config(text=f"FPS: {self.fps_val:.0f}")
            self._update_algo()

        # 下一帧（目标 30fps，Python 慢所以放宽）
        frame_ms = (time.time() - t0) * 1000
        delay = max(1, int(33 - frame_ms))
        self.root.after(delay, self._render)

if __name__ == '__main__':
    SPHApp()
