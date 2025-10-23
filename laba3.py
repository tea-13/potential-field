from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ---------- Константы ----------
EPS = 1e-9
BIG_PENALTY = 1e4  # штраф внутри препятствий (большой, но удобен для визуализации)

# ---------- Геометрия / препятствия ----------
@dataclass
class CircularObstacle:
    center: Tuple[float, float]
    radius: float
    eta: float = 2.0      # сила отталкивания
    rho0: float = 2.0     # радиус влияния за границей

    def c_np(self) -> np.ndarray:
        return np.array(self.center, dtype=float)

    def phi(self, x: np.ndarray) -> float:
        """SDF (подписанное расстояние): >0 снаружи, 0 на границе, <0 внутри."""
        return np.linalg.norm(x - self.c_np()) - self.radius

# ---------- Потенциал / градиенты ----------
@dataclass
class PotentialParams:
    k_att: float = 1.0

class PotentialField:
    def __init__(self, goal: Tuple[float, float], obstacles: List[CircularObstacle], params: PotentialParams):
        self.g = np.array(goal, dtype=float)
        self.obstacles = obstacles
        self.p = params

    # притяжение
    def U_att(self, x: np.ndarray) -> float:
        return 0.5 * self.p.k_att * float(np.dot(x - self.g, x - self.g))

    def grad_U_att(self, x: np.ndarray) -> np.ndarray:
        return self.p.k_att * (x - self.g)

    # отталкивание для одного круга
    @staticmethod
    def U_rep_i(x: np.ndarray, obs: CircularObstacle) -> float:
        dvec = x - obs.c_np()
        d = np.linalg.norm(dvec)
        phi = d - obs.radius
        if phi <= 0.0:      # внутри
            return BIG_PENALTY
        if phi > obs.rho0:  # вне зоны влияния
            return 0.0
        return 0.5 * obs.eta * (1.0 / phi - 1.0 / obs.rho0) ** 2

    @staticmethod
    def grad_U_rep_i(x: np.ndarray, obs: CircularObstacle) -> np.ndarray:
        dvec = x - obs.c_np()
        d = np.linalg.norm(dvec)
        if d < EPS:
            dvec = np.array([1.0, 0.0]); d = 1.0
        n = dvec / d
        phi = d - obs.radius
        if phi <= 0.0:
            return BIG_PENALTY * n
        if phi > obs.rho0:
            return np.zeros(2)
        return -obs.eta * (1.0 / phi - 1.0 / obs.rho0) * (1.0 / (phi ** 2)) * n

    # суммарно
    def U(self, x: np.ndarray) -> float:
        val = self.U_att(x)
        for obs in self.obstacles:
            val += self.U_rep_i(x, obs)
        return val

    def grad_U(self, x: np.ndarray) -> np.ndarray:
        g = self.grad_U_att(x)
        for obs in self.obstacles:
            g += self.grad_U_rep_i(x, obs)
        return g

# ---------- Планировщик пути ----------
@dataclass
class PlannerConfig:
    step_size: float = 0.2
    max_iters: int = 4000
    tol_goal: float = 0.05
    use_ema: bool = True
    beta: float = 0.85        # сглаживание направления
    clip_step: Optional[float] = 0.35  # ограничение длины шага (None = без клипа)

class PotentialFieldPlanner:
    def __init__(self, field: PotentialField, cfg: PlannerConfig):
        self.field = field
        self.cfg = cfg

    @staticmethod
    def project_outside(x: np.ndarray, obstacles: List[CircularObstacle], eps: float = 1e-4) -> np.ndarray:
        x_new = x.copy()
        for obs in obstacles:
            dvec = x_new - obs.c_np()
            d = np.linalg.norm(dvec)
            if d <= obs.radius + eps:
                if d < EPS:
                    dvec = np.array([1.0, 0.0]); d = 1.0
                x_new = obs.c_np() + (dvec / d) * (obs.radius + eps)
        return x_new

    def plan(self, start: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        x = np.array(start, dtype=float)
        path = [x.copy()]
        energies = [self.field.U(x)]
        v = np.zeros(2)

        for _ in range(self.cfg.max_iters):
            x = self.project_outside(x, self.field.obstacles, eps=1e-3)

            grad = self.field.grad_U(x)
            direction = -grad

            if self.cfg.use_ema:
                v = self.cfg.beta * v + (1.0 - self.cfg.beta) * direction
                move = v.copy()
            else:
                move = direction

            # нормируем и масштабируем шаг
            move = self.cfg.step_size * move / (np.linalg.norm(move) + EPS)

            # клип по длине шага
            if self.cfg.clip_step is not None:
                m = np.linalg.norm(move)
                if m > self.cfg.clip_step:
                    move *= (self.cfg.clip_step / (m + EPS))

            x_next = x + move
            x_next = self.project_outside(x_next, self.field.obstacles, eps=1e-4)

            path.append(x_next.copy())
            energies.append(self.field.U(x_next))

            if np.linalg.norm(x_next - self.field.g) < self.cfg.tol_goal:
                x = x_next
                break
            x = x_next

        return np.array(path), np.array(energies)

# ---------- Визуализация ----------
@dataclass
class GridVizConfig:
    res: int = 180                       # разрешение сетки
    colormap: str = "turbo"              # палитра ("turbo", "viridis", "magma" и т.п.)
    robust_clip: Optional[Tuple[float, float]] = (3.0, 96.0)  # процентили (None = без клипа)
    arrows_per_axis: int = 26            # плотность стрелок поля
    draw_influence: bool = True          # рисовать радиусы влияния препятствий
    linewidth_path: float = 2.6
    linewidth_obstacle: float = 2.0
    quiver_normalize: bool = True        # нормировать ли длину стрелок
    quiver_color_by_magnitude: bool = True  # окраска по |f|, чтобы видеть силу

class Visualizer:
    def __init__(self, field: PotentialField, viz: GridVizConfig):
        self.field = field
        self.viz = viz

    @staticmethod
    def _auto_limits(start: Tuple[float, float], goal: Tuple[float, float], obstacles: List[CircularObstacle], margin: float = 2.0):
        pts = [np.array(start), np.array(goal)]
        for o in obstacles:
            c = o.c_np()
            pts += [c + [ o.radius, 0], c + [-o.radius, 0], c + [0,  o.radius], c + [0, -o.radius]]
        P = np.vstack(pts)
        xmin, ymin = P.min(axis=0) - margin
        xmax, ymax = P.max(axis=0) + margin
        return (xmin, xmax), (ymin, ymax)

    def _grid(self, xlim, ylim):
        xs = np.linspace(xlim[0], xlim[1], self.viz.res)
        ys = np.linspace(ylim[0], ylim[1], self.viz.res)
        X, Y = np.meshgrid(xs, ys)

        U_total = np.zeros_like(X)
        U_att   = np.zeros_like(X)
        U_rep   = np.zeros_like(X)
        Fx = np.zeros_like(X); Fy = np.zeros_like(X)

        for i in range(self.viz.res):
            for j in range(self.viz.res):
                p = np.array([X[i, j], Y[i, j]])
                ua = self.field.U_att(p)
                ur = 0.0
                for obs in self.field.obstacles:
                    ur += self.field.U_rep_i(p, obs)
                U_att[i, j] = ua
                U_rep[i, j] = ur
                U_total[i, j] = ua + ur

                g = self.field.grad_U(p)
                f = -g
                Fx[i, j], Fy[i, j] = f[0], f[1]

        # маска кругов, чтобы не «забивать» тепловую карту штрафом
        mask = np.zeros_like(U_total, dtype=bool)
        for obs in self.field.obstacles:
            C = obs.c_np(); R = obs.radius
            mask |= ((X - C[0])**2 + (Y - C[1])**2) <= (R**2 + 1e-8)

        U_att_m   = np.ma.array(U_att,   mask=mask)
        U_rep_m   = np.ma.array(U_rep,   mask=mask)
        U_total_m = np.ma.array(U_total, mask=mask)

        def clip_range(Um):
            if self.viz.robust_clip is None:
                return float(np.min(Um)), float(np.max(Um))
            lo, hi = self.viz.robust_clip
            data = Um.compressed()
            vmin = float(np.nanpercentile(data, lo))
            vmax = float(np.nanpercentile(data, hi))
            return vmin, max(vmax, vmin + 1e-6)

        ranges = {
            "att":   clip_range(U_att_m),
            "rep":   clip_range(U_rep_m),
            "total": clip_range(U_total_m),
        }

        Fmag = np.sqrt(Fx**2 + Fy**2) + 1e-9
        if self.viz.quiver_normalize:
            Fx_show, Fy_show = Fx / Fmag, Fy / Fmag
        else:
            Fx_show, Fy_show = Fx, Fy

        # диапазон для окраски по |f|
        if self.viz.robust_clip is None:
            fmin, fmax = float(np.min(Fmag)), float(np.max(Fmag))
        else:
            lo, hi = self.viz.robust_clip
            fmin = float(np.nanpercentile(Fmag, lo))
            fmax = float(np.nanpercentile(Fmag, hi))
        fmax = max(fmax, fmin + 1e-9)
        f_norm = Normalize(vmin=fmin, vmax=fmax)

        return X, Y, U_att_m, U_rep_m, U_total_m, ranges, Fx_show, Fy_show, Fmag, f_norm

    def plot_two_maps(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        path: np.ndarray,
        mode: Literal["heat_total", "heat_att", "heat_rep", "quiver", "stream"] = "heat_total"
    ):
        xlim, ylim = self._auto_limits(start, goal, self.field.obstacles, margin=2.0)
        (X, Y, U_att_m, U_rep_m, U_total_m, ranges,
         Fx_show, Fy_show, Fmag, f_norm) = self._grid(xlim, ylim)

        fig = plt.figure(figsize=(13, 5))

        # ---- Левая панель: путь + препятствия ----
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(path[:, 0], path[:, 1], linewidth=self.viz.linewidth_path, label="путь", color="#1f77b4")
        ax1.scatter([start[0]], [start[1]], s=60, marker="o", label="старт", color="#2ca02c", zorder=3)
        ax1.scatter([goal[0]], [goal[1]], s=140, marker="*", label="цель",  color="#d62728", zorder=3)
        for obs in self.field.obstacles:
            ax1.add_patch(plt.Circle(obs.c_np(), obs.radius, fill=False, linewidth=self.viz.linewidth_obstacle, color="#111"))
            if self.viz.draw_influence:
                ax1.add_patch(plt.Circle(obs.c_np(), obs.radius + obs.rho0, fill=False, linestyle="--", linewidth=1.0, color="#555"))
        ax1.set_xlim(*xlim); ax1.set_ylim(*ylim)
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_title("Путь и препятствия")
        ax1.set_xlabel("x"); ax1.set_ylabel("y")
        ax1.legend(loc="best")

        # ---- Правая панель: тепловая карта / стрелки / поток ----
        ax2 = fig.add_subplot(1, 2, 2)
        if mode == "heat_total":
            U, (vmin, vmax), title = U_total_m, ranges["total"], "Тепловая карта U_total"
        elif mode == "heat_att":
            U, (vmin, vmax), title = U_att_m,   ranges["att"],   "Тепловая карта U_att"
        elif mode == "heat_rep":
            U, (vmin, vmax), title = U_rep_m,   ranges["rep"],   "Тепловая карта U_rep"
        elif mode in ("quiver", "stream"):
            U, (vmin, vmax), title = None, (None, None), None
        else:
            raise ValueError("mode: heat_total | heat_att | heat_rep | quiver | stream")

        if U is not None:
            hm = ax2.imshow(
                U, origin="lower",
                extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                cmap=self.viz.colormap, vmin=vmin, vmax=vmax,
                interpolation="bilinear", aspect="equal"
            )
            cbar = plt.colorbar(hm, ax=ax2, shrink=0.9); cbar.set_label("U")
            ax2.set_title(title)
        elif mode == "quiver":
            skip = max(1, int(self.viz.res // self.viz.arrows_per_axis))
            if self.viz.quiver_color_by_magnitude:
                C = f_norm(Fmag[::skip, ::skip])
            else:
                C = None
            q = ax2.quiver(
                X[::skip, ::skip], Y[::skip, ::skip],
                Fx_show[::skip, ::skip], Fy_show[::skip, ::skip],
                C, angles="xy", scale_units="xy",
                scale=0.8 if self.viz.quiver_normalize else None,
                width=0.003, cmap=self.viz.colormap
            )
            if C is not None:
                # создаём общую шкалу для цвета по |f|
                sm = plt.cm.ScalarMappable(cmap=self.viz.colormap, norm=f_norm)
                sm.set_array([])
                plt.colorbar(sm, ax=ax2, shrink=0.9, label="|f|")
            ax2.set_title("−∇U: стрелки (цвет = |f|)" if self.viz.quiver_color_by_magnitude else "−∇U: стрелки")
        elif mode == "stream":
            strm = ax2.streamplot(
                X, Y, Fx_show, Fy_show,
                density=1.2, linewidth=1.0, arrowsize=1.2,
                color=f_norm(Fmag), cmap=self.viz.colormap
            )
            sm = plt.cm.ScalarMappable(cmap=self.viz.colormap, norm=f_norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax2, shrink=0.9, label="|f|")
            ax2.set_title("−∇U: потоковые линии (цвет = |f|)")

        for obs in self.field.obstacles:
            ax2.add_patch(plt.Circle(obs.c_np(), obs.radius, fill=False, linewidth=1.2, color="#111"))
        ax2.scatter([goal[0]], [goal[1]], s=70, marker="*", color="#d62728", zorder=3)
        ax2.set_xlim(*xlim); ax2.set_ylim(*ylim)
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_xlabel("x"); ax2.set_ylabel("y")

        plt.tight_layout()
        plt.show()

# ---------- Пример использования (РАСКОММЕНТИРУЙТЕ у себя, чтобы запустить) ----------
if __name__ == "__main__":
    # Сцена
    start = (0.0, 0.0)
    goal  = (20.0, 20.0)
    obstacles = [
        CircularObstacle(center=(13.0, 13.0), radius=1.0, eta=20.5, rho0=5.0),
        CircularObstacle(center=(4.0, 20.0), radius=1.2, eta=13.5, rho0=10.5),
        CircularObstacle(center=(8.5, 2.5), radius=1.0, eta=5.5, rho0=4.6),
    ]

    # Потенциал и планировщик
    pot = PotentialField(goal, obstacles, PotentialParams(k_att=0.8))
    planner = PotentialFieldPlanner(
        pot,
        PlannerConfig(step_size=0.22, max_iters=3000, tol_goal=0.05,
                      use_ema=True, beta=0.87, clip_step=0.35)
    )
    path, energies = planner.plan(start)

    # Визуализация: две карты
    viz_cfg = GridVizConfig(
        res=200, colormap="turbo", robust_clip=(3, 96),
        arrows_per_axis=28, draw_influence=True,
        quiver_normalize=False,            # показать реальную длину стрелок
        quiver_color_by_magnitude=True
    )
    vis = Visualizer(pot, viz_cfg)
    vis.plot_two_maps(start, goal, path, mode="heat_total")  # Тепловая карта суммарного потенциала
    #vis.plot_two_maps(start, goal, path, mode="heat_rep")   # Только отталкивание (для проверки eta/rho0)
    #vis.plot_two_maps(start, goal, path, mode="quiver")     # Векторное поле (цвет = |f|)
    #vis.plot_two_maps(start, goal, path, mode="stream")     # Потоковые линии (цвет = |f|)
