# ============================================================
#  ЗАДАЧА 2: Потенциальные поля на решётке + АНИМАЦИЯ
#  ------------------------------------------------------------
#  Что здесь (ООП, полностью самодостаточно):
#   • Дискретное поле U = U_att + Σ U_rep на решётке width×height.
#   • Жёсткая маска препятствий (occ_mask) + карта клиренса (SDF/clearance).
#   • Планировщик-«шагатель» без прохода сквозь препятствия и без «среза угла».
#   • Динамические препятствия, преследующие робота (настраиваемая частота/скорость).
#   • Две визуализации: тепловая карта U и поле −∇U.
#   • АНИМАЦИЯ движения робота и препятствий (matplotlib.animation.FuncAnimation).
#
#  ВАЖНО:
#   • Код НИЧЕГО не сохраняет и НЕ запускается автоматически.
#   • Пример запуска внизу — закомментирован.
#
#  Параметры (где «крутить ручки»):
#   GridPotentialParams:
#     k_att          — сила «тяги к цели». Чем меньше, тем заметнее влияние препятствий.
#     repulsive_layers = [RepulsiveConfig(eta, rho0, cell_radius)]
#                      — отталкивание слоями: ближний сильнее (большая eta, малая rho0),
#                        дальний мягче (меньше eta, больше rho0). Обычно 1–2 слоя.
#     big_penalty    — штраф внутри препятствий/на границе (большой, но конечный).
#   RepulsiveConfig:
#     eta            — «высота холма» вокруг препятствия (сила отталкивания).
#     rho0           — «радиус влияния» в КЛЕТКАХ (за пределами U_rep=0).
#     cell_radius    — «радиус узла» для дискретного SDF (обычно 0.5 клетки).
#   GridPlannerConfig:
#     max_steps      — лимит шагающего алгоритма.
#     smooth, beta   — EMA-сглаживание направления (уменьшает «дёрганья»).
#     prefer_grad    — приоритизация шага по −∇U (а не только минимум соседей).
#     tie_bias       — насколько поощрять согласование шага с −∇U.
#     allow_diag     — разрешать диагональные ходы.
#     safety_clearance — «подушка безопасности» (в клетках): 0..1+. Если 0.5 — держим полклетки до стены.
#     reeval_every   — как часто пересчитывать U при движении динамики (в шагах).
#   DynamicObstacle:
#     pos            — текущая клетка динамического препятствия (рисуется квадратом).
#     every          — двигать каждые N шагов робота (частота).
#     speed_cells    — на сколько клеток сдвигать за обновление (скорость).
#     mode="pursue"  — простое преследование (в сторону робота по знаку dx,dy).
#   GridVizConfig:
#     colormap       — палитра для теплокарты/стрелок ("turbo","viridis","magma",...).
#     robust_clip    — процентили для обрезки экстремумов (None — без клипа).
#     show_occ_mask  — сероватая подложка занятости (понятные «стены»).
#     quiver_*       — как рисовать поле стрелок (нормировать длины, красить по |−∇U|).
#   AnimationConfig:
#     mode           — "heat" или "quiver" (что рисуем справа).
#     interval_ms    — скорость анимации (мс между кадрами).
#     max_frames     — ограничение на кадры (защита от вечной анимации).
#     trail          — длина «хвоста» пути в кадрах (None — весь путь).
#
#  Советы:
#    • Если кажется, что «параметры не влияют», проверьте, что препятствия попадают
#      в зону rho0; уменьшите k_att или увеличьте eta/rho0; включите mode="heat"
#      и посмотрите только U_rep (см. отметки препятствий).
#    • Если «робот режет углы» — это отключено corner-cutting правилом, но если
#      всё же видите визуально, увеличьте safety_clearance, чтобы он держал дистанцию.
# ============================================================

from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional, Set
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

# ---------- Константы ----------
BIG_PENALTY = 1e4
EPS = 1e-9

# ---------- Динамическое препятствие ----------
@dataclass
class DynamicObstacle:
    pos: Tuple[int, int]           # текущая клетка
    every: int = 2                 # двигать каждые N шагов робота
    speed_cells: int = 1           # клеток за одно обновление
    mode: Literal["pursue"] = "pursue"  # тип поведения

# ---------- Параметры потенциала ----------
@dataclass
class RepulsiveConfig:
    eta: float = 120.0             # сила отталкивания
    rho0: float = 6.0              # радиус влияния в клетках
    cell_radius: float = 0.5       # «радиус» узла (для SDF)

@dataclass
class GridPotentialParams:
    k_att: float = 0.5
    repulsive_layers: List[RepulsiveConfig] = None
    big_penalty: float = BIG_PENALTY

    def __post_init__(self):
        if self.repulsive_layers is None:
            self.repulsive_layers = [
                RepulsiveConfig(eta=220.0, rho0=4.0, cell_radius=0.5),
                RepulsiveConfig(eta=90.0,  rho0=8.0, cell_radius=0.5),
            ]

# ---------- Поле потенциала на решётке ----------
class GridPotentialField:
    def __init__(self, width: int, height: int,
                 goal: Tuple[int, int],
                 static_obstacles: List[Tuple[int, int]],
                 dyn_obstacles: Optional[List[DynamicObstacle]],
                 params: GridPotentialParams):
        self.W = int(width)
        self.H = int(height)
        self.goal = (int(goal[0]), int(goal[1]))
        self.params = params

        self.static_obs: Set[Tuple[int, int]] = set(map(tuple, static_obstacles))
        self.dyn_obs: List[DynamicObstacle] = dyn_obstacles[:] if dyn_obstacles else []

        # Карты (обновляются при rebuild_potential)
        self.occ_mask = np.zeros((self.H, self.W), dtype=bool)  # занятость
        self.clearance = None                                   # SDF: >0 — свободно
        self.U_att = None
        self.U_rep = None
        self.U_total = None
        self.Ux = None
        self.Uy = None

        self.rebuild_potential()

    # ---- Вспомогательные сетки/множества ----
    def _grid_xy(self):
        X, Y = np.meshgrid(np.arange(self.W), np.arange(self.H), indexing='xy')
        return X, Y

    def _all_obstacles(self) -> Set[Tuple[int, int]]:
        # Исключаем цель на всякий случай
        goal_t = tuple(self.goal)
        all_obs = set(self.static_obs)
        for d in self.dyn_obs:
            if tuple(d.pos) != goal_t:
                all_obs.add(tuple(d.pos))
        all_obs.discard(goal_t)
        return all_obs

    def _update_occ_mask(self):
        self.occ_mask[:] = False
        for (x, y) in self._all_obstacles():
            if 0 <= x < self.W and 0 <= y < self.H:
                self.occ_mask[y, x] = True

    # ---- Построение потенциалов ----
    def build_U_att(self):
        X, Y = self._grid_xy()
        gx, gy = self.goal
        self.U_att = 0.5 * self.params.k_att * ((X - gx) ** 2 + (Y - gy) ** 2)

    @staticmethod
    def _euclidean_distance_to_set(W: int, H: int, occ: Set[Tuple[int, int]]) -> np.ndarray:
        """
        Наивный трансформ расстояний O(WH * |occ|). Для учебных размеров ок.
        """
        D = np.empty((H, W), dtype=float)
        if not occ:
            D.fill(np.inf)
            return D
        pts = np.array(list(occ), dtype=float)  # [K,2]
        for y in range(H):
            for x in range(W):
                dx = pts[:, 0] - x
                dy = pts[:, 1] - y
                D[y, x] = float(np.sqrt(np.min(dx * dx + dy * dy)))
        return D

    def build_U_rep(self):
        occ = self._all_obstacles()
        self._update_occ_mask()

        D = self._euclidean_distance_to_set(self.W, self.H, occ)
        Urep_total = np.zeros((self.H, self.W), dtype=float)

        # Единый клиренс (для логики планировщика)
        min_cell_r = min(cfg.cell_radius for cfg in self.params.repulsive_layers)
        self.clearance = D - min_cell_r  # >0 — свободно, <=0 — граница/внутри

        # Суммируем слои отталкивания
        for cfg in self.params.repulsive_layers:
            q = D - cfg.cell_radius
            Urep = np.zeros_like(Urep_total)

            inside = q <= 0           # в/на препятствии
            shell  = (q > 0) & (q <= cfg.rho0)  # зона влияния

            Urep[inside] += self.params.big_penalty
            q_shell = q[shell]
            Urep[shell] += 0.5 * cfg.eta * (1.0 / q_shell - 1.0 / cfg.rho0) ** 2

            Urep_total += Urep

        # Гарантия: занятость = большой штраф
        Urep_total[self.occ_mask] = np.maximum(Urep_total[self.occ_mask], self.params.big_penalty)

        self.U_rep = Urep_total

    def rebuild_potential(self):
        self.build_U_att()
        self.build_U_rep()
        self.U_total = self.U_att + self.U_rep
        Uy, Ux = np.gradient(self.U_total)  # (dU/dy, dU/dx)
        self.Ux, self.Uy = Ux, Uy

    # ---- Динамика препятствий ----
    def step_dynamics(self, robot_xy: Tuple[int, int], step_idx: int, reeval_every: int = 1):
        moved = False
        rx, ry = robot_xy
        for d in self.dyn_obs:
            if d.every <= 0 or step_idx % d.every != 0:
                continue
            if d.mode == "pursue":
                dx = np.sign(rx - d.pos[0])
                dy = np.sign(ry - d.pos[1])
                px, py = d.pos
                for _ in range(max(1, d.speed_cells)):
                    nx, ny = px + int(dx), py + int(dy)
                    if 0 <= nx < self.W and 0 <= ny < self.H and (nx, ny) != self.goal:
                        px, py = nx, ny
                        moved = True
                d.pos = (px, py)

        if moved and reeval_every > 0 and (step_idx % reeval_every == 0):
            self.rebuild_potential()

# ---------- Планировщик на решётке (пошаговый) ----------
@dataclass
class GridPlannerConfig:
    max_steps: int = 5000
    smooth: bool = True
    beta: float = 0.85                 # EMA-сглаживание направления
    prefer_grad: bool = True           # учитывать −∇U при выборе шага
    reeval_every: int = 1              # период пересчёта U при динамике
    allow_diag: bool = True            # разрешить диагонали
    tie_bias: float = 0.02             # бонус за согласование с направлением
    safety_clearance: float = 0.0      # «подушка» до препятствий (0..1+ клетки)

class GridStepper:
    """
    Пошаговый «симулятор» планирования: удобно для анимации.
    """
    def __init__(self, field: GridPotentialField, cfg: GridPlannerConfig, start: Tuple[int, int]):
        self.F = field
        self.cfg = cfg
        self.start = (int(start[0]), int(start[1]))
        self.reset()

    def reset(self):
        self.x, self.y = self.start
        self.step_idx = 0
        self.path = [(self.x, self.y)]
        self.energies = [float(self.F.U_total[self.y, self.x])]
        self.v = np.zeros(2)  # EMA-направление
        self.finished = False
        self.reached = False

    @staticmethod
    def neighbors(x: int, y: int, W: int, H: int, allow_diag: bool = True):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if allow_diag:
            dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                yield nx, ny

    def _diag_corner_free(self, x: int, y: int, nx: int, ny: int) -> bool:
        # запрет «среза угла» через два соседних препятствия
        if abs(nx - x) == 1 and abs(ny - y) == 1:
            if self.F.occ_mask[y, nx] or self.F.occ_mask[ny, x]:
                return False
        return True

    def step(self):
        """
        Один шаг планировщика:
          • двигаем динамические препятствия
          • считаем −∇U и выбираем лучшего соседа
          • применяем жёсткую маску + safety_clearance + corner-cutting запрет
        Возвращает: dict(state) для удобства анимации.
        """
        if self.finished:
            return {"done": True, "reached": self.reached}

        self.step_idx += 1

        # динамика препятствий и (возможно) пересчёт поля
        self.F.step_dynamics((self.x, self.y), step_idx=self.step_idx, reeval_every=self.cfg.reeval_every)

        gx, gy = self.F.goal
        if (self.x, self.y) == (gx, gy) or self.step_idx > self.cfg.max_steps:
            self.finished = True
            self.reached = (self.x, self.y) == (gx, gy)
            return {"done": True, "reached": self.reached}

        gvec = np.array([self.F.Ux[self.y, self.x], self.F.Uy[self.y, self.x]], dtype=float)
        dvec = -gvec
        guide = self.cfg.beta * self.v + (1.0 - self.cfg.beta) * dvec if self.cfg.smooth else dvec
        if self.cfg.smooth:
            self.v = guide

        best = None
        best_score = +1e18

        for nx, ny in self.neighbors(self.x, self.y, self.F.W, self.F.H, self.cfg.allow_diag):
            # 1) Жёсткая занятость
            if self.F.occ_mask[ny, nx]:
                continue
            # 2) «Подушка» безопасности
            if self.F.clearance is not None and self.F.clearance[ny, nx] <= self.cfg.safety_clearance:
                continue
            # 3) Corner-cutting
            if not self._diag_corner_free(self.x, self.y, nx, ny):
                continue

            Ucand = self.F.U_total[ny, nx]
            score = Ucand
            if self.cfg.prefer_grad:
                step_dir = np.array([nx - self.x, ny - self.y], dtype=float)
                align = float(np.dot(step_dir, guide) /
                              ((np.linalg.norm(step_dir) + EPS) * (np.linalg.norm(guide) + EPS)))
                score -= self.cfg.tie_bias * align

            if score < best_score:
                best_score = score
                best = (nx, ny)

        if best is None:
            self.finished = True
            self.reached = False
            return {"done": True, "reached": False}

        self.x, self.y = best
        self.path.append((self.x, self.y))
        self.energies.append(float(self.F.U_total[self.y, self.x]))

        if (self.x, self.y) == (gx, gy):
            self.finished = True
            self.reached = True

        return {
            "done": self.finished,
            "reached": self.reached,
            "x": self.x, "y": self.y,
            "step": self.step_idx,
            "path": self.path
        }

# ---------- Визуализация (статичная) ----------
@dataclass
class GridVizConfig:
    colormap: str = "turbo"
    robust_clip: Optional[Tuple[float, float]] = (3.0, 97.0)  # процентили для клипа (None — без клипа)
    draw_penalty: bool = True          # показать точки препятствий
    show_occ_mask: bool = True         # полупрозрачная маска стен
    arrows_per_axis: int = 28          # плотность стрелок для −∇U
    quiver_normalize: bool = True      # нормировать длину стрелок
    quiver_color_by_magnitude: bool = True  # цвет по |−∇U|
    path_linewidth: float = 2.6

class GridVisualizer:
    def __init__(self, field: GridPotentialField, viz: GridVizConfig):
        self.F = field
        self.viz = viz

    def _clip_range(self, M: np.ndarray) -> Tuple[float, float]:
        if self.viz.robust_clip is None:
            vmin, vmax = float(np.min(M)), float(np.max(M))
            return vmin, max(vmax, vmin + 1e-9)
        lo, hi = self.viz.robust_clip
        vmin = float(np.nanpercentile(M, lo))
        vmax = float(np.nanpercentile(M, hi))
        return vmin, max(vmax, vmin + 1e-9)

    def plot_two_maps(self,
                      start: Tuple[int, int],
                      goal: Tuple[int, int],
                      path: List[Tuple[int, int]],
                      mode: Literal["heat", "quiver"] = "heat"):
        X, Y = np.meshgrid(np.arange(self.F.W), np.arange(self.F.H), indexing='xy')
        U = self.F.U_total.copy()
        vmin, vmax = self._clip_range(U)

        fig = plt.figure(figsize=(13, 5))

        # Левая панель — путь + препятствия
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(U, origin="lower", cmap="Greys", alpha=0.15, vmin=vmin, vmax=vmax)
        if self.viz.show_occ_mask:
            ax1.imshow(self.F.occ_mask, origin="lower", cmap="gray_r", alpha=0.25)
        xs, ys = zip(*path)
        ax1.plot(xs, ys, linewidth=self.viz.path_linewidth, color="#1f77b4", label="путь")
        ax1.scatter([start[0]], [start[1]], s=60, marker="o", color="#2ca02c", label="старт")
        ax1.scatter([goal[0]],  [goal[1]],  s=140, marker="*", color="#d62728", label="цель")

        # Разделяем статические и динамические препятствия для ясности
        if self.viz.draw_penalty:
            if self.F.static_obs:
                xs_s, ys_s = zip(*self.F.static_obs)
                ax1.scatter(xs_s, ys_s, s=12, color="#111", label="статич. преп.")
            if self.F.dyn_obs:
                xs_d, ys_d = zip(*[d.pos for d in self.F.dyn_obs])
                ax1.scatter(xs_d, ys_d, s=30, marker="s", color="#e377c2", label="динамич. преп.")

        ax1.set_xlim(-0.5, self.F.W - 0.5)
        ax1.set_ylim(-0.5, self.F.H - 0.5)
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_title("Путь и препятствия")
        ax1.set_xlabel("x"); ax1.set_ylabel("y")
        ax1.legend(loc="best")

        # Правая панель — тепловая карта U или поле −∇U
        ax2 = fig.add_subplot(1, 2, 2)
        if mode == "heat":
            hm = ax2.imshow(U, origin="lower",
                            cmap=self.viz.colormap, vmin=vmin, vmax=vmax,
                            interpolation="nearest", aspect="equal")
            if self.viz.show_occ_mask:
                ax2.imshow(self.F.occ_mask, origin="lower", cmap="gray_r", alpha=0.25)
            cbar = plt.colorbar(hm, ax=ax2, shrink=0.9); cbar.set_label("U")
            ax2.set_title("Тепловая карта U (rep+att)")
        elif mode == "quiver":
            Fx = -self.F.Ux
            Fy = -self.F.Uy
            Fmag = np.sqrt(Fx * Fx + Fy * Fy) + 1e-9
            if self.viz.quiver_normalize:
                Fx_show, Fy_show = Fx / Fmag, Fy / Fmag
            else:
                Fx_show, Fy_show = Fx, Fy

            skip = max(1, int(min(self.F.W, self.F.H) // self.viz.arrows_per_axis))
            if self.viz.quiver_color_by_magnitude:
                fmin, fmax = self._clip_range(Fmag)
                norm = Normalize(vmin=fmin, vmax=fmax)
                C = norm(Fmag[::skip, ::skip])
                q = ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                               Fx_show[::skip, ::skip], Fy_show[::skip, ::skip],
                               C, cmap=self.viz.colormap, angles="xy", scale_units="xy",
                               scale=0.8 if self.viz.quiver_normalize else None, width=0.003)
                sm = plt.cm.ScalarMappable(cmap=self.viz.colormap, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=ax2, shrink=0.9, label="|−∇U|")
            else:
                ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                           Fx_show[::skip, ::skip], Fy_show[::skip, ::skip],
                           angles="xy", scale_units="xy",
                           scale=0.8 if self.viz.quiver_normalize else None, width=0.003)
            if self.viz.show_occ_mask:
                ax2.imshow(self.F.occ_mask, origin="lower", cmap="gray_r", alpha=0.25)
            ax2.set_title("Векторное поле −∇U")
        else:
            raise ValueError("mode должен быть 'heat' или 'quiver'")

        ax2.scatter([goal[0]], [goal[1]], s=70, marker="*", color="#d62728", zorder=3)
        ax2.set_xlim(-0.5, self.F.W - 0.5)
        ax2.set_ylim(-0.5, self.F.H - 0.5)
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_xlabel("x"); ax2.set_ylabel("y")

        plt.tight_layout()
        plt.show()

# ---------- АНИМАЦИЯ ----------
@dataclass
class AnimationConfig:
    mode: Literal["heat", "quiver"] = "heat"
    interval_ms: int = 120                 # скорость анимации
    max_frames: int = 4000                 # ограничение кадров
    trail: Optional[int] = None            # длина хвоста пути (None — весь путь)
    arrows_per_axis: int = 28              # для quiver
    quiver_normalize: bool = False         # хотим видеть «силу» по длине
    quiver_color_by_magnitude: bool = True
    colormap: str = "turbo"
    robust_clip: Optional[Tuple[float, float]] = (3.0, 97.0)
    show_occ_mask: bool = True

def animate_simulation(field: GridPotentialField,
                       stepper: GridStepper,
                       anim_cfg: AnimationConfig):
    """
    Возвращает объект FuncAnimation. Ничего не сохраняет.
    Чтобы показать: plt.show().
    """
    # Подготовка данных
    X, Y = np.meshgrid(np.arange(field.W), np.arange(field.H), indexing='xy')
    U = field.U_total.copy()

    # Клип для цветов
    def clip_range(M: np.ndarray) -> Tuple[float, float]:
        if anim_cfg.robust_clip is None:
            vmin, vmax = float(np.min(M)), float(np.max(M))
        else:
            lo, hi = anim_cfg.robust_clip
            vmin = float(np.nanpercentile(M, lo))
            vmax = float(np.nanpercentile(M, hi))
        return vmin, max(vmax, vmin + 1e-9)

    vmin, vmax = clip_range(U)

    fig = plt.figure(figsize=(13, 5))

    # Левая панель: путь/препятствия
    ax1 = fig.add_subplot(1, 2, 1)
    bg1 = ax1.imshow(U, origin="lower", cmap="Greys", alpha=0.12, vmin=vmin, vmax=vmax)
    if anim_cfg.show_occ_mask:
        occ1 = ax1.imshow(field.occ_mask, origin="lower", cmap="gray_r", alpha=0.25)
    else:
        occ1 = None
    (px,), (py,) = (np.array([stepper.x]),), (np.array([stepper.y]),)
    path_line, = ax1.plot([stepper.x], [stepper.y], color="#1f77b4", lw=2.6)
    robot_sc = ax1.scatter([stepper.x], [stepper.y], s=60, c="#1f77b4", edgecolors="k", zorder=3, label="робот")
    ax1.scatter([stepper.start[0]], [stepper.start[1]], s=50, marker="o", color="#2ca02c", label="старт")
    ax1.scatter([field.goal[0]], [field.goal[1]], s=140, marker="*", color="#d62728", label="цель")
    # статические
    if field.static_obs:
        xs_s, ys_s = zip(*field.static_obs)
        ax1.scatter(xs_s, ys_s, s=12, color="#111", label="статич. преп.")
    # динамические
    dyn_sc = None
    if field.dyn_obs:
        xs_d, ys_d = zip(*[d.pos for d in field.dyn_obs])
        dyn_sc = ax1.scatter(xs_d, ys_d, s=30, marker="s", color="#e377c2", label="динамич. преп.")

    ax1.set_xlim(-0.5, field.W - 0.5)
    ax1.set_ylim(-0.5, field.H - 0.5)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Анимация: путь и препятствия")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.legend(loc="best")

    # Правая панель: тепловая карта или −∇U
    ax2 = fig.add_subplot(1, 2, 2)
    if anim_cfg.mode == "heat":
        hm = ax2.imshow(U, origin="lower", cmap=anim_cfg.colormap, vmin=vmin, vmax=vmax,
                        interpolation="nearest", aspect="equal")
        cbar = plt.colorbar(hm, ax=ax2, shrink=0.9); cbar.set_label("U")
        if anim_cfg.show_occ_mask:
            occ2 = ax2.imshow(field.occ_mask, origin="lower", cmap="gray_r", alpha=0.25)
        else:
            occ2 = None
        quiv = None; sm = None
    elif anim_cfg.mode == "quiver":
        Fx = -field.Ux; Fy = -field.Uy
        Fmag = np.sqrt(Fx*Fx + Fy*Fy) + 1e-9
        if anim_cfg.quiver_normalize:
            Fx_show, Fy_show = Fx/Fmag, Fy/Fmag
        else:
            Fx_show, Fy_show = Fx, Fy
        skip = max(1, int(min(field.W, field.H) // anim_cfg.arrows_per_axis))
        if anim_cfg.quiver_color_by_magnitude:
            fmin, fmax = clip_range(Fmag)
            norm = Normalize(vmin=fmin, vmax=fmax)
            C = norm(Fmag[::skip, ::skip])
            quiv = ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                              Fx_show[::skip, ::skip], Fy_show[::skip, ::skip],
                              C, cmap=anim_cfg.colormap, angles="xy", scale_units="xy",
                              scale=0.8 if anim_cfg.quiver_normalize else None, width=0.003)
            sm = plt.cm.ScalarMappable(cmap=anim_cfg.colormap, norm=norm); sm.set_array([])
            plt.colorbar(sm, ax=ax2, shrink=0.9, label="|−∇U|")
        else:
            quiv = ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                              Fx_show[::skip, ::skip], Fy_show[::skip, ::skip],
                              angles="xy", scale_units="xy",
                              scale=0.8 if anim_cfg.quiver_normalize else None, width=0.003)
            sm = None
        if anim_cfg.show_occ_mask:
            occ2 = ax2.imshow(field.occ_mask, origin="lower", cmap="gray_r", alpha=0.25)
        else:
            occ2 = None
        hm = None
    else:
        raise ValueError("AnimationConfig.mode: 'heat' или 'quiver'")

    ax2.scatter([field.goal[0]], [field.goal[1]], s=70, marker="*", color="#d62728", zorder=3)
    ax2.set_xlim(-0.5, field.W - 0.5)
    ax2.set_ylim(-0.5, field.H - 0.5)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.set_title("Анимация: " + ("U (rep+att)" if anim_cfg.mode=="heat" else "поле −∇U"))

    plt.tight_layout()

    # --- Функции обновления кадров ---
    def update_frame(_):
        # один шаг планировщика
        state = stepper.step()

        # пересобираем локальные ссылки (поле могло измениться)
        if anim_cfg.mode == "heat":
            U[:] = field.U_total
            vmin_, vmax_ = clip_range(U)
            hm.set_data(U)
            hm.set_clim(vmin_, vmax_)
            if anim_cfg.show_occ_mask:
                occ2.set_data(field.occ_mask)
            bg1.set_data(U)
            bg1.set_clim(vmin_, vmax_)
            if anim_cfg.show_occ_mask:
                occ1.set_data(field.occ_mask)
        else:
            Fx = -field.Ux; Fy = -field.Uy
            Fmag = np.sqrt(Fx*Fx + Fy*Fy) + 1e-9
            if anim_cfg.quiver_normalize:
                Fx_show, Fy_show = Fx/Fmag, Fy/Fmag
            else:
                Fx_show, Fy_show = Fx, Fy
            skip = max(1, int(min(field.W, field.H) // anim_cfg.arrows_per_axis))
            # Обновление поля стрелок
            quiv.set_UVC(Fx_show[::skip, ::skip], Fy_show[::skip, ::skip])
            if anim_cfg.quiver_color_by_magnitude and sm is not None:
                fmin, fmax = clip_range(Fmag)
                sm.norm = Normalize(vmin=fmin, vmax=fmax)
                quiv.set_array(sm.norm(Fmag[::skip, ::skip]))
            if anim_cfg.show_occ_mask:
                occ2.set_data(field.occ_mask)
            if anim_cfg.show_occ_mask:
                occ1.set_data(field.occ_mask)

        # путь/робот
        trace = stepper.path if anim_cfg.trail is None else stepper.path[-anim_cfg.trail:]
        xs, ys = zip(*trace)
        path_line.set_data(xs, ys)
        robot_sc.set_offsets(np.c_[stepper.x, stepper.y])

        # динамические препятствия
        if field.dyn_obs and dyn_sc is not None:
            xs_d, ys_d = zip(*[d.pos for d in field.dyn_obs])
            dyn_sc.set_offsets(np.c_[xs_d, ys_d])

        # завершение
        if state.get("done") or (stepper.step_idx >= anim_cfg.max_frames):
            anim.event_source.stop()
        return []

    anim = FuncAnimation(fig, update_frame, interval=anim_cfg.interval_ms, blit=False)
    return anim

# ---------- ПРИМЕР ИСПОЛЬЗОВАНИЯ (РАСКОММЕНТИРУЙТЕ У СЕБЯ) ----------
if __name__ == "__main__":
    width, height = 60, 40
    start = (2, 2)
    goal  = (50, 30)

    # Статика: стенки и точки
    static_obstacles = [(20, y) for y in range(5, 35)] + \
                       [(35, y) for y in range(4, 25)] + \
                       [(10, 10), (11, 10), (12, 11)]
    # Динамика: «преследователь»
    dyn = [DynamicObstacle(pos=(25, 5), every=2, speed_cells=1, mode="pursue")]

    pot_params = GridPotentialParams(
        k_att=0.35,
        repulsive_layers=[
            RepulsiveConfig(eta=220.0, rho0=4.0, cell_radius=0.5),
            RepulsiveConfig(eta=90.0,  rho0=8.0, cell_radius=0.5),
        ],
        big_penalty=1e4
    )

    field = GridPotentialField(width, height, goal, static_obstacles, dyn, pot_params)
    stepper = GridStepper(field,
                          GridPlannerConfig(max_steps=4000, smooth=True, beta=0.85,
                                            prefer_grad=True, reeval_every=1,
                                            allow_diag=True, tie_bias=0.02,
                                            safety_clearance=0.0),
                          start)

    # Статичные карты (для отладки):
    # GridVisualizer(field, GridVizConfig()).plot_two_maps(start, goal, stepper.path, mode="heat")

    # АНИМАЦИЯ:
    anim_cfg = AnimationConfig(mode="heat", interval_ms=120, max_frames=3000,
                               trail=300, arrows_per_axis=28,
                               quiver_normalize=False, quiver_color_by_magnitude=True,
                               colormap="turbo", robust_clip=(3, 97), show_occ_mask=True)
    anim = animate_simulation(field, stepper, anim_cfg)
    plt.show()  # <- РАСКОММЕНТИРУЙТЕ у себя, чтобы запустить анимацию
