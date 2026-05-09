# 圆管 Green solver analytic mirror-charge 对照 — 设计

日期：2026-05-09
范围：在 `dcbeamtrack` 中加入 analytic mirror-charge 空间电荷 solver 作为长期 ground truth，并以两个独立脚本（静态场扫描、t2d_cir_V1 端到端追踪）覆盖 `CircularGreenSpaceChargeSolver` 的诊断需求。

## 背景

当前 `CircularGreenSpaceChargeSolver`（`src/dcbeamtrack/spacecharge/circular_green.py`）在初始化时预算 Green 函数表 `G_Er[i, j, k]`、`G_Eu[i, j, k]`，索引为 `(target 环 i, source 环 j, Δθ_k)`。求和时对每个粒子按 `r` snap 到最近一环、对 `Δθ` 做角向线性插值。这两步引入两个数值近似：

- 径向零阶 snap（`r → gr[ir]`）
- 角向一阶线性插值

`tests/test_circular_green.py` 中已有 `_direct_mirror_charge_field`（纯 Python O(N²)）作为参考，验证在 ring/spoke 节点上 Green 与 analytic 严格匹配（atol ~1e-9）。但 off-grid 配置下 Green 路径的真实误差量级、收敛阶、对 tracker 端到端 envelope 的影响，目前**没有持续监测**。

旧 `t2dcore` 的 `RUGrid` 路径中 Eu 公式有 bug（多 `ra²` 因子），所以 `scripts/compare_t2d_cir_v1.py`（vs legacy）只能作为"同量级"参照、不能作 ground truth。本次工作补齐这一空缺。

## 目标

1. 在 `dcbeamtrack.spacecharge` 下新增 `AnalyticMirrorChargeSpaceChargeSolver`，与 `CircularGreenSpaceChargeSolver` 在 tracker 集成层完全等价（YAGNI：不支持 off-source `field_at`）
2. 以静态场扫描脚本量化 Green 在 off-grid 下的实测误差与 `n_theta` 收敛
3. 以端到端追踪脚本在 t2d_cir_V1 参数下确认 Green 与 analytic 的 envelope 差异在工程可接受范围
4. 在 pytest 里保留 analytic 自身实现正确性 + 一个 off-grid 宽松 canary 防止 Green 全局回退
5. 同步更新 `docs/status_and_plan.md`

## 非目标

- 不为 analytic solver 实现 off-source `field_at` —— `CircularGreenFieldMap` 当前也只支持 source 位置查询，对齐两者 API 行为；off-source 等做场可视化时再补
- 不动现有 `CircularGreenSpaceChargeSolver` 实现
- 不动 `tests/test_circular_green.py` 中现有 ring/spoke 严格匹配测试
- 不实现 GPU/分块加速 —— numba CPU 在 t2d_cir_V1 规模（~2k 粒子）足够
- pytest 不跑 tracker 端到端 vs analytic（量级耗时不适合 CI）

## 设计

### 文件布局

新增：
```
src/dcbeamtrack/spacecharge/analytic_mirror_charge.py
tests/test_analytic_mirror_charge.py
scripts/static_compare_analytic_vs_green.py
scripts/compare_cir_v1_analytic.py
docs/superpowers/specs/2026-05-09-circular-green-analytic-comparison-design.md   # 本文件
```

修改：
```
src/dcbeamtrack/spacecharge/__init__.py    # +导出 AnalyticMirrorChargeSpaceChargeSolver, AnalyticMirrorChargeFieldMap
src/dcbeamtrack/__init__.py                # +导出 AnalyticMirrorChargeSpaceChargeSolver, AnalyticMirrorChargeFieldMap
docs/status_and_plan.md                    # 详见 §6
```

### `AnalyticMirrorChargeSpaceChargeSolver`

API：
```python
class AnalyticMirrorChargeSpaceChargeSolver:
    def __init__(self, radius: float) -> None: ...
    def solve(self, distribution) -> AnalyticMirrorChargeFieldMap: ...

@dataclass(frozen=True)
class AnalyticMirrorChargeFieldMap:
    ex: np.ndarray
    ey: np.ndarray
    def field_at(self, x, y) -> tuple[np.ndarray, np.ndarray]: ...
```

与 `CircularGreenSpaceChargeSolver` 的对齐点：
- 跳过 `wall_hit=True` 粒子（既不作 source，也不写出 Ex/Ey）
- 跳过 `r >= radius` 粒子
- `FieldMap.field_at(x, y)`：仅 source size 匹配时返回缓存值，否则 `raise NotImplementedError`
- 系数 `1/(2π·EPS0)` 用 `dcbeamtrack.constants.EPS0`

差异点：
- 无 `n_r` / `n_theta` / `r_eps` 参数（解析公式不需要离散网格）
- 每对 (i, j) 直接算两项：
  - 源贡献：`coef * q_j * (r_i - r_j) / |r_i - r_j|^2`
  - 镜像项：`-coef * q_j * (r_i - r_j_image) / |r_i - r_j_image|^2`
  - 其中 `r_j_image = (radius² / |r_j|²) * r_j`
- 跳过 `i == j`

numba kernel 签名（与 `_pp_fast_numba` 同模式）：
```python
@njit(parallel=True, fastmath=True, cache=True)
def _analytic_mirror_charge_numba(
    radius: float,
    x: np.ndarray, y: np.ndarray, q: np.ndarray,
    active_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ex, ey) sized like x; inactive entries left at 0.0."""
```

外层 `prange` over target i，内层串行 j。

### `scripts/static_compare_analytic_vs_green.py`

CLI：
```bash
python scripts/static_compare_analytic_vs_green.py [--output-dir DIR] [--seed SEED]
```

固定参数：`radius=150e-6`, `n_r=200`。无需 t2d 物理参数。

底线三构型（每个 ~200 粒子）：
1. **on-axis chain** — 沿 +x 轴均匀分布 `[0.05R, 0.85R]`，所有 `λ=1`
2. **two-cluster off-axis** — 两团高斯，中心 `(+0.4R, 0)` 和 `(-0.3R, +0.2R)`，σ=0.05R
3. **random uniform disk** — `r ≤ 0.8R` 内均匀采样，固定 seed

对每个构型独立计算：
- analytic solver `(Ex, Ey)_ref`
- Green solver（`n_r=200, n_theta=360`）`(Ex, Ey)_green`
- 逐粒子 `|ΔE| / max(|E_ref|)`、max abs、max rel

`n_theta` 扫描（仅构型 #3）：
- `n_theta ∈ {90, 180, 360, 720, 1440}`，固定 `n_r=200`
- 输出收敛表 + loglog 折线图

输出：
- `artifacts/analytic_vs_green/static_summary.json`
- `artifacts/analytic_vs_green/static_arrays.npz`
- `artifacts/analytic_vs_green/n_theta_convergence.png`

诊断脚本，**不设硬通过线**。

### `scripts/compare_cir_v1_analytic.py`

CLI（与 `compare_t2d_cir_v1.py` 对齐）：
```bash
python scripts/compare_cir_v1_analytic.py \
  [--steps 5000] [--snapshot-stride 10] [--seed 2000] \
  [--precision float32] [--n-r 200] [--n-theta 360] \
  [--output-dir DIR]
```
`--n-r`/`--n-theta` 只影响 Green solver。

流程：
1. 复用 `compare_t2d_cir_v1.py` 的初始粒子构造路径（legacy `TriGrid` + `UniformDisk` + `randomize_directions(2 mrad, rebuild_transverse, seed)`），三对照脚本共享同一份初始分布定义
2. 转 `SliceDistribution`
3. 跑两次：
   - `DcBeamSimulation.from_solver(spacecharge_solver=AnalyticMirrorChargeSpaceChargeSolver(radius), boundary=CircularBoundary(radius), magnetic_field=UniformMagneticField(bz=0.6), config=...)`
   - `DcBeamSimulation.circular_green(radius=..., n_r=..., n_theta=..., magnetic_field=..., config=...)`

对比指标（逐 snapshot）：
- `r_max(z)`, `r_mean(z)` 包络的 max abs/rel diff
- 终态 `(x, y, vx, vy, vz)` 的 max abs diff
- `wall_hit` 计数（两侧应一致）

输出（与 `compare_t2d_cir_v1.py` 同结构）：
- `artifacts/cir_v1_analytic/summary_steps{S}_seed{X}_{precision}.json`
- `artifacts/cir_v1_analytic/arrays_steps{S}_seed{X}_{precision}.npz`

预计运行时间 30-90s 量级（~3800 step × 2095² 对，两次跑）。诊断脚本，不设硬通过线。

### `tests/test_analytic_mirror_charge.py`

新增 5 个测试：

1. `test_analytic_two_particle_symmetry_about_xaxis` — analytic 自身对称性
2. `test_analytic_matches_pure_python_reference` — analytic numba 与 `tests/test_circular_green.py:_direct_mirror_charge_field` 在小样本严格匹配。`atol = 1e-10 * scale`（`scale = max(|E_ref|)`），余量给 numba `fastmath=True` 可能引入的浮点重排（fma、并行求和顺序）
3. `test_analytic_skips_wall_hit_sources` — 与 Green solver 同行为
4. `test_analytic_field_at_size_mismatch_raises`
5. `test_circular_green_within_tolerance_of_analytic_off_grid` — 随机 off-grid 粒子（`n_r=200, n_theta=720`），`max_rel_err ≤ 5e-2`，作为 Green 全局回退的 canary

阈值 5e-2 是工程余量，远高于 §3 静态扫描的实测水位（~1e-2 量级）。

不动 `tests/test_circular_green.py` 中现有 5 个测试（保留 ring/spoke 严格匹配）。

`pytest` 总数：27 → 32（+5）。

### `docs/status_and_plan.md` 更新

不动旧内容，只增量：

1. 顶部日期头 → `2026-05-09（圆管 Green analytic 对照就位）`
2. "圆管 Green solver" 小节末尾追加一行：`analytic mirror-charge 对照实现 AnalyticMirrorChargeSpaceChargeSolver（无离散网格、O(N²) numba）作为长期 ground truth；off-grid 误差实测见 artifacts/analytic_vs_green/static_summary.json`
3. "当前测试状态" 测试列表加一条 analytic 对照覆盖；测试总数 27 → 32
4. "近期计划 §4 圆管 Green solver 后续" — 把 analytic ground truth 脚本与 t2d 风格端到端对照两条标为已完成；保留未完成的 `field_at` off-source、性能两条
5. 新增 "analytic mirror-charge 对照" 小节，记录三脚本各自用途、t2d_cir_V1 端到端对照实测数字、`n_theta` 收敛趋势
6. "下一步建议" 删掉 "圆管 Green 长期对照脚本" 项

## 数据流

```
SliceDistribution
   │
   ├── extra "wall_hit" 标签（可选）
   │
   ▼
AnalyticMirrorChargeSpaceChargeSolver.solve
   │
   ├── 提取 x, y, lam
   ├── 构造 active mask（!wall_hit & r < radius）
   ├── _analytic_mirror_charge_numba(radius, x, y, lam, active_idx)
   │     prange over target i ∈ active:
   │        for source j ∈ active, j != i:
   │            ex_i += coef * q_j * (dx_src/r2_src - dx_img/r2_img)
   │            ey_i += coef * q_j * (dy_src/r2_src - dy_img/r2_img)
   │
   ▼
AnalyticMirrorChargeFieldMap(ex, ey)
   │
   └── field_at(x, y) → 仅 size 匹配时返回 (ex.copy(), ey.copy())
```

## 错误处理

- `AnalyticMirrorChargeSpaceChargeSolver.__init__`：`radius <= 0` 抛 `ValueError`
- `field_at(x, y)`：size 不匹配抛 `NotImplementedError`（与 Green 一致的措辞）
- numba kernel：`r_src2 == 0`（粒子在轴心）时镜像点定义不良；analytic 公式自然给出 ∞——靠 `r < radius` 上游过滤 + `r_eps` 概念上的安全保障。**实际上**：`r=0` 粒子既是 source 也是 target 时，self-skip (`i == j`) 已避开除零；`r=0` 作为 source 对其它 target，分母 `|r_i - r_j_image|² → ∞`，贡献趋零，数学上合理但浮点会发散。**实现上**：在 numba kernel 入口检查 `r_src2 < r_eps²`（`r_eps = 1e-18`），若是则跳过该 source（贡献置零）。

## 测试

详见 §设计/`tests/test_analytic_mirror_charge.py`。

## 文档

`docs/status_and_plan.md` 更新详见 §设计/最后一节。本设计文档落到 `docs/superpowers/specs/2026-05-09-circular-green-analytic-comparison-design.md`。

## 验收

- 5 个新 pytest 全部通过；总数 32 passed
- `scripts/static_compare_analytic_vs_green.py` 跑完输出三 case 摘要 + 收敛图
- `scripts/compare_cir_v1_analytic.py` 跑完 t2d_cir_V1 全量参数，summary JSON 中 `r_max max rel diff` 落在 1e-4 ~ 1e-3 量级（实测预期，不是硬门槛）
- `docs/status_and_plan.md` 更新点全部就位
