# dcbeamtrack 工作进度与计划

更新日期：2026-05-09（圆管 Green solver 第一版完成）

## 目标

`dcbeamtrack` 是对旧 `t2d` / `t2dcore` 代码的重构。目标不是保持旧接口兼容，而是在保留直流电子束核心计算能力的前提下，形成结构清楚、可测试、可扩展的新包。

当前物理模型聚焦：

- 直流电子束；
- 固定参考 `z` 平面推进；
- 三维粒子动量与二维横向空间电荷场；
- 以 `partdist.SliceDistribution` 作为粒子数据层；
- 外磁场由独立对象提供，tracker 只从对象读取 `Bx, By, Bz`。

## 当前状态

### 已完成的主计算路径

已经跑通并验证的第一条数值路径是：

- `SliceDistribution(z=...)` 粒子数据层；
- 矩形边界 FDM 空间电荷；
- CIC / NGP / TSC 线电荷沉积；
- CPU relativistic Boris 推进；
- 固定参考 `z` 平面推进；
- 轴线 `Bz(z)` 表构造外磁场；
- 近轴 `Br/r = -0.5 dBz/dz`，导数采用旧代码的局部多项式稳定算法；
- `float32` / `float64` 状态推进；
- `Lz` 按旧代码语义截断有效步数；
- 结果对象保存、加载和基础包络分析。

### 与旧代码的数值对比

已使用 `/mnt/d/OneDrive/pys/t2d_rec_V4.py` 的参数做过严格对比。对比方式是：

- 旧代码仍使用旧 `t2dcore.py` 运行；
- 初始粒子分布使用旧 `TriGrid(hmax=30e-7)` 生成；
- 新代码使用同一份初始粒子转换成 `SliceDistribution`；
- 旧代码和新代码使用同一矩形 FDM 网格、同一磁场表、同一 `boris_bzbr` 物理含义；
- 比较逐快照坐标、速度和包络。

完整 `t2d_rec_V4` 参数下，旧代码请求 `5000` step，但实际按 `Lz` 截断为 `3771` step。主要对比结果：

- 粒子数：`1363`
- 快照数：`3772`
- 最终 `z`：新旧均为 `2.999877557158e-02 m`
- 全过程 `r_max` 最大差异：约 `4.9e-11 m`
- 全过程 `r_mean` 最大差异：约 `1.0e-12 m`

短步数回归脚本也保持一致：

- `x_max_abs = 3.64e-12 m`
- `y_max_abs = 3.64e-12 m`
- `r_max_max_abs = 5.22e-13 m`
- `r_mean_max_abs = 1.25e-14 m`

本地旧代码对比工具：

```bash
PYTHONPATH=src python scripts/compare_t2d_rec_v4.py --steps 5000
```

### 当前公共 API

正式入口：

- `DcBeamSimulation.rect_fdm(...)`
- `DcBeamSimulation.from_solver(...)`
- `track_slice(...)`

主要配置与对象：

- `TrackConfig`
- `AxialMagneticField`
- `UniformMagneticField`
- `RectGrid`
- `RectFdmSpaceChargeSolver`
- `RectBoundary`
- `UnboundedBoundary`
- `SliceTrackResult`

结果类能力：

- 序列式访问快照；
- `array(key)` 提取逐快照数组；
- `radial_envelope("max" | "mean" | "rms")`；
- `r_max()` / `r_mean()` / `r_rms()`；
- `waist(kind)`；
- `get_at_z(z_value)`；
- `save_npz(path)` / `load_npz(path)`。

### Solver / Tracker 解耦状态

解耦协议：

- `SpaceChargeSolver` 协议：`solve(distribution) -> field_map`
- `FieldMap2DProtocol` 协议：`field_at(x, y) -> Ex, Ey`
- `TransverseBoundary` 协议：`inside_open(...)` 与 `absorb(...)`

`track_slice` 主循环只依赖三个协议：`spacecharge_solver`、`magnetic_field`、`boundary`。tracker 模块不再 import 任何矩形实现。`track_slice` 的签名为：

```python
track_slice(
    distribution,
    *,
    spacecharge_solver,   # 必填
    magnetic_field,       # 必填
    config,               # 必填
    boundary=None,        # 可选，None → UnboundedBoundary
)
```

矩形场景的便捷工厂仍是 `DcBeamSimulation.rect_fdm(grid=..., magnetic_field=..., config=...)`，它内部用同一份 `RectGrid` 同时构造 `RectFdmSpaceChargeSolver` 和 `RectBoundary`，保证场域和粒子壁一致。

撞墙行为完全交给 `TransverseBoundary` 表达：

- `UnboundedBoundary`：无壁，粒子永远活跃
- `RectBoundary`：超出矩形墙的粒子被 clip 到墙上、横向速度清零
- `CircularBoundary(radius)`：超出圆管的粒子被沿径向 snap 到 `r=radius`、横向速度清零

`boundary.absorb(x, y, vx, vy)` 返回新撞墙粒子的 bool mask，tracker OR 累加到 `wall_hit` 标签里写入 `SliceDistribution.extras`，`SliceTrackResult.live_mask()` 可用来过滤"贴墙滑行"假轨迹。原 `TrackConfig.absorbing_boundary` 布尔 flag 已删除；`track_slice(..., grid=...)` legacy 兜底已删除。

### 圆管 Green solver

`CircularGreenSpaceChargeSolver(radius, n_r=96, n_theta=720)`：

- 镜像电荷 Green 函数，求接地圆管内的横向空间电荷场；
- 极坐标 kernel `G_Er[i, j, k]`、`G_Eu[i, j, k]` 在初始化时一次性预计算，scaled by `1/(2π·ε0)`；
- 求和后端：numba `@njit(parallel=True, cache=True)`，O(N²) 全部对粒子；
- 求解时自动跳过 `wall_hit=True` 的粒子（既不作为 source，也不写入 Ex/Ey）；
- `FieldMap.field_at(x, y)` 仅在 source 粒子处返回缓存值，多余查询 raise；
- 工厂方法：`DcBeamSimulation.circular_green(radius=..., magnetic_field=..., config=...)`；
- 数值正确性：在 ring/spoke 节点上摆粒子时与 analytic mirror-charge 公式严格匹配（相对误差 ~1e-9）。

**与旧代码的差异**：旧 `t2dcore.py` 第 762 行的 Eu 公式存在 bug：`Eu = (ra²·ro·sind)/c1 - (ro·sind)/c2`，第一项的 `ra²` 因子让两项量纲不一致（m vs 1/m），数值上偏差最大可达 ~50%。新代码按正确推导实现 `Eu = (ro·sind)/c1 - (ro·sind)/c2`，因此**新旧圆管 Green 路径的逐点数值结果不会一致**，旧代码的圆管对比脚本不能再作为正确性参照。

### 旧 `t2d_cir_V1.py` 参数下的回归（仅参考）

`scripts/compare_t2d_cir_v1.py` 用 `t2d_cir_V1.py` 的参数 (`I=-15 mA, U=19 kV, Lz=30 mm, dt=1e-13 s, T=5000, pipe ra=150 μm, n_r=200, n_theta=360, Bz=0.6 T uniform, precision=float32, randomize_directions(2 mrad, rebuild_transverse, seed=2000)`) 同时跑：

- 旧 `t2dcore.RUGrid`（带 Eu bug，仅作参考）；
- 新 `CircularGreenSpaceChargeSolver` + `CircularBoundary`。

`Lz=30 mm` 在 `vz0 ≈ 7.96e7 m/s` 下截断到 ~3790 step（对应 379 个 stride=10 快照）。结果（`particles=2095`）：

| 量 | 新（正确公式） | 旧（bug Eu，参考） | max rel diff |
|---|---|---|---|
| `r_max` 终值 | 36.895 μm | 36.890 μm | 1.4e-4 |
| `r_mean` 终值 | 23.787 μm | 23.787 μm | 8.8e-6 |
| `wall_hit` 计数 | 0 | — | — |

包络一致性比预期好：本案例束流近轴对称、横向角动量 ~ mrad，Eu bug 主要影响 `Eθ`，对 `Er`-主导的径向包络贡献被平均掉。**这并不能反过来证明 Eu bug 在所有场景都可忽略**——非轴对称分布下偏差可能放大到 ~50%（见 ring/spoke 节点的逐点对比）。回归脚本的目的是确认新路径在 t2d 风格参数下能跑通且与旧代码同量级，不是把旧代码当 ground truth。

## 当前测试状态

当前测试覆盖：

- Boris 推进器；
- 电荷沉积；
- 矩形 FDM Poisson 求解；
- 初始分布采样；
- tracker smoke test；
- 正式 API + NPZ roundtrip；
- 小型数值回归；
- solver/tracker 解耦；
- `wall_hit` 标签生命周期（撞墙标记、冻结、初始 wall_hit 沿用、`live_mask` 单调）；
- 圆管 Green solver（边界 absorb、对称性、与 analytic mirror-charge 在 ring/spoke 节点上的严格匹配、wall_hit source 跳过、端到端 smoke）。

当前验证命令：

```bash
PYTHONPATH=src python -m compileall -q src tests scripts
MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=src pytest -q
```

最近一次结果：

```text
27 passed
```

## 仍未完成的旧功能边界

旧代码中尚未正式移植的功能包括：

- GPU 分支；
- `euler_bz`、`euler_bzbr`、`boris_bz` 等非当前主路径 pusher；
- 完整 `Trajectories3D` / `traj3dpost` 后处理能力；
- 正式可视化 API；
- CST / Opera / PID 导出相关工具；
- gyrocenter、轨迹筛选、复杂扫描脚本等分析工具。

圆管 Green / `RUGrid` 路径已移植为 `CircularGreenSpaceChargeSolver`（注意 Eu 公式与旧代码不同，见上节）。

## 近期计划

### 1. 固化解耦后的 API

目标：让后续新增 solver 时不需要再改 tracker 主循环。

已完成：

- 删除 `track_slice(grid=...)` legacy 兜底，tracker 不再 import 矩形实现；
- 删除 `TrackConfig.absorbing_boundary`，撞墙行为完全由 boundary 类型表达；
- `DcBeamSimulation` 的死字段 `grid` 已移除；
- `boundary.absorb(...)` 改为返回 hit mask，tracker 累加到 `wall_hit` 并写入 `SliceDistribution.extras`；
- `SliceTrackResult.live_mask()` 用于过滤撞墙后的"贴墙滑行"假轨迹。

待做：

- 给 `SpaceChargeSolver`、`FieldMap2DProtocol`、`TransverseBoundary` 补文档和示例；
- 增加一个自定义 solver 的最小 example。

### 2. 整理 examples

把临时脚本整理成正式 examples：

- `examples/rect_fdm_minimal.py`
- `examples/rect_fdm_t2d_rec_v4.py`
- `examples/compare_legacy_t2d_rec_v4.py`

当前 `scripts/` 可以继续保留为开发/验证工具。

### 3. 增强结果诊断

建议下一批加入：

- `SliceTrackResult.compare(other)`；
- `final_summary()`；
- `envelope_summary()`；
- 保存/加载更多 metadata，例如网格摘要、磁场摘要、solver 类型；
- 更方便的粒子轨迹抽样方法。

### 4. 圆管 Green solver 后续

第一版已完成（见上文"圆管 Green solver"小节）。后续可做：

- 用旧代码圆管路径生成对比脚本时，**只能用作宏观趋势对比**，不能逐点对齐（旧 Eu 公式有 bug）；
- 写一个独立的 analytic mirror-charge 对照脚本，作为长期 ground truth；
- 性能：当前 N² numba 实现单步 ~ms 量级，数千粒子可用；更大规模再考虑分块或 cupy；
- 给 `field_at` 增加 off-source 查询能力（当前会 raise），用于场可视化；
- 圆管路径的端到端 t2d_rec_v4 风格对比脚本（用 analytic 做参照）。

### 5. 可视化 API

在计算路径稳定后再做。优先顺序：

- 初始分布图；
- 网格图；
- 包络图；
- 轨迹图；
- 空间电荷场图。

## 风险与注意事项

- `partdist.SliceDistribution` 已收紧为固定 `z` 纵向；`dcbeamtrack` 当前已经适配这个接口。
- 旧 `t2d_rec_V4.py` 的有效步数不是 `T=5000`，而是按 `Lz/(vz0*dt)` 截断为 `3771`。
- 旧代码 `precision='float32'` 会影响逐粒子速度差异；对比时应明确 precision。
- 旧代码和新版默认物理常数略有差异；严格同常数对比时可以使用旧 `EPS0`。
- 当前 `TriGrid` 不是旧 `meshpy` 三角网格的完全复刻；严格旧代码对比脚本仍调用旧 `TriGrid` 生成初始粒子。
- 旧 `t2dcore.py` 的圆管 Green Eu 公式有 bug（多一个 `ra²` 因子，量纲不一致），新代码已按正确推导实现，圆管路径不能与旧代码逐点对齐。

## 下一步建议

圆管 Green solver 第一版已经就位，下一步可选方向：

- **examples 整理**：把 `scripts/` 里的临时对比脚本提炼为正式 `examples/`，包括圆管 Green 的最小例子；
- **结果诊断增强**：`SliceTrackResult.compare`、`final_summary`、`envelope_summary` 等；
- **圆管 Green 长期对照脚本**：用 analytic mirror-charge 做参照，固定回归基准；
- **性能**：当前 numba O(N²) 单步可承受千粒子级；大规模场景再考虑分块/GPU。
