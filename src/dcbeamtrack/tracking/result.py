from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from partdist import SliceDistribution

from dcbeamtrack.config import TrackConfig


_NPZ_FORMAT = "dcbeamtrack.SliceTrackResult"
_NPZ_VERSION = 2
_NPZ_MIN_READABLE_VERSION = 1


def _encode_config(config: object | None) -> dict:
    if config is None:
        return {"type": None, "data": None}
    if isinstance(config, TrackConfig):
        return {"type": "TrackConfig", "data": asdict(config)}
    return {"type": config.__class__.__name__, "data": None}


def _decode_config(payload: dict) -> object | None:
    if payload.get("type") == "TrackConfig" and payload.get("data") is not None:
        return TrackConfig(**payload["data"])
    return None


def _safe_summary(obj) -> dict:
    """Call ``obj.summary()`` if defined, else fall back to ``{'type': classname}``.

    Lets third-party solvers/fields work without forcing a Protocol-level
    summary contract.
    """
    fn = getattr(obj, "summary", None)
    if callable(fn):
        return fn()
    return {"type": type(obj).__name__}


def _config_summary(config) -> dict:
    """Encode TrackConfig (or any config object) as a flat dict suitable for
    inclusion in result.metadata. Distinct from the existing _encode_config
    (which produces the npz envelope's ``config`` slot)."""
    if config is None:
        return {"type": None}
    if isinstance(config, TrackConfig):
        return {"type": "TrackConfig", **asdict(config)}
    return {"type": type(config).__name__}


def _masked_weighted_envelope(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    live: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (r_max, r_mean, r_rms) per snapshot on live particles only.

    Inputs:
      x, y: (n_snap, n_part) coordinates
      weights: (n_part,) per-particle absolute weight
      live: (n_snap, n_part) bool mask, True = particle alive in that snapshot

    Returns three (n_snap,) arrays. Snapshots with zero live weight produce nan
    in all three outputs (caller decides how to react).
    """
    r = np.hypot(x, y)
    r2 = x * x + y * y
    w_row = weights[None, :]
    w_live = np.where(live, w_row, 0.0)
    wsum = w_live.sum(axis=1)
    valid = wsum > 0

    r_max = np.full_like(wsum, np.nan, dtype=float)
    r_mean = np.full_like(wsum, np.nan, dtype=float)
    r_rms = np.full_like(wsum, np.nan, dtype=float)
    if valid.any():
        r_masked = np.where(live, r, np.nan)
        with np.errstate(invalid="ignore"):
            r_max_all = np.nanmax(r_masked, axis=1)
        r_max[valid] = r_max_all[valid]

        num_mean = np.where(live, r * w_row, 0.0).sum(axis=1)
        num_rms2 = np.where(live, r2 * w_row, 0.0).sum(axis=1)
        r_mean[valid] = num_mean[valid] / wsum[valid]
        r_rms[valid] = np.sqrt(num_rms2[valid] / wsum[valid])
    return r_max, r_mean, r_rms


def _diff_metrics(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-30) -> tuple[float, float, float]:
    """Compute (max_abs_diff, max_rel_diff, final_diff) between two 1-D arrays.

    max_rel_diff uses denom = max(|a|, |b|, eps) element-wise.
    final_diff = a[-1] - b[-1] (signed). Uses np.nanmax so nan entries do not
    poison the result. If every element is nan, max_abs and max_rel are nan.
    """
    abs_diff = np.abs(a - b)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), eps)
    rel_diff = abs_diff / denom
    with np.errstate(invalid="ignore"):
        max_abs = float(np.nanmax(abs_diff)) if not np.all(np.isnan(abs_diff)) else float("nan")
        max_rel = float(np.nanmax(rel_diff)) if not np.all(np.isnan(rel_diff)) else float("nan")
    final_diff = float(a[-1] - b[-1])
    return max_abs, max_rel, final_diff


@dataclass(frozen=True)
class SliceTrackResult(Sequence[SliceDistribution]):
    snapshots: tuple[SliceDistribution, ...]
    config: object | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        snapshots = tuple(self.snapshots)
        if not snapshots:
            raise ValueError("SliceTrackResult requires at least one snapshot.")
        first = snapshots[0]
        n = len(first)
        for snapshot in snapshots:
            if len(snapshot) != n:
                raise ValueError("All snapshots must have the same particle count.")
        z = np.asarray([snapshot.z for snapshot in snapshots], dtype=float)
        if z.size > 1 and np.any(np.diff(z) < 0.0):
            raise ValueError("Snapshot z positions must be non-decreasing.")
        object.__setattr__(self, "snapshots", snapshots)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, index):
        return self.snapshots[index]

    def __iter__(self) -> Iterator[SliceDistribution]:
        return iter(self.snapshots)

    def __repr__(self) -> str:
        z = self.z
        return (
            f"SliceTrackResult(n_snapshots={self.n_snapshots}, "
            f"n_particles={self.n_particles}, "
            f"z=[{z[0]:.4g}, {z[-1]:.4g}])"
        )

    @property
    def n_snapshots(self) -> int:
        return len(self.snapshots)

    @property
    def n_particles(self) -> int:
        return len(self.snapshots[0])

    @property
    def z(self) -> np.ndarray:
        return np.asarray([snapshot.z for snapshot in self.snapshots], dtype=float)

    @property
    def initial(self) -> SliceDistribution:
        return self.snapshots[0]

    @property
    def final(self) -> SliceDistribution:
        return self.snapshots[-1]

    def get_snapshot(self, index: int) -> SliceDistribution:
        return self.snapshots[index]

    def array(self, key: str) -> np.ndarray:
        if key == "z":
            return np.broadcast_to(self.z[:, None], (self.n_snapshots, self.n_particles)).copy()
        return np.vstack([np.asarray(snapshot.get_data(key), dtype=float) for snapshot in self.snapshots])

    def live_mask(self) -> np.ndarray:
        """(n_snapshots, n_particles) bool array, True where the particle has not hit a wall yet."""
        if "wall_hit" not in self.initial.extra_quantity_keys:
            return np.ones((self.n_snapshots, self.n_particles), dtype=bool)
        return self.array("wall_hit") <= 0.5

    def trajectories(
        self,
        mask: np.ndarray | None = None,
        *,
        keys: tuple[str, ...] = ("x", "y", "px", "py", "pz"),
    ) -> "ParticleTrajectories":
        """Extract per-snapshot data for the particles selected by ``mask``.

        ``mask`` must be a bool ndarray of shape ``(n_particles,)``, or ``None``
        (default) to select every particle. ``keys`` is normalized to a tuple;
        each key is resolved via :meth:`array`, so the core kinematic keys
        (``x``, ``y``, ``px``, ``py``, ``pz``, ``t``, ``lam``, ``z``) and any
        extra quantity carried by the snapshots (e.g. ``wall_hit``) are
        accepted. Returns a :class:`ParticleTrajectories` whose ``data[k]`` has
        shape ``(n_snapshots, k_selected)``, with columns in original
        particle-index order.
        """
        if mask is None:
            mask_arr = np.ones(self.n_particles, dtype=bool)
        else:
            mask_arr = np.asarray(mask)
            if mask_arr.dtype != bool:
                raise TypeError(
                    f"trajectories: mask must be bool dtype, got {mask_arr.dtype}."
                )
            if mask_arr.shape != (self.n_particles,):
                raise ValueError(
                    f"trajectories: mask shape {mask_arr.shape} does not match "
                    f"({self.n_particles},)."
                )
        keys_tuple = tuple(keys)
        if len(keys_tuple) == 0:
            raise ValueError("trajectories: keys must be non-empty.")
        extras = set(self.initial.extra_quantity_keys)
        core = {"x", "y", "px", "py", "pz", "t", "lam", "z"}
        for k in keys_tuple:
            if k not in core and k not in extras:
                raise ValueError(f"trajectories: unknown key: {k!r}")

        data = {k: self.array(k)[:, mask_arr] for k in keys_tuple}
        live = self.live_mask()[:, mask_arr]
        return ParticleTrajectories(
            z=self.z,
            data=data,
            live=live,
            mask=mask_arr.copy(),
            keys=keys_tuple,
        )

    def random_mask(self, n: int, *, seed: int | None = None) -> np.ndarray:
        """Bool mask of length n_particles with n uniformly-random True entries.

        Same ``seed`` produces the same mask. Raises ``ValueError`` if n is out
        of range [1, n_particles].
        """
        n_int = int(n)
        if n_int < 1 or n_int > self.n_particles:
            raise ValueError(
                f"random_mask: n must be in [1, {self.n_particles}], got {n_int}."
            )
        rng = np.random.default_rng(seed)
        idx = rng.choice(self.n_particles, size=n_int, replace=False)
        out = np.zeros(self.n_particles, dtype=bool)
        out[idx] = True
        return out

    def evenly_spaced_mask(self, n: int) -> np.ndarray:
        """Bool mask of length n_particles with n evenly-spaced True entries.

        Indices are ``np.linspace(0, n_particles - 1, n).astype(int)``. For
        ``n <= n_particles`` the resulting indices are distinct. Raises
        ``ValueError`` if n is out of range [1, n_particles].
        """
        n_int = int(n)
        if n_int < 1 or n_int > self.n_particles:
            raise ValueError(
                f"evenly_spaced_mask: n must be in [1, {self.n_particles}], got {n_int}."
            )
        idx = np.linspace(0, self.n_particles - 1, n_int).astype(int)
        out = np.zeros(self.n_particles, dtype=bool)
        out[idx] = True
        return out

    def radial_position(self) -> np.ndarray:
        x = self.array("x")
        y = self.array("y")
        return np.sqrt(x * x + y * y)

    def r_max(self) -> np.ndarray:
        return np.max(self.radial_position(), axis=1)

    def r_mean(self) -> np.ndarray:
        r = self.radial_position()
        w = self.weights(absolute=True)
        wsum = float(np.sum(w))
        if wsum == 0.0:
            raise ValueError("Cannot compute weighted mean radius because sum(abs(lam)) is zero.")
        return np.sum(r * w[None, :], axis=1) / wsum

    def r_rms(self) -> np.ndarray:
        x = self.array("x")
        y = self.array("y")
        w = self.weights(absolute=True)
        wsum = float(np.sum(w))
        if wsum == 0.0:
            raise ValueError("Cannot compute weighted rms radius because sum(abs(lam)) is zero.")
        return np.sqrt(np.sum((x * x + y * y) * w[None, :], axis=1) / wsum)

    def weights(self, *, absolute: bool = True) -> np.ndarray:
        lam = np.asarray(self.initial.lam, dtype=float)
        return np.abs(lam) if absolute else lam.copy()

    def radial_envelope(self, kind: str = "rms") -> tuple[np.ndarray, np.ndarray]:
        kind = kind.lower()
        if kind not in {"mean", "rms", "max"}:
            raise ValueError("kind must be 'mean', 'rms', or 'max'.")
        x = self.array("x")
        y = self.array("y")
        r = np.sqrt(x * x + y * y)
        if kind == "max":
            return self.z, np.max(r, axis=1)

        w = self.weights(absolute=True)
        wsum = float(np.sum(w))
        if wsum == 0.0:
            raise ValueError("Cannot compute weighted envelope because sum(abs(lam)) is zero.")
        if kind == "mean":
            env = self.r_mean()
        else:
            env = self.r_rms()
        return self.z, env

    def envelope_summary(self) -> dict:
        """Scalar summary of the full-trajectory envelopes on live particles.

        Returns a flat dict with keys: r_{max,mean,rms}_{peak,peak_z,final},
        waist_z, waist_r_rms. Raises ValueError if no snapshot has any live
        particles. Particles flagged via wall_hit are excluded via live_mask().

        Note: ``r_{max,mean,rms}_final`` keys also appear in :meth:`final_summary`.
        Merging both dicts under the same prefix lets final_summary's values win;
        the two paths agree numerically when wall_hit is cumulative (the standard
        case), so the overwrite is harmless. Use distinct prefixes if you want
        both copies to coexist in one flat dict.
        """
        x = self.array("x")
        y = self.array("y")
        w = self.weights(absolute=True)
        live = self.live_mask()
        r_max_arr, r_mean_arr, r_rms_arr = _masked_weighted_envelope(x, y, w, live)

        if np.isnan(r_rms_arr).all():
            raise ValueError("envelope_summary: no live particles in any snapshot")

        z = self.z
        out: dict = {}
        for kind, arr in (("r_max", r_max_arr), ("r_mean", r_mean_arr), ("r_rms", r_rms_arr)):
            idx = int(np.nanargmax(arr))
            out[f"{kind}_peak"] = float(arr[idx])
            out[f"{kind}_peak_z"] = float(z[idx])
            out[f"{kind}_final"] = float(arr[-1])

        widx = int(np.nanargmin(r_rms_arr))
        out["waist_z"] = float(z[widx])
        out["waist_r_rms"] = float(r_rms_arr[widx])
        return out

    def final_summary(self) -> dict:
        """Final-snapshot scalars on live particles.

        Returns flat dict with: z_final, n_particles, n_live_final,
        n_wall_hit_final (0 if no wall_hit extra), and r_{max,mean,rms}_final
        on the live subset. r_*_final is nan when n_live_final == 0 (no raise:
        terminal full absorption is a legitimate physical outcome).

        Note: the three ``r_*_final`` keys overlap with :meth:`envelope_summary`;
        see that docstring for the merging convention.
        """
        final = self.snapshots[-1]
        if "wall_hit" in final.extra_quantity_keys:
            wh = np.asarray(final.get_data("wall_hit"), dtype=float)
        else:
            wh = np.zeros(self.n_particles, dtype=float)
        live = wh <= 0.5
        n_live = int(live.sum())
        n_wh = int((wh > 0.5).sum())

        out: dict = {
            "z_final": float(final.z),
            "n_particles": int(self.n_particles),
            "n_live_final": n_live,
            "n_wall_hit_final": n_wh,
        }
        if n_live == 0:
            out["r_max_final"] = float("nan")
            out["r_mean_final"] = float("nan")
            out["r_rms_final"] = float("nan")
            return out

        x = np.asarray(final.x, dtype=float)[live]
        y = np.asarray(final.y, dtype=float)[live]
        w = self.weights(absolute=True)[live]
        wsum = float(w.sum())
        r2 = x * x + y * y
        r = np.sqrt(r2)
        out["r_max_final"] = float(np.max(r))
        out["r_mean_final"] = float(np.sum(r * w) / wsum)
        out["r_rms_final"] = float(np.sqrt(np.sum(r2 * w) / wsum))
        return out

    def compare(self, other: "SliceTrackResult") -> dict:
        """Pairwise envelope diff against another result on the same z grid.

        Strict validation: raises ValueError if snapshot count, particle count, or
        z arrays differ. Envelope diffs are computed on the intersection live mask
        (particles alive on both sides at each snapshot), using self.weights() for
        both sides so the comparison is consistent.

        Returns flat dict with 9 envelope keys plus n_wall_hit_diff (int, 0 if
        either side lacks the wall_hit extra).
        """
        if len(self) != len(other):
            raise ValueError(
                f"compare: snapshot count differs ({len(self)} vs {len(other)})."
            )
        if self.n_particles != other.n_particles:
            raise ValueError(
                f"compare: particle count differs ({self.n_particles} vs {other.n_particles})."
            )
        if not np.allclose(self.z, other.z, rtol=1e-9, atol=1e-12):
            raise ValueError("compare: z arrays do not match within tolerance.")

        x_s = self.array("x")
        y_s = self.array("y")
        x_o = other.array("x")
        y_o = other.array("y")
        w_s = self.weights(absolute=True)
        live_both = self.live_mask() & other.live_mask()

        env_s = _masked_weighted_envelope(x_s, y_s, w_s, live_both)
        env_o = _masked_weighted_envelope(x_o, y_o, w_s, live_both)

        out: dict = {}
        for kind, a, b in zip(("r_max", "r_mean", "r_rms"), env_s, env_o):
            ma, mr, fd = _diff_metrics(a, b)
            out[f"{kind}_max_abs_diff"] = ma
            out[f"{kind}_max_rel_diff"] = mr
            out[f"{kind}_final_diff"] = fd

        s_has_wh = "wall_hit" in self.snapshots[-1].extra_quantity_keys
        o_has_wh = "wall_hit" in other.snapshots[-1].extra_quantity_keys
        if s_has_wh and o_has_wh:
            s_wh = int(np.asarray(self.snapshots[-1].get_data("wall_hit"), dtype=bool).sum())
            o_wh = int(np.asarray(other.snapshots[-1].get_data("wall_hit"), dtype=bool).sum())
            out["n_wall_hit_diff"] = s_wh - o_wh
        else:
            out["n_wall_hit_diff"] = 0
        return out

    def waist(self, kind: str = "rms") -> tuple[float, float]:
        z, envelope = self.radial_envelope(kind=kind)
        idx = int(np.nanargmin(envelope))
        return float(z[idx]), float(envelope[idx])

    def get_at_z(self, z_value: float) -> SliceDistribution:
        z = self.z
        z_value = float(z_value)
        if z_value < z[0] or z_value > z[-1]:
            raise ValueError("z_value is outside the stored snapshot range.")
        if z.size == 1:
            if not np.isclose(z_value, z[0]):
                raise ValueError("Only one snapshot is stored; z_value must match it.")
            return self.initial.copy()

        def interp(key: str) -> np.ndarray:
            values = self.array(key)
            return np.asarray([np.interp(z_value, z, values[:, i]) for i in range(values.shape[1])])

        template = self.initial
        extras = {key: template.get_quantity(key).copy() for key in template.extra_quantity_keys}
        return SliceDistribution(
            z=z_value,
            x=interp("x"),
            y=interp("y"),
            px=interp("px"),
            py=interp("py"),
            pz=interp("pz"),
            t=interp("t"),
            lam=template.lam.copy(),
            extras=extras or None,
        )

    def save_npz(self, path: str | Path, *, include_extras: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {
            "z": self.z,
            "x": self.array("x"),
            "y": self.array("y"),
            "px": self.array("px"),
            "py": self.array("py"),
            "pz": self.array("pz"),
            "t": self.array("t"),
            "lam": self.array("lam"),
        }

        extra_keys: tuple[str, ...] = ()
        extra_array_names: dict[str, str] = {}
        if include_extras:
            extra_keys = self.initial.extra_quantity_keys
            for snapshot in self.snapshots:
                if snapshot.extra_quantity_keys != extra_keys:
                    raise ValueError("All snapshots must have identical extra quantities to save them.")
            for idx, key in enumerate(extra_keys):
                array_name = f"extra_{idx}"
                arrays[array_name] = self.array(key)
                extra_array_names[key] = array_name

        metadata = {
            "format": _NPZ_FORMAT,
            "version": _NPZ_VERSION,
            "longitudinal_axis": "z",
            "config": _encode_config(self.config),
            "extra_keys": list(extra_keys),
            "extra_array_names": extra_array_names,
            "result_metadata": self.metadata,
        }
        try:
            metadata_json = json.dumps(metadata)
        except TypeError as exc:
            raise ValueError(
                f"result.metadata is not JSON-serializable: {exc}"
            ) from exc
        arrays["metadata"] = np.asarray(metadata_json)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> "SliceTrackResult":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("format") != _NPZ_FORMAT:
                raise ValueError(f"Unsupported result file format: {metadata.get('format')!r}.")
            version = int(metadata.get("version", -1))
            if version < _NPZ_MIN_READABLE_VERSION or version > _NPZ_VERSION:
                raise ValueError(f"Unsupported result file version: {metadata.get('version')!r}.")
            if metadata.get("longitudinal_axis", metadata.get("fixed_axis")) != "z":
                raise ValueError("Only z-longitudinal result files are supported.")

            z = np.asarray(data["z"], dtype=float)
            x = np.asarray(data["x"], dtype=float)
            y = np.asarray(data["y"], dtype=float)
            px = np.asarray(data["px"], dtype=float)
            py = np.asarray(data["py"], dtype=float)
            pz = np.asarray(data["pz"], dtype=float)
            t = np.asarray(data["t"], dtype=float)
            lam = np.asarray(data["lam"], dtype=float)
            extra_array_names = dict(metadata.get("extra_array_names", {}))
            extra_arrays = {
                key: np.asarray(data[array_name], dtype=float)
                for key, array_name in extra_array_names.items()
            }

        snapshots = []
        for idx, z_value in enumerate(z):
            extras = {key: values[idx].copy() for key, values in extra_arrays.items()}
            snapshots.append(
                SliceDistribution(
                    z=float(z_value),
                    x=x[idx],
                    y=y[idx],
                    px=px[idx],
                    py=py[idx],
                    pz=pz[idx],
                    t=t[idx],
                    lam=lam[idx],
                    extras=extras or None,
                )
            )
        result_metadata = metadata.get("result_metadata", {})
        return cls(
            tuple(snapshots),
            config=_decode_config(metadata.get("config", {})),
            metadata=result_metadata,
        )

    save = save_npz
    load = load_npz


@dataclass(frozen=True)
class ParticleTrajectories:
    """Particle-major view of a SliceTrackResult selection.

    Returned by :meth:`SliceTrackResult.trajectories`. Holds raw per-snapshot
    data for the selected particles, plus the (n_snap, k) live mask so callers
    can opt into nan-fill via :meth:`where_live`.
    """

    z: np.ndarray
    data: dict[str, np.ndarray]
    live: np.ndarray
    mask: np.ndarray
    keys: tuple[str, ...]

    def where_live(self, value: float = np.nan) -> "ParticleTrajectories":
        """Return a new ParticleTrajectories with dead positions filled by ``value``.

        All 2D arrays in ``data`` are replaced by ``np.where(self.live, arr, value)``.
        ``z``, ``live``, ``mask``, and ``keys`` are preserved. ``data`` insertion
        order matches ``self.keys``.
        """
        new_data = {k: np.where(self.live, self.data[k], value) for k in self.keys}
        return ParticleTrajectories(
            z=self.z,
            data=new_data,
            live=self.live,
            mask=self.mask,
            keys=self.keys,
        )
