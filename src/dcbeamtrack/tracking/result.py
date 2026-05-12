from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from partdist import SliceDistribution

from dcbeamtrack.config import TrackConfig


_NPZ_FORMAT = "dcbeamtrack.SliceTrackResult"
_NPZ_VERSION = 1


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


@dataclass(frozen=True)
class SliceTrackResult(Sequence[SliceDistribution]):
    snapshots: tuple[SliceDistribution, ...]
    config: object | None = None

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
        }
        arrays["metadata"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> "SliceTrackResult":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("format") != _NPZ_FORMAT:
                raise ValueError(f"Unsupported result file format: {metadata.get('format')!r}.")
            if int(metadata.get("version", -1)) != _NPZ_VERSION:
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
        return cls(tuple(snapshots), config=_decode_config(metadata.get("config", {})))

    save = save_npz
    load = load_npz
