"""
Slim on-disk registrar format for fast loading (metadata JSON + float32 displacement binaries).

Written alongside the VALIS pickle at registration time; preferred for downstream tile warping.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

SLIM_FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
META_FILENAME = "meta.json"
BK_DXDY_FILENAME = "bk_dxdy.bin"


def slim_dir_for_pickle(pickle_path: str) -> str:
    """Directory that holds the slim export for a given registrar pickle path."""
    directory, filename = os.path.split(os.path.abspath(pickle_path))
    if filename.endswith("_registrar.pickle"):
        slim_name = filename[: -len("_registrar.pickle")] + "_slim"
    else:
        slim_name = os.path.splitext(filename)[0] + "_slim"
    return os.path.join(directory, slim_name)


def manifest_path_for_pickle(pickle_path: str) -> str:
    return os.path.join(slim_dir_for_pickle(pickle_path), MANIFEST_FILENAME)


def is_slim_manifest(path: str) -> bool:
    path = os.path.abspath(path)
    return os.path.isfile(path) and os.path.basename(path) == MANIFEST_FILENAME


def resolve_registrar_path(path: str) -> str:
    """
    Prefer the slim manifest when it exists next to (or instead of) a pickle path.
    """
    path = os.path.abspath(path)
    if is_slim_manifest(path):
        return path
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, MANIFEST_FILENAME)):
        return os.path.join(path, MANIFEST_FILENAME)
    if path.endswith(".pickle") or path.endswith(".pkl"):
        manifest = manifest_path_for_pickle(path)
        if os.path.isfile(manifest):
            return manifest
    return path


def _normalize_bk_dxdy_array(bk_dxdy: Any) -> Optional[np.ndarray]:
    """Return bk_dxdy as float32 ndarray with shape (2, rows, cols), or None."""
    if bk_dxdy is None:
        return None
    try:
        if isinstance(bk_dxdy, (list, tuple)) and len(bk_dxdy) == 2:
            dx = np.asarray(bk_dxdy[0], dtype=np.float32)
            dy = np.asarray(bk_dxdy[1], dtype=np.float32)
            return np.stack([dx, dy], axis=0)

        if hasattr(bk_dxdy, "numpy"):
            arr = np.asarray(bk_dxdy.numpy(), dtype=np.float32)
        else:
            try:
                import pyvips
                from valis import warp_tools

                if isinstance(bk_dxdy, pyvips.Image):
                    arr = np.asarray(warp_tools.vips2numpy(bk_dxdy), dtype=np.float32)
                else:
                    arr = np.asarray(bk_dxdy, dtype=np.float32)
            except ImportError:
                arr = np.asarray(bk_dxdy, dtype=np.float32)

        if arr.ndim != 3:
            return None
        if arr.shape[0] == 2:
            return np.ascontiguousarray(arr, dtype=np.float32)
        if arr.shape[2] == 2:
            return np.ascontiguousarray(
                np.stack([arr[..., 0], arr[..., 1]], axis=0), dtype=np.float32
            )
        return None
    except Exception:
        return None


def _compute_overlap_bbox_at_level(
    slides: List[Any],
    zoom_level: int,
    non_rigid: bool = True,
) -> Optional[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = None, None, None, None
    for slide in slides:
        src_w, src_h = slide.slide_dimensions_wh[zoom_level]
        src_corners_xy = np.array(
            [
                [0.0, 0.0],
                [float(src_w), 0.0],
                [float(src_w), float(src_h)],
                [0.0, float(src_h)],
            ],
            dtype=np.float32,
        )
        reg_corners_xy = slide.warp_xy(
            src_corners_xy,
            slide_level=zoom_level,
            pt_level=zoom_level,
            non_rigid=non_rigid,
            crop=False,
        )
        if not np.any(np.isfinite(reg_corners_xy)):
            continue
        if np.any(~np.isfinite(reg_corners_xy)):
            finite = np.isfinite(reg_corners_xy)
            reg_corners_xy = np.where(finite, reg_corners_xy, np.nan)
        bx0 = int(np.floor(np.nanmin(reg_corners_xy[:, 0])))
        by0 = int(np.floor(np.nanmin(reg_corners_xy[:, 1])))
        bx1 = int(np.ceil(np.nanmax(reg_corners_xy[:, 0])))
        by1 = int(np.ceil(np.nanmax(reg_corners_xy[:, 1])))
        if x0 is None:
            x0, y0, x1, y1 = bx0, by0, bx1, by1
        else:
            x0 = max(x0, bx0)
            y0 = max(y0, by0)
            x1 = min(x1, bx1)
            y1 = min(y1, by1)
    if x0 is None or x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _compute_overlap_bboxes(valis: Any, non_rigid: bool = True) -> Dict[str, Optional[List[int]]]:
    slides = list(valis.slide_dict.values())
    if not slides:
        return {}
    max_level = min(max(0, len(s.slide_dimensions_wh) - 1) for s in slides)
    out: Dict[str, Optional[List[int]]] = {}
    for level in range(max_level + 1):
        bbox = _compute_overlap_bbox_at_level(slides, level, non_rigid=non_rigid)
        out[str(level)] = list(bbox) if bbox is not None else None
    return out


def _aligned_shapes_by_level(valis: Any) -> Dict[str, List[int]]:
    ref = valis.get_ref_slide()
    max_level = max(0, len(ref.slide_dimensions_wh) - 1)
    shapes: Dict[str, List[int]] = {}
    for level in range(max_level + 1):
        shape_rc = valis.get_aligned_slide_shape(level)
        shapes[str(level)] = [int(shape_rc[0]), int(shape_rc[1])]
    return shapes


def export_slim_registrar(valis: Any, pickle_path: str) -> str:
    """
    Write slim registrar next to the pickle file. Returns path to manifest.json.
    """
    pickle_path = os.path.abspath(pickle_path)
    slim_dir = slim_dir_for_pickle(pickle_path)
    slides_dir = os.path.join(slim_dir, "slides")
    pathlib.Path(slides_dir).mkdir(parents=True, exist_ok=True)

    ref_slide = valis.get_ref_slide()
    slide_names: List[str] = []
    for slide_name, slide in valis.slide_dict.items():
        slide_names.append(slide_name)
        slide_dir = os.path.join(slides_dir, slide_name)
        pathlib.Path(slide_dir).mkdir(parents=True, exist_ok=True)

        M = slide.M
        if M is not None:
            M_list = np.asarray(M, dtype=np.float64).reshape(3, 3).tolist()
        else:
            M_list = None

        dims_wh = np.asarray(slide.slide_dimensions_wh).tolist()
        meta = {
            "name": slide_name,
            "src_f": str(slide.src_f),
            "M": M_list,
            "processed_img_shape_rc": list(slide.processed_img_shape_rc),
            "reg_img_shape_rc": list(slide.reg_img_shape_rc),
            "slide_dimensions_wh": dims_wh,
        }

        bk = _normalize_bk_dxdy_array(getattr(slide, "bk_dxdy", None))
        if bk is not None:
            meta["bk_dxdy_shape"] = list(bk.shape)
            meta["bk_dxdy_dtype"] = "float32"
            bk.tofile(os.path.join(slide_dir, BK_DXDY_FILENAME))
        else:
            meta["bk_dxdy_shape"] = None

        with open(os.path.join(slide_dir, META_FILENAME), "w", encoding="utf-8") as f:
            json.dump(meta, f, separators=(",", ":"))

    overlap = _compute_overlap_bboxes(valis, non_rigid=True)
    aligned_shapes = _aligned_shapes_by_level(valis)

    name_dict = {str(k): str(v) for k, v in valis.name_dict.items()}
    manifest = {
        "version": SLIM_FORMAT_VERSION,
        "pickle_path": pickle_path,
        "name": valis.name,
        "data_dir": getattr(valis, "data_dir", None),
        "dst_dir": getattr(valis, "dst_dir", None),
        "name_dict": name_dict,
        "reference_img_f": str(valis.reference_img_f),
        "reference_slide_name": ref_slide.name,
        "aligned_shape_rc_by_level": aligned_shapes,
        "overlap_bbox_xyxy_by_level": overlap,
        "slide_names": slide_names,
    }
    manifest_path = os.path.join(slim_dir, MANIFEST_FILENAME)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, separators=(",", ":"))

    return manifest_path


def save_registrar_artifacts(valis: Any) -> str:
    """
    Pickle the Valis object and write the slim export. Sets ``valis.reg_f`` to the manifest path.
    """
    import pickle

    pathlib.Path(valis.data_dir).mkdir(exist_ok=True, parents=True)
    pickle_path = os.path.join(valis.data_dir, valis.name + "_registrar.pickle")
    with open(pickle_path, "wb") as f:
        pickle.dump(valis, f)
    manifest_path = export_slim_registrar(valis, pickle_path)
    valis.reg_f = manifest_path
    return manifest_path


class SlimSlide:
    """Minimal slide record for backward mapping and tile warping."""

    def __init__(
        self,
        registrar: "SlimRegistrar",
        name: str,
        src_f: str,
        M: Optional[np.ndarray],
        processed_img_shape_rc: Tuple[int, int],
        reg_img_shape_rc: Tuple[int, int],
        slide_dimensions_wh: np.ndarray,
        bk_dxdy: Optional[np.ndarray],
    ):
        self.val_obj = registrar
        self.name = name
        self.src_f = src_f
        self.M = M
        self.processed_img_shape_rc = tuple(processed_img_shape_rc)
        self.reg_img_shape_rc = tuple(reg_img_shape_rc)
        self.slide_dimensions_wh = slide_dimensions_wh
        self.bk_dxdy = bk_dxdy
        self.aligned_slide_shape_rc = None

    @property
    def stack_idx(self) -> int:
        return self.val_obj.slide_names.index(self.name)


class SlimRegistrar:
    """Lightweight registrar loaded from slim export on disk."""

    def __init__(self, manifest_path: str, manifest: Dict[str, Any]):
        self.reg_f = manifest_path
        self.manifest_path = manifest_path
        self.slim_dir = os.path.dirname(manifest_path)
        self.name = manifest["name"]
        self.data_dir = manifest.get("data_dir")
        self.dst_dir = manifest.get("dst_dir")
        self.pickle_path = manifest.get("pickle_path")
        self.name_dict = manifest["name_dict"]
        self.reference_img_f = manifest["reference_img_f"]
        self.reference_slide_name = manifest["reference_slide_name"]
        self.slide_names = list(manifest["slide_names"])
        self.aligned_shape_rc_by_level = {
            int(k): tuple(v) for k, v in manifest["aligned_shape_rc_by_level"].items()
        }
        overlap = manifest.get("overlap_bbox_xyxy_by_level") or {}
        self.overlap_bbox_xyxy_by_level = {
            int(k): (tuple(v) if v is not None else None) for k, v in overlap.items()
        }
        self.slide_dict: Dict[str, SlimSlide] = {}
        self.error_df = None
        self.summary_df = None

    def get_slide(self, src_f: str) -> Optional[SlimSlide]:
        from valis import valtils

        default_name = valtils.get_name(src_f)
        if default_name in self.slide_dict:
            return self.slide_dict[default_name]
        assigned = self.name_dict.get(src_f)
        if assigned and assigned in self.slide_dict:
            return self.slide_dict[assigned]
        if src_f in self.slide_dict:
            return self.slide_dict[src_f]
        return None

    def get_ref_slide(self) -> SlimSlide:
        slide = self.slide_dict.get(self.reference_slide_name)
        if slide is None:
            raise KeyError(f"Reference slide {self.reference_slide_name!r} not in slide_dict")
        return slide

    def get_aligned_slide_shape(self, level: Union[int, float]) -> np.ndarray:
        if int(level) in self.aligned_shape_rc_by_level:
            return np.array(self.aligned_shape_rc_by_level[int(level)], dtype=int)
        ref_slide = self.get_ref_slide()
        if np.issubdtype(type(level), np.integer):
            n_levels = len(ref_slide.slide_dimensions_wh)
            lvl = int(level)
            if lvl >= n_levels:
                lvl = n_levels - 1
            slide_shape_rc = ref_slide.slide_dimensions_wh[lvl][::-1]
            s_rc = slide_shape_rc / np.array(ref_slide.processed_img_shape_rc)
        else:
            s_rc = level
        return np.ceil(np.array(ref_slide.reg_img_shape_rc) * s_rc).astype(int)


def load_slim_registrar(manifest_path: str) -> SlimRegistrar:
    manifest_path = os.path.abspath(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("version") != SLIM_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported slim registrar version {manifest.get('version')!r}; "
            f"expected {SLIM_FORMAT_VERSION}"
        )

    registrar = SlimRegistrar(manifest_path, manifest)
    slides_root = os.path.join(registrar.slim_dir, "slides")
    for slide_name in registrar.slide_names:
        slide_dir = os.path.join(slides_root, slide_name)
        with open(os.path.join(slide_dir, META_FILENAME), encoding="utf-8") as f:
            meta = json.load(f)
        bk = None
        shape = meta.get("bk_dxdy_shape")
        bk_path = os.path.join(slide_dir, BK_DXDY_FILENAME)
        if shape is not None and os.path.isfile(bk_path):
            bk = np.fromfile(bk_path, dtype=np.float32).reshape(tuple(shape))
        M = np.asarray(meta["M"], dtype=np.float64) if meta.get("M") is not None else None
        slide = SlimSlide(
            registrar=registrar,
            name=slide_name,
            src_f=meta["src_f"],
            M=M,
            processed_img_shape_rc=tuple(meta["processed_img_shape_rc"]),
            reg_img_shape_rc=tuple(meta["reg_img_shape_rc"]),
            slide_dimensions_wh=np.asarray(meta["slide_dimensions_wh"]),
            bk_dxdy=bk,
        )
        registrar.slide_dict[slide_name] = slide

    return registrar


def load_registrar(path: str) -> Any:
    """Load slim registrar if available, otherwise fall back to pickle."""
    resolved = resolve_registrar_path(path)
    if is_slim_manifest(resolved) or (
        os.path.isdir(resolved) and os.path.isfile(os.path.join(resolved, MANIFEST_FILENAME))
    ):
        return load_slim_registrar(
            resolved if is_slim_manifest(resolved) else os.path.join(resolved, MANIFEST_FILENAME)
        )
    from valis import registration

    return registration.load_registrar(resolved)
