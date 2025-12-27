#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect visualisation images produced by test_tpac.py for all experiments
under a given case/tag prefix.

用法示例：
    python collect_test_vis.py \
        --case ICA_norm \
        --tag 1205 \
        --out /mnt/e/.../paper_figs

可选：
    --root  指定工程根目录（包含 Rotate/ 的那个目录）
    --samples 10 42 99  只收集某些样本 id（从文件名里解析）
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence


# ---------------------- 路径辅助 ----------------------

def _default_root() -> Path:
    """
    默认 root：脚本所在目录的父目录/Rotate，如果不存在就用脚本所在目录。
    和 wo.py / test.sh 的风格保持一致。
    """
    script_dir = Path(__file__).resolve().parent
    rotate_dir = script_dir / "Rotate"
    if rotate_dir.is_dir():
        return script_dir
    return script_dir


def discover_experiments(root: Path, case: str, tag_prefix: str) -> List[str]:
    """
    在 results/temp_results/<case> 下面找所有以 tag_prefix_ 开头的实验目录名。
    例如 tag_prefix=1205，则匹配 1205_01_xxx, 1205_02_xxx, ...
    """
    base = root / "results" / "temp_results" / case
    if not base.is_dir():
        return []
    names: List[str] = []
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith(f"{tag_prefix}_"):
            names.append(d.name)
    names.sort()
    return names


def find_vis_dir(root: Path, case: str, exp_tag: str) -> Optional[Path]:
    """
    为每个 exp_tag 尝试若干个候选可视化目录，找到第一个存在的。
    如果你 test_tpac 的保存路径不同，可以在这里补一行即可。
    """
    base = root / "results" / "temp_results" / case / exp_tag
    candidates = [
        base / "samples_vis",
        base / "test_vis",
        base / "visualizations",
        base / "plots",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return None


# ---------------------- 文件名解析：样本 id ----------------------

_SAMPLE_PATTERNS = [
    re.compile(r"(?:sample[_\-]?|s_)(\d+)", re.IGNORECASE),
    re.compile(r"(?:id[_\-]?|case[_\-]?)(\d+)", re.IGNORECASE),
]


def extract_sample_id(name: str) -> Optional[int]:
    """
    尝试从文件名里解析样本 id，例如：
      - sample_0010_xxx.png -> 10
      - s_42_pred.png       -> 42
      - case15_pred.png     -> 15
    解析不到就返回 None。
    """
    for pat in _SAMPLE_PATTERNS:
        m = pat.search(name)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


# ---------------------- 主逻辑 ----------------------

def collect_vis(
    root: Path,
    case: str,
    tag_prefix: str,
    out_root: Path,
    sample_filter: Optional[Sequence[int]] = None,
) -> None:
    """
    - 遍历所有 exp_tag
    - 在各自的 vis 目录下找 png/jpg
    - 按 sample_id / exp_tag 重新组织到 out_root 下
    """
    exp_tags = discover_experiments(root, case, tag_prefix)
    if not exp_tags:
        raise SystemExit(
            f"[ERR] No experiments found under {root / 'results' / 'temp_results' / case} "
            f"with prefix {tag_prefix}_"
        )

    out_root.mkdir(parents=True, exist_ok=True)

    # 把 sample_filter 转成 set 方便判断
    sample_set = set(int(s) for s in sample_filter) if sample_filter else None

    print(f"[INFO] root     = {root}")
    print(f"[INFO] case     = {case}")
    print(f"[INFO] tag pref = {tag_prefix}")
    print(f"[INFO] out dir  = {out_root}")
    print(f"[INFO] found {len(exp_tags)} exp_tags")

    copied = 0
    skipped_no_vis = 0
    skipped_no_sample_id = 0
    skipped_sample_filter = 0

    for exp_tag in exp_tags:
        vis_dir = find_vis_dir(root, case, exp_tag)
        if vis_dir is None:
            print(f"[WARN] No vis dir for {exp_tag}, skip")
            skipped_no_vis += 1
            continue

        imgs = list(vis_dir.glob("*.png")) + list(vis_dir.glob("*.jpg"))
        if not imgs:
            print(f"[WARN] No images found in {vis_dir}, skip")
            continue

        print(f"[INFO] {exp_tag}: found {len(imgs)} images in {vis_dir}")

        for img in imgs:
            sid = extract_sample_id(img.name)
            if sid is None:
                skipped_no_sample_id += 1
                continue

            if sample_set is not None and sid not in sample_set:
                skipped_sample_filter += 1
                continue

            sample_dir = out_root / f"sample_{sid:04d}" / exp_tag
            sample_dir.mkdir(parents=True, exist_ok=True)

            dst = sample_dir / img.name
            # 可以改成 symlink，如果你只想软链：
            # dst.symlink_to(img)
            shutil.copy2(img, dst)
            copied += 1

    print()
    print(f"[OK] Copied {copied} images into {out_root}")
    if skipped_no_vis:
        print(f"     Skipped {skipped_no_vis} experiments with no vis dir")
    if skipped_no_sample_id:
        print(f"     Skipped {skipped_no_sample_id} images with no parseable sample id")
    if skipped_sample_filter:
        print(f"     Skipped {skipped_sample_filter} images due to sample filter")


# ---------------------- CLI ----------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect visualisation images produced by test_tpac.py"
    )
    p.add_argument(
        "--root",
        type=Path,
        default=_default_root(),
        help="Project root (包含 Rotate/ 的目录). 默认为脚本所在目录或其父目录",
    )
    p.add_argument("--case", required=True, help="Dataset case, e.g. C1 / ICA_norm")
    p.add_argument("--tag", required=True, help="Base tag prefix, e.g. 1205")
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="输出根目录，用于集中保存可视化结果",
    )
    p.add_argument(
        "--samples",
        type=int,
        nargs="*",
        help="可选：只收集这些样本 id（从文件名解析），例如 --samples 10 42 99",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    root = args.root.resolve()
    out_root = args.out.resolve()

    collect_vis(
        root=root,
        case=args.case,
        tag_prefix=args.tag,
        out_root=out_root,
        sample_filter=args.samples,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
