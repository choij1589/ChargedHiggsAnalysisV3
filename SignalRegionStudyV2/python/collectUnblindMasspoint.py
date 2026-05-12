#!/usr/bin/env python3
"""Collect unblind review artifacts for one mass point.

This script copies existing GoF, post-fit mass, impact, and nuisance-pull
artifacts into a compact per-masspoint folder. It does not run Combine or
regenerate plots.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ERAS = ("Run2", "Run3", "All")


@dataclass
class Entry:
    category: str
    era: str
    status: str
    source: str
    destination: str = ""
    note: str = ""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def binning_suffix(binning: str, nuisance: str) -> str:
    suffix = f"{binning}_unblind"
    if nuisance == "preserve_shape":
        suffix = f"{suffix}_preserve_shape"
    return suffix


def parse_eras(raw: str) -> tuple[str, ...]:
    eras = tuple(era.strip() for era in raw.split(",") if era.strip())
    if not eras:
        raise argparse.ArgumentTypeError("at least one era is required")
    return eras


def copy_artifact(
    source: Path,
    destination: Path,
    category: str,
    era: str,
    entries: list[Entry],
    dry_run: bool,
    note: str = "",
) -> None:
    if not source.is_file():
        entries.append(
            Entry(
                category=category,
                era=era,
                status="missing",
                source=str(source),
                destination=str(destination),
                note=note,
            )
        )
        return

    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    entries.append(
        Entry(
            category=category,
            era=era,
            status="copied" if not dry_run else "dry_run",
            source=str(source),
            destination=str(destination),
            note=note,
        )
    )


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def copy_first_existing(
    sources: Iterable[Path],
    destination: Path,
    category: str,
    era: str,
    entries: list[Entry],
    dry_run: bool,
) -> None:
    source_list = list(sources)
    source = first_existing(source_list)
    if source is None:
        entries.append(
            Entry(
                category=category,
                era=era,
                status="missing",
                source=";".join(str(path) for path in source_list),
                destination=str(destination),
                note="no candidate source exists",
            )
        )
        return
    copy_artifact(source, destination, category, era, entries, dry_run)


def copy_plot_grid(
    plot_dir: Path,
    output_dir: Path,
    source_era: str,
    entries: list[Entry],
    dry_run: bool,
) -> int:
    if not plot_dir.is_dir():
        return 0

    copied = 0
    patterns = (
        ("prefit_mass", "prefit_mass_*.png"),
        ("postfit_b_mass", "postfit_b_mass_*.png"),
    )
    for category, pattern in patterns:
        sources = sorted(plot_dir.glob(pattern))
        if not sources:
            entries.append(
                Entry(
                    category=category,
                    era=source_era,
                    status="missing",
                    source=str(plot_dir / pattern),
                    note="no matching PNG files",
                )
            )
            continue

        for source in sources:
            destination_name = source.name if source_era == "All" else f"{source_era}_{source.name}"
            destination = output_dir / destination_name
            copy_artifact(source, destination, category, source_era, entries, dry_run)
            copied += 1
    return copied


def write_reports(output_dir: Path, entries: list[Entry], dry_run: bool) -> None:
    if dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "entries": [asdict(entry) for entry in entries],
        "summary": {
            "copied": sum(entry.status == "copied" for entry in entries),
            "missing": sum(entry.status == "missing" for entry in entries),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    missing = [entry for entry in entries if entry.status == "missing"]
    if missing:
        lines = [
            f"[{entry.era}] {entry.category}: {entry.source}"
            + (f" ({entry.note})" if entry.note else "")
            for entry in missing
        ]
        (output_dir / "missing.txt").write_text("\n".join(lines) + "\n")
    else:
        missing_path = output_dir / "missing.txt"
        if missing_path.exists():
            missing_path.unlink()


def collect(args: argparse.Namespace) -> int:
    template_root = args.template_root.resolve()
    output_dir = (args.output_dir / args.masspoint / args.method).resolve()
    suffix = binning_suffix(args.binning, args.nuisance)
    entries: list[Entry] = []
    era_bases: dict[str, Path] = {}

    for era in args.eras:
        base = (
            template_root
            / era
            / "Combined"
            / args.masspoint
            / args.method
            / suffix
            / "combine_output"
        )
        era_bases[era] = base

        copy_artifact(
            base / "gof" / "gof_plot.png",
            output_dir / f"gof_{era}.png",
            "gof",
            era,
            entries,
            args.dry_run,
        )
        if era == "All":
            copy_first_existing(
                (
                    base / "impacts_obs" / "condor" / "impacts.pdf",
                    base / "impacts_obs" / "impacts.pdf",
                ),
                output_dir / "impact_All.pdf",
                "impact",
                era,
                entries,
                args.dry_run,
            )
            copy_artifact(
                base / "fitdiag" / "nuisance_pulls.pdf",
                output_dir / "nuisance_pulls_All.pdf",
                "nuisance_pulls",
                era,
                entries,
                args.dry_run,
            )

    n_mass_plots = 0
    for era, base in era_bases.items():
        n_mass_plots += copy_plot_grid(
            base / "fitdiag" / "plots_mass", output_dir, era, entries, args.dry_run
        )
    if n_mass_plots == 0:
        entries.append(
            Entry(
                category="postfit_mass",
                era=",".join(args.eras),
                status="missing",
                source=";".join(str(base / "fitdiag" / "plots_mass") for base in era_bases.values()),
                note="no prefit_mass_*.png or postfit_b_mass_*.png files found",
            )
        )

    write_reports(output_dir, entries, args.dry_run)

    copied = sum(entry.status in {"copied", "dry_run"} for entry in entries)
    missing = sum(entry.status == "missing" for entry in entries)
    print(
        f"{args.masspoint} {args.method}: "
        f"{copied} {'would be copied' if args.dry_run else 'copied'}, "
        f"{missing} missing -> {output_dir}"
    )
    if missing and args.strict:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect existing unblind plots for one mass point."
    )
    parser.add_argument("--masspoint", required=True)
    parser.add_argument("--method", required=True, choices=("Baseline", "ParticleNet"))
    parser.add_argument("--binning", default="extended")
    parser.add_argument(
        "--nuisance",
        default="fallback_lnn",
        choices=("fallback_lnn", "preserve_shape"),
    )
    parser.add_argument("--eras", type=parse_eras, default=DEFAULT_ERAS)
    parser.add_argument(
        "--template-root",
        type=Path,
        default=repo_root() / "templates",
        help="Template artifact root [default: repo/templates]",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "results" / "unblind",
        help="Collection output root [default: repo/results/unblind]",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any artifact is missing")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be collected")
    return parser


def main() -> int:
    return collect(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
