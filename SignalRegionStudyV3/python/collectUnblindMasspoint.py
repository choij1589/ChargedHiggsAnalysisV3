#!/usr/bin/env python3
"""Collect unblind review artifacts for one mass point.

This script copies existing GoF, prefit/postfit mass, impact, and nuisance-pull
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
DEFAULT_CHANNELS = ("SR1E2Mu", "SR3Mu", "Combined")
DEFAULT_POSTFIT_TARGETS = tuple(
    (era, channel) for era in DEFAULT_ERAS for channel in DEFAULT_CHANNELS
)
DEFAULT_GOF_TARGETS = (
    ("All", "Combined"),
    ("All", "SR1E2Mu"),
    ("All", "SR3Mu"),
    ("Run2", "Combined"),
    ("Run3", "Combined"),
)
DEFAULT_SCORE_TARGETS = DEFAULT_POSTFIT_TARGETS
SCORE_FILENAMES = (
    "LR_modified.png",
)


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


def parse_targets(raw: str) -> tuple[tuple[str, str], ...]:
    targets: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                f"target '{item}' must use ERA:CHANNEL format"
            )
        era, channel = (part.strip() for part in item.split(":", 1))
        if era not in DEFAULT_ERAS:
            raise argparse.ArgumentTypeError(f"unsupported era '{era}'")
        if channel not in DEFAULT_CHANNELS:
            raise argparse.ArgumentTypeError(f"unsupported channel '{channel}'")
        targets.append((era, channel))
    if not targets:
        raise argparse.ArgumentTypeError("at least one target is required")
    return tuple(targets)


def format_targets(targets: Iterable[tuple[str, str]]) -> str:
    return ",".join(f"{era}:{channel}" for era, channel in targets)


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
    note: str = "",
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
                note=f"{note}: no candidate source exists" if note else "no candidate source exists",
            )
        )
        return
    copy_artifact(source, destination, category, era, entries, dry_run, note=note)


def prepare_output_dir(output_dir: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def fitdiag_plot_candidates(
    template_root: Path,
    target_era: str,
    target_channel: str,
    masspoint: str,
    method: str,
    suffix: str,
    filename: str,
) -> tuple[Path, ...]:
    target_base = (
        template_root
        / target_era
        / target_channel
        / masspoint
        / method
        / suffix
        / "combine_output"
        / "fitdiag"
        / "plots_mass"
        / filename
    )
    combined_all_base = (
        template_root
        / "All"
        / "Combined"
        / masspoint
        / method
        / suffix
        / "combine_output"
        / "fitdiag"
        / "plots_mass"
        / filename
    )
    if target_base == combined_all_base:
        return (target_base,)
    return (target_base, combined_all_base)


def collect_postfit_grid(
    template_root: Path,
    output_dir: Path,
    targets: Iterable[tuple[str, str]],
    masspoint: str,
    method: str,
    suffix: str,
    entries: list[Entry],
    dry_run: bool,
) -> int:
    copied = 0
    for era, channel in targets:
        for category, filename in (
            ("prefit_mass", f"prefit_mass_{era}_{channel}.png"),
            ("postfit_b_mass", f"postfit_b_mass_{era}_{channel}.png"),
            ("postfit_s_mass", f"postfit_s_mass_{era}_{channel}.png"),
        ):
            destination = output_dir / "FitDiag" / filename
            copy_first_existing(
                fitdiag_plot_candidates(
                    template_root,
                    era,
                    channel,
                    masspoint,
                    method,
                    suffix,
                    filename,
                ),
                destination,
                category,
                era,
                entries,
                dry_run,
                note=f"{era}/{channel}",
            )
            if entries[-1].status in {"copied", "dry_run"}:
                copied += 1
    return copied


def collect_gof_grid(
    template_root: Path,
    output_dir: Path,
    targets: Iterable[tuple[str, str]],
    masspoint: str,
    method: str,
    suffix: str,
    entries: list[Entry],
    dry_run: bool,
) -> int:
    copied = 0
    for era, channel in targets:
        base = (
            template_root
            / era
            / channel
            / masspoint
            / method
            / suffix
            / "combine_output"
            / "gof"
        )
        for category, filename, destination_name in (
            ("gof_plot", "gof_plot.png", f"gof_plot_{era}_{channel}.png"),
            ("gof_plot", "gof_plot.pdf", f"gof_plot_{era}_{channel}.pdf"),
            ("gof_json", "gof.json", f"gof_{era}_{channel}.json"),
        ):
            destination = output_dir / "GoF" / destination_name
            copy_artifact(
                base / filename,
                destination,
                category,
                era,
                entries,
                dry_run,
                note=f"{era}/{channel}",
            )
            if entries[-1].status in {"copied", "dry_run"}:
                copied += 1
    return copied


def collect_score_grid(
    template_root: Path,
    output_dir: Path,
    targets: Iterable[tuple[str, str]],
    masspoint: str,
    method: str,
    suffix: str,
    entries: list[Entry],
    dry_run: bool,
) -> int:
    if method != "ParticleNet":
        return 0

    copied = 0
    for era, channel in targets:
        source_dir = (
            template_root
            / era
            / channel
            / masspoint
            / method
            / suffix
            / "scores"
            / channel
        )
        era_tag = "ALL" if era == "All" else era
        for filename in SCORE_FILENAMES:
            source_name = Path(filename)
            destination = (
                output_dir
                / "Scores"
                / f"{source_name.stem}_{era_tag}_{channel}{source_name.suffix}"
            )
            copy_artifact(
                source_dir / filename,
                destination,
                "score_plot",
                era,
                entries,
                dry_run,
                note=f"{era}/{channel}",
            )
            if entries[-1].status in {"copied", "dry_run"}:
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
    prepare_output_dir(output_dir, args.dry_run)

    if not args.skip_gof:
        collect_gof_grid(
            template_root,
            output_dir,
            args.gof_targets,
            args.masspoint,
            args.method,
            suffix,
            entries,
            args.dry_run,
        )

    if not args.skip_scores:
        collect_score_grid(
            template_root,
            output_dir,
            args.score_targets,
            args.masspoint,
            args.method,
            suffix,
            entries,
            args.dry_run,
        )

    if "All" in args.eras and not args.skip_fitdiag:
        base = (
            template_root
            / "All"
            / "Combined"
            / args.masspoint
            / args.method
            / suffix
            / "combine_output"
        )
        copy_first_existing(
            (
                base / "impacts_obs" / "condor" / "impacts.pdf",
                base / "impacts_obs" / "impacts.pdf",
            ),
            output_dir / "impacts.pdf",
            "impact",
            "All",
            entries,
            args.dry_run,
            note="observed final-unblind impact plot (no --blind result hiding)",
        )
        copy_first_existing(
            (
                base / "fitdiag" / "nuisance_pulls_both.pdf",
                base / "fitdiag" / "nuisance_pulls.pdf",
            ),
            output_dir / "nuisance_pulls.pdf",
            "nuisance_pulls",
            "All",
            entries,
            args.dry_run,
            note="prefer _both source; destination name preserved",
        )
        copy_first_existing(
            (
                base / "fitdiag" / "nuisance_pulls_filtered_both.pdf",
                base / "fitdiag" / "nuisance_pulls_filtered.pdf",
            ),
            output_dir / "nuisance_pulls_filtered.pdf",
            "nuisance_pulls_filtered",
            "All",
            entries,
            args.dry_run,
            note="prefer _both source; destination name preserved",
        )

    n_mass_plots = 0
    if not args.skip_fitdiag:
        n_mass_plots = collect_postfit_grid(
            template_root,
            output_dir,
            args.postfit_targets,
            args.masspoint,
            args.method,
            suffix,
            entries,
            args.dry_run,
        )
    if not args.skip_fitdiag and n_mass_plots == 0:
        entries.append(
            Entry(
                category="postfit_mass",
                era=format_targets(args.postfit_targets),
                status="missing",
                source=str(template_root),
                note="no requested prefit_mass, postfit_b_mass, or postfit_s_mass files found",
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
    parser.add_argument("--channel", default="Combined", choices=("Combined", "SR1E2Mu", "SR3Mu"))
    parser.add_argument("--binning", default="extended")
    parser.add_argument(
        "--nuisance",
        default="fallback_lnn",
        choices=("fallback_lnn", "preserve_shape"),
    )
    parser.add_argument("--eras", type=parse_eras, default=DEFAULT_ERAS)
    parser.add_argument(
        "--postfit-targets",
        type=parse_targets,
        default=DEFAULT_POSTFIT_TARGETS,
        help=(
            "Comma-separated ERA:CHANNEL targets for prefit/postfit_b/postfit_s mass plots "
            f"[default: {format_targets(DEFAULT_POSTFIT_TARGETS)}]"
        ),
    )
    parser.add_argument(
        "--gof-targets",
        type=parse_targets,
        default=DEFAULT_GOF_TARGETS,
        help=(
            "Comma-separated ERA:CHANNEL targets for GoF results "
            f"[default: {format_targets(DEFAULT_GOF_TARGETS)}]"
        ),
    )
    parser.add_argument(
        "--score-targets",
        type=parse_targets,
        default=DEFAULT_SCORE_TARGETS,
        help=(
            "Comma-separated ERA:CHANNEL targets for ParticleNet score plots "
            f"[default: {format_targets(DEFAULT_SCORE_TARGETS)}]"
        ),
    )
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
    parser.add_argument("--skip-fitdiag", action="store_true", help="Do not collect prefit/postfit or fitdiag artifacts")
    parser.add_argument("--skip-gof", action="store_true", help="Do not collect GoF artifacts")
    parser.add_argument("--skip-scores", action="store_true", help="Do not collect ParticleNet score plots")
    return parser


def main() -> int:
    return collect(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
