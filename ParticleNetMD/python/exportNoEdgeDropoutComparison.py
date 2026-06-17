#!/usr/bin/env python3
"""Export aligned ModelComparison predictions for ParticleNet no-edge variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import runModelComparison as mc
import trainBDT as bdt
from SglConfig import load_sgl_config


def find_no_edge_artifacts(signal: str, backend: str, model_glob: str) -> tuple[Path, Path]:
    out_dir = mc.COMPARISON_ROOT / backend / "Combined" / signal / "fold-4"
    model_paths = sorted((out_dir / "models").glob(model_glob))
    if not model_paths:
        raise RuntimeError(f"No checkpoint matching {model_glob} found under {out_dir / 'models'}")
    model_path = model_paths[-1]
    info_path = out_dir / f"{model_path.stem}.json"
    if not info_path.exists():
        raise RuntimeError(f"Missing no-edge metadata: {info_path}")
    return model_path, info_path


def export_predictions(args: argparse.Namespace) -> None:
    config = load_sgl_config(args.config)
    split_args = SimpleNamespace(
        pilot=args.pilot,
        pilot_events_per_class=args.pilot_events_per_class,
        max_events_per_class=args.max_events_per_class,
        cap_test=args.cap_test,
    )
    folds, caps, pilot_events_per_class = mc.split_settings(split_args, config)
    table_dir = mc.COMPARISON_ROOT / "dataset" / args.signal / "fold-4" / "tables"
    out_dir = mc.COMPARISON_ROOT / args.backend / "Combined" / args.signal / "fold-4"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path, info_path = find_no_edge_artifacts(args.signal, args.backend, args.model_glob)
    with open(info_path) as handle:
        metadata = json.load(handle)
    hyperparameters = metadata.get("hyperparameters", {})
    loss_type = args.loss_type or hyperparameters.get("loss_type", "weighted_ce")
    disco_lambda = (
        args.disco_lambda
        if args.disco_lambda is not None
        else float(hyperparameters.get("disco_lambda", 0.0))
    )
    edge_dropout_p = float(hyperparameters.get("edge_dropout_p", 0.0))
    thresholds = bdt.load_thresholds(args.signal)
    summary = {
        "signal": args.signal,
        "backend": args.backend,
        "channel": "Combined",
        "source_model": str(model_path),
        "source_metadata": str(info_path),
        "loss_type": loss_type,
        "disco_lambda": disco_lambda,
        "edge_dropout_p": edge_dropout_p,
    }

    for split in ["train", "test"]:
        table_path = table_dir / f"{split}_{bdt.cache_suffix(caps[split], pilot_events_per_class)}.npz"
        table = mc.load_table(table_path)
        scores = mc.evaluate_particle_net_artifacts(
            config,
            args.signal,
            folds[split],
            caps[split],
            args.workers,
            args.device,
            model_path,
            info_path,
            pilot_events_per_class=pilot_events_per_class,
            split_name=split,
        )
        if len(scores) != len(table["y"]):
            raise RuntimeError(f"{split} length mismatch: {len(scores)} != {len(table['y'])}")
        lr = bdt.compute_lr_modified(scores, table["era"], table["channel_id"], thresholds)
        mc.save_table(
            out_dir / f"predictions_{split}.npz",
            {
                "y": table["y"],
                "weight": table["weight"],
                "mass1": table["mass1"],
                "mass2": table["mass2"],
                "era": table["era"],
                "channel_id": table["channel_id"],
                "pn_scores": scores,
                "pn_lr": lr,
            },
        )
        avg_auc, aucs = bdt.average_signal_vs_bg_auc(table["y"], scores, table["weight"])
        summary[split] = {
            "auc": aucs,
            "average_auc": avg_auc,
            "mass_correlation": bdt.mass_correlation_metrics(scores, table),
        }

    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", default="MHc130_MA90")
    parser.add_argument("--config", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="ParticleNet_NoEdgeDropout")
    parser.add_argument("--model-glob", default="*edgeDrop0*.pt")
    parser.add_argument("--loss-type", default=None, help="Override exported loss metadata")
    parser.add_argument("--disco-lambda", type=float, default=None, help="Override exported DisCo lambda metadata")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-events-per-class", type=int, default=250)
    parser.add_argument("--max-events-per-class", type=int, default=None)
    parser.add_argument("--cap-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    export_predictions(parse_args())


if __name__ == "__main__":
    main()
