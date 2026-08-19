#!/usr/bin/env python3
"""Nuisance grouping for the uncertainty breakdown (docs/BREAKDOWN.md).

The breakdown decomposes sigma(r) by cumulatively freezing nuisance
groups, which Combine takes from `<name> group = <members>` lines in the
datacard.  V4's production datacards carry no such lines and must keep
their exact bytes, so the group definitions are appended to a throwaway
copy -- `printDatacard.py` output is never touched.

Both the worker (`scripts/runBreakdown.sh`) and the collector
(`python/collectBreakdown.py`) need the group list and, crucially, its
ORDER: the order is the cumulative freeze order and therefore fixes which
component each scan measures.  That is why this lives here rather than in
a heredoc inside the worker.
"""
import json
import os
import re
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import srspaths

CONFIG_NAME = "nuisance_groups.json"

# Datacard line-type handling, following the combine datacard grammar.
# `autoMCStats` sitting in SKIP_TYPES is what leaves the generated
# prop_bin* parameters ungrouped, and hence in the residual 'stat'
# component -- deliberate, see docs/BREAKDOWN.md.
SKIP_FIRST_TOKENS = {
    "imax", "jmax", "kmax", "shapes", "bin", "observation", "process",
    "rate",
}
SKIP_TYPES = {"rateParam", "flatParam", "extArg", "autoMCStats", "group"}
CONSTRAINED_TYPES = {
    "lnN", "lnU", "shape", "shape?", "gmN", "gmM", "param", "constr",
}


def load_config():
    with open(srspaths.config_path(CONFIG_NAME)) as f:
        return json.load(f)


def group_names(config=None):
    """Group names in cumulative freeze order."""
    config = config or load_config()
    return [g["name"] for g in config["groups"]]


def component_names(config=None):
    """Every breakdown component: the groups, then the residual."""
    config = config or load_config()
    return group_names(config) + [config["residual"]["name"]]


def component_labels(config=None):
    config = config or load_config()
    labels = {g["name"]: g["label"] for g in config["groups"]}
    labels[config["residual"]["name"]] = config["residual"]["label"]
    return labels


def parse_datacard_nuisances(path):
    """Constrained nuisance names of a datacard, in file order."""
    names = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if set(stripped) == {"-"}:
                continue
            fields = stripped.split()
            if len(fields) < 2:
                continue
            if fields[0] in SKIP_FIRST_TOKENS:
                continue
            if fields[1] in SKIP_TYPES:
                continue
            if fields[1] not in CONSTRAINED_TYPES:
                continue
            names.append(fields[0])
    return names


def classify(names, config=None):
    """Map nuisance names onto groups, in cumulative freeze order.

    Raises ValueError listing every unmatched name.  There is no
    catch-all on purpose: V3 swept unmatched names into `experimental`,
    which would have silently mis-assigned V4's CMS_interp_* family had
    it existed then.  A new family must fail loudly instead.
    """
    config = config or load_config()
    compiled = [(g["name"], [re.compile(p) for p in g["patterns"]])
                for g in config["groups"]]
    groups = OrderedDict((name, []) for name, _ in compiled)
    unmatched = []
    for name in names:
        for group, patterns in compiled:
            if any(p.search(name) for p in patterns):
                groups[group].append(name)
                break
        else:
            unmatched.append(name)
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} nuisance(s) match no group in "
            f"{CONFIG_NAME}: {', '.join(unmatched)}. Add a pattern for "
            "the new family rather than letting it fall into an "
            "unrelated component.")
    return groups


def write_grouped_datacard(src, dst, config=None):
    """Copy `src` verbatim and append its `group =` lines.

    Returns the OrderedDict of groups.  Only non-empty groups get a line
    and are returned as active; a group with no members in this datacard
    would make Combine fail on an unknown group name.
    """
    config = config or load_config()
    groups = classify(parse_datacard_nuisances(src), config)
    with open(src) as f:
        lines = f.readlines()
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w") as out:
        out.writelines(lines)
        if lines and not lines[-1].endswith("\n"):
            out.write("\n")
        out.write("\n# Nuisance groups for the uncertainty breakdown "
                  f"({CONFIG_NAME})\n")
        for group, members in groups.items():
            if members:
                out.write(f"{group} group = {' '.join(members)}\n")
    return groups


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datacard", required=True)
    parser.add_argument("--output", help="grouped datacard to write")
    parser.add_argument("--group-json", help="group membership summary")
    parser.add_argument("--groups-file",
                        help="active group names, one per line, in "
                             "cumulative freeze order")
    args = parser.parse_args()

    config = load_config()
    if args.output:
        groups = write_grouped_datacard(args.datacard, args.output, config)
    else:
        groups = classify(parse_datacard_nuisances(args.datacard), config)

    print(f"Nuisance groups ({args.datacard}):")
    total = 0
    for group, members in groups.items():
        print(f"  {group:16s} {len(members)}")
        total += len(members)
    print(f"  {'TOTAL':16s} {total}")

    if args.group_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.group_json)),
                    exist_ok=True)
        with open(args.group_json, "w") as f:
            json.dump({g: {"count": len(m), "members": m}
                       for g, m in groups.items()}, f, indent=2)
            f.write("\n")
    if args.groups_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.groups_file)),
                    exist_ok=True)
        with open(args.groups_file, "w") as f:
            for group, members in groups.items():
                if members:
                    f.write(group + "\n")


if __name__ == "__main__":
    main()
