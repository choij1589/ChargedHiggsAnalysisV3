#!/usr/bin/env python3
"""
HTCondor job submission script for HybridNew toy generation.

Usage:
    submit_hybridnew.py --workspace /path/to/workspace.root --rmin 0.1 --rmax 5.0 --rstep 0.5 --ntoys 500 --njobs 10
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path


def generate_r_values(rmin, rmax, rstep):
    """Generate list of r values to scan."""
    r_values = []
    r = rmin
    while r <= rmax:
        r_values.append(round(r, 4))
        r += rstep
    return r_values


def create_wrapper_script(output_dir, ntoys):
    """Create wrapper script for condor job."""
    wrapper_path = output_dir / "run_toy.sh"

    wrapper_content = f"""#!/bin/bash
R_VALUE=$1
SEED=$2

# Setup CMSSW environment
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el8_amd64_gcc12

# Run combine
combine -M HybridNew workspace.root \\
    --LHCmode LHC-limits \\
    --singlePoint ${{R_VALUE}} \\
    --saveToys \\
    --saveHybridResult \\
    -T {ntoys} \\
    -s ${{SEED}} \\
    -n .r${{R_VALUE}}.seed${{SEED}}
"""

    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)

    wrapper_path.chmod(0o755)
    return wrapper_path


def create_job_params(output_dir, r_values, njobs):
    """Create job parameters file."""
    params_path = output_dir / "job_params.txt"

    with open(params_path, 'w') as f:
        for r in r_values:
            for j in range(njobs):
                seed = 1000 + j
                f.write(f"{r} {seed}\n")

    return params_path


def create_submission_file(output_dir, workspace_path, wrapper_path, params_path, flavour="longlunch"):
    """Create HTCondor submission file."""
    sub_path = output_dir / "hybridnew.sub"

    sub_content = f"""universe = vanilla
executable = {wrapper_path}
arguments = $(r_value) $(seed)

output = {output_dir}/job_$(Cluster)_$(Process).out
error = {output_dir}/job_$(Cluster)_$(Process).err
log = {output_dir}/condor.log

request_cpus = 1
request_memory = 2GB
request_disk = 500MB

+JobFlavour = "{flavour}"

should_transfer_files = YES
transfer_input_files = {workspace_path}
when_to_transfer_output = ON_EXIT

queue r_value,seed from {params_path}
"""

    with open(sub_path, 'w') as f:
        f.write(sub_content)

    return sub_path


def main():
    parser = argparse.ArgumentParser(description="Submit HybridNew jobs to HTCondor")
    parser.add_argument("--workspace", required=True, help="Path to workspace.root")
    parser.add_argument("--output-dir", default=None, help="Output directory for condor files")
    parser.add_argument("--rmin", type=float, default=0.1, help="Minimum r value")
    parser.add_argument("--rmax", type=float, default=5.0, help="Maximum r value")
    parser.add_argument("--rstep", type=float, default=0.5, help="r value step")
    parser.add_argument("--ntoys", type=int, default=500, help="Toys per job")
    parser.add_argument("--njobs", type=int, default=10, help="Jobs per r value")
    parser.add_argument("--flavour", default="longlunch",
                        choices=["espresso", "microcentury", "longlunch", "workday", "tomorrow", "testmatch", "nextweek"],
                        help="HTCondor job flavour")
    parser.add_argument("--dry-run", action="store_true", help="Create files but don't submit")
    args = parser.parse_args()

    # Validate workspace
    workspace_path = Path(args.workspace).resolve()
    if not workspace_path.exists():
        print(f"ERROR: Workspace not found: {workspace_path}")
        sys.exit(1)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = workspace_path.parent / "combine_output" / "hybridnew" / "condor"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate r values
    r_values = generate_r_values(args.rmin, args.rmax, args.rstep)
    total_jobs = len(r_values) * args.njobs

    print(f"HybridNew HTCondor Submission")
    print(f"=" * 50)
    print(f"Workspace: {workspace_path}")
    print(f"Output dir: {output_dir}")
    print(f"r values: {r_values}")
    print(f"Toys per job: {args.ntoys}")
    print(f"Jobs per r: {args.njobs}")
    print(f"Total jobs: {total_jobs}")
    print(f"Total toys: {total_jobs * args.ntoys}")
    print(f"Job flavour: {args.flavour}")
    print()

    # Create files
    wrapper_path = create_wrapper_script(output_dir, args.ntoys)
    print(f"Created: {wrapper_path}")

    params_path = create_job_params(output_dir, r_values, args.njobs)
    print(f"Created: {params_path}")

    sub_path = create_submission_file(output_dir, workspace_path, wrapper_path, params_path, args.flavour)
    print(f"Created: {sub_path}")

    # Submit
    if args.dry_run:
        print()
        print("[DRY-RUN] Would submit with: condor_submit", sub_path)
    else:
        print()
        print("Submitting jobs...")
        result = subprocess.run(["condor_submit", str(sub_path)], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"ERROR: Submission failed: {result.stderr}")
            sys.exit(1)

        print(result.stdout)
        print()
        print("Jobs submitted successfully!")
        print("Monitor with: condor_q")
        print(f"After completion, merge outputs with:")
        print(f"  hadd {output_dir.parent}/hybridnew_grid.root {output_dir}/../toys/higgsCombine*.root")


if __name__ == "__main__":
    main()
