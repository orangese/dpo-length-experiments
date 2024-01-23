import os
import sys
import glob
import uuid
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name to use"
    )
    parser.add_argument(
        "--archive_dir",
        type=str,
        help="path to directory containing runs/archives"
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        help="path to directory to sample samples in",
    )
    parser.add_argument(
        "--template",
        type=str,
        help="template sbatch file to load",
        default="scripts/templates/sample.sh"
    )
    parser.add_argument(
        "--n_ckpts",
        type=int,
        help="how many checkpoints to sample from per run",
        default=5,
    )
    args = parser.parse_args()

    sample_dir = os.path.abspath(os.path.realpath(args.sample_dir))
    with open(args.template, "r") as f_template:
        template = "\n".join(map(str.strip, f_template.readlines()))

    collected = []
    fractions = [x / (args.n_ckpts - 1) for x in range(args.n_ckpts - 1)]
    fractions.append(1)

    for i, fdir in enumerate(os.listdir(args.archive_dir)):
        print(f"({i + 1}) checking {fdir}... ", end="")

        archive_dirs = glob.glob(os.path.join(args.archive_dir, fdir, "step-*"))
        archive_dirs = map(os.path.basename, archive_dirs)
        archive_dirs = sorted(archive_dirs, key=lambda p: int(p.replace("step-", "")))
        
        if len(archive_dirs) < args.n_ckpts:
            print(f"too few ckpts ({len(archive_dirs)} < {args.n_ckpts})")
            continue

        print(f"found {len(archive_dirs)}")
        ind = [min(int(round(f * len(archive_dirs))), len(archive_dirs) - 1) for f in fractions]
        archive_dirs = [list(archive_dirs)[i] for i in ind]
        collected.append((archive_dirs, fdir))

    collected = list(sorted(collected, key=lambda d: len(d[0])))
    print("=" * 40)
    print(f"collected {len(collected)} dirs, {len(collected) * args.n_ckpts} ckpts to sample from")
    print("=" * 40)

    for archive_dirs, fdir in collected:
        for archive_dir in archive_dirs:
            model_archive = os.path.join(args.archive_dir, fdir, archive_dir, "policy.pt")
            sample_path = os.path.join(sample_dir, f"{fdir}__{archive_dir}.json")
            if os.path.exists(sample_path):
                print(f"skipping existing sample {sample_path}")
                continue 
            cmd = template.format(
                model_archive=model_archive,
                sample_path=sample_path,
                dataset=args.dataset
            )
            tmpfile = f"{uuid.uuid4()}__sample_tmp.sh"
            with open(tmpfile, "w+") as write:
                for char in cmd:
                    write.write(char)
            os.system(f"sbatch {tmpfile}")
            os.remove(tmpfile)

