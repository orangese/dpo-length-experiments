import re
import os
import sys
import glob
import uuid
import yaml
import argparse


def is_sft(path):
    return (
        "sft" in path
        or "b0-a0" in path
        and "sample" not in path
        and "reward" not in path
    )


def get_sample_path(sample_dir, archive, ckpt_dir):
    with open(os.path.join(args.archive_dir, archive, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config['loss']['name'] == "dpo":
        beta, alpha = config['loss']['beta'], config['loss']['alpha']
    else:
        beta, alpha = 0.0, 0.0

    b, a = str(beta), str(alpha)
    ckpt_dir = ckpt_dir.replace("step-", "")
    if args.rewards:
        p = f"pythia28__{args.dataset}__b{b}__a{a}__s{ckpt_dir}__rewards.csv"
    else:
        p = f"pythia28__{args.dataset}__b{b}__a{a}__s{ckpt_dir}__samples.json"
    return os.path.join(args.sample_dir, p)


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
        "--rewards",
        action="store_true",
        help="if true, compute rewards instead of sampling"
    )
    parser.add_argument(
        "--n_ckpts",
        type=int,
        help="how many checkpoints to sample from per run",
        default=5,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="if true, only match run dirs 'pythia28-[DATASET]-b[#]-a[#]'"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="if true, only submit one batch job"
    )
    args = parser.parse_args()

    template = "scripts/templates/sample.sh"
    if args.rewards:
        print("collecting rewards, not samples")
        template = "scripts/templates/rewards.sh"

    with open(template, "r") as f_template:
        template = "\n".join(map(str.strip, f_template.readlines()))

    sample_dir = os.path.abspath(os.path.realpath(args.sample_dir))
    pattern = f"^pythia28-{args.dataset}+-b\d+-a\d+$"
    collected = []
    fractions = [x / (args.n_ckpts - 1) for x in range(args.n_ckpts - 1)]
    fractions.append(1)

    for i, fdir in enumerate(os.listdir(args.archive_dir)):
        print(f"({i + 1}) checking {fdir}... ", end="")
        if args.strict and re.match(pattern, fdir) is None:
            print("failed strict re match")
            continue
        elif args.dataset not in fdir:
            print("dataset name not in path")
            continue

        archive_dirs = glob.glob(os.path.join(args.archive_dir, fdir, "step-*"))
        if len(archive_dirs) == 0:
            archive_dirs.extend(glob.glob(os.path.join(args.archive_dir, fdir, "LATEST")))
        archive_dirs = map(os.path.basename, archive_dirs)
        archive_dirs = sorted(archive_dirs, key=lambda p: int(p.replace("step-", "").replace("LATEST", '0')))
        
        if not is_sft(fdir) and len(archive_dirs) < args.n_ckpts:
            print(f"too few ckpts ({len(archive_dirs)} < {args.n_ckpts})")
            continue

        print(f"found {len(archive_dirs)}")
        if len(archive_dirs) > args.n_ckpts:
            ind = [min(int(round(f * len(archive_dirs))), len(archive_dirs) - 1) for f in fractions]
            archive_dirs = list(archive_dirs)
            archive_dirs = [(archive_dirs[i], get_sample_path(sample_dir, fdir, archive_dirs[i])) for i in ind]
        else:
            archive_dirs = [(d, get_sample_path(sample_dir, fdir, d)) for d in archive_dirs]
        collected.append((archive_dirs, fdir))

    collected = list(sorted(collected, key=lambda d: len(d[0])))
    print()
    print("=" * 40)
    print(f"collected {len(collected)} dirs, {len(collected) * args.n_ckpts} ckpts to sample from")
    for archive_dirs, fdir in collected:
        for archive_dir, sample_path in archive_dirs:
            print(f"- DIR: {fdir}, CKPT: {archive_dir}, SAMPLE: .../{os.path.basename(sample_path)}")

    sft_archive = None
    if args.rewards:
        try:
            sft_archives = list(
                sorted(
                    filter(lambda p: is_sft(p[1]), collected),
                    key=lambda p: os.path.getsize(os.path.join(args.archive_dir, p[1]))
                )
            )
            sft_archive = sft_archives[0]
            print(f"using {os.path.join(args.archive_dir, sft_archive[1], sft_archive[0][-1][0])} as sft archive")

        except IndexError:
            raise ValueError("no sft archive found, can't compute rewards")

    print("=" * 40)
    first = True

    for archive_dirs, fdir in collected:
        for archive_dir, sample_path in archive_dirs:
            if os.path.exists(sample_path):
                print(f"skipping existing sample {sample_path}")
                continue
            elif (sft_archive or [None, None])[1] == fdir:
                print("skipping sft archive for reward computation")
                continue

            model_archive = os.path.join(args.archive_dir, fdir, archive_dir, "policy.pt")
            try:
                sft_archive_ = os.path.join(args.archive_dir, sft_archive[1], sft_archive[0][-1][0], "policy.pt")
            except:
                sft_archive_ = sft_archive
            
            cmd = template.format(
                model_archive=model_archive,
                sample_path=sample_path,
                dataset=args.dataset,
                sft_archive=sft_archive_
            )

            if first:
                print()
                print("=" * 20 + " EXAMPLE " + "=" * 20)
                print(cmd)
                print("=" * 20 + "=========" + "=" * 20)

                if input("press enter to execute, q/n to quit: ").lower() in ("q", "n"):
                    print("quitting program...")
                    sys.exit(0)
                first = False
            elif args.debug:
                print("quitting early for debug...")
                sys.exit(0)

            tmpfile = f"{uuid.uuid4()}__sample_tmp.sh"
            with open(tmpfile, "w+") as write:
                for char in cmd:
                    write.write(char)
            os.system(f"sbatch {tmpfile}")
            os.remove(tmpfile)
