import re
import os
import sys
import glob
import uuid
import yaml
import argparse


def get_step(p):
    return int(p.replace("step-", "").replace("LATEST", '0'))


def is_sft(path):
    return (
        "sft" in path
        or "b0-a0" in path
        and "sample" not in path
        and "reward" not in path
    )


def get_sample_path(sample_dir, archive, ckpt_dir, ds, do_sample, sampling_params, use_sampling_params):
    try:
        with open(os.path.join(args.archive_dir, archive, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        if config['loss']['name'] == "dpo":
            beta, alpha = config['loss']['beta'], config['loss']['alpha']
        else:
            beta, alpha = 0.0, 0.0

    except FileNotFoundError:
        print(f"no config.yaml, assuming sft... ", end="")
        beta, alpha = 0.0, 0.0

    if args.beta is not None and args.beta != beta:
        if beta != 0 or (not args.rewards and not args.rewards_on_samples):
            return None
    elif args.alpha is not None and args.alpha != alpha:
        return None

    model = args.model
    ckpt_dir = ckpt_dir.replace("step-", "")
    sample_type = "rewards" if not do_sample else "samples"
    ext = "csv" if not do_sample else "json"

    path_template = f"{model}__{ds}__b{beta}__a{alpha}__s{ckpt_dir}__{sample_type}"
    if use_sampling_params:
        for pname, pvalue in sampling_params.items():
            path_template += f"__{pname}{pvalue}"
    path_template += f".{ext}"

    return os.path.join(args.sample_dir, path_template)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="dataset name to use"
    )
    parser.add_argument(
        "--archive_dir",
        type=str,
        required=True,
        help="path to directory containing runs/archives"
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        required=True,
        help="path to directory to sample samples in",
    )
    parser.add_argument(
        "--archive",
        type=str,
        nargs="*",
        help="archive(s) in archive_dir to sample from, default to all"
    )
    parser.add_argument(
        "--rewards",
        action="store_true",
        help="if true, compute rewards instead of sampling"
    )
    parser.add_argument(
        "--rewards_on_samples",
        action="store_true",
        help="if true, compute rewards over sampled responses instead of dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="name of model we're sampling from",
        default="pythia28"
    )
    parser.add_argument(
        "--n_ckpts",
        type=int,
        help="how many checkpoints to sample from per run",
        default=None,
    )
    parser.add_argument(
        "--allow_fewer_ckpts",
        action="store_true",
        help="if true, don't skip archives with less than n_ckpts checkpoints"
    )
    parser.add_argument(
        "--sft_archive",
        type=str,
        help="sft archive for rewards sampling if not automatically determined",
        default=None
    )
    parser.add_argument(
        "--min_step",
        type=int,
        help="min checkpoint step to sample from",
        default=0
    )
    parser.add_argument(
        "--max_step",
        type=int,
        help="max checkpoint step, useful if runs are different lengths",
        default=None
    )
    parser.add_argument(
        "--max_len",
        type=int,
        help="if sampling, max number of new tokens to generate on top of prompt",
        default=512
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="if true, only match run dirs '[MODEL]-[DATASET]-b[#]-a[#]'"
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="filter by a certain beta value",
        default=None
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="filter by a certain alpha value",
        default=None
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size for sampling",
        default=4
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="number of beams for sampling"
    )
    parser.add_argument(
        "--repetition_penalty",
        default=1.0,
        type=float,
        help="repetition penalty for sampling"
    )
    parser.add_argument(
        "--top_k",
        default=50,
        type=int,
        help="top k for sampling"
    )
    parser.add_argument(
        "--penalty_alpha",
        default=0.0,
        type=float,
        help="alpha penalty for sampling"
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        help="sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        default=1.0,
        type=float,
        help="nucleus parameter for sampling"
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        default=0,
        type=int,
        help="disallowed ngram repeat"
    )
    parser.add_argument(
        "--use_sampling_params",
        action="store_true",
        help="if true, encode sampling params in filenames"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="if true, overwrites existing sample files"
    )
    parser.add_argument(
        "--run_anyways",
        action="store_true",
        help="if true, run rewards sampling even if sampling script doesn't exist yet"
    )
    parser.add_argument(
        "--exclude_last_ckpt",
        action="store_true",
        help="if true, exclude last checkpoint from sampling (n_ckpt - 1 will be used)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="if true, only submit one batch job"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="if true, do dry run and don't submit any jobs"
    )
    args = parser.parse_args()

    os.makedirs(args.sample_dir, exist_ok=True)

    template = "scripts/templates/sample.sh"
    if args.rewards or args.rewards_on_samples:
        print("collecting rewards, not samples")
        template = "scripts/templates/rewards.sh"

    with open(template, "r") as f_template:
        template = "\n".join(map(str.strip, f_template.readlines()))

    sample_dir = os.path.abspath(os.path.realpath(args.sample_dir))
    pattern = f"^{args.model}-{args.dataset}+-b\d+-a\d+$"
    collected = []
    if args.n_ckpts is not None:
        fractions = [x / (args.n_ckpts - 1) for x in range(max(args.n_ckpts, 1) - 1)]
        if not args.exclude_last_ckpt:
            fractions.append(1)
        print("fractions for sampling:", fractions)

    sampling_params = {
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
        "top_k": args.top_k,
        "penalty_alpha": args.penalty_alpha,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "no_repeat_ngram_size": args.no_repeat_ngram_size
    }

    n_to_sample = 0
    files = args.archive or os.listdir(args.archive_dir)
    for i, fdir in enumerate(files):
        print(f"({i + 1}) checking {fdir}... ", end="")
        if args.strict and re.match(pattern, fdir) is None:
            print("failed strict re match")
            continue
        elif args.dataset not in fdir:
            print("dataset name not in path")
            continue
        elif fdir not in os.listdir(args.archive_dir):
            print("not found in archive dir")
            continue

        archive_dirs = glob.glob(os.path.join(args.archive_dir, fdir, "step-*"))
        if len(archive_dirs) == 0:
            archive_dirs.extend(glob.glob(os.path.join(args.archive_dir, fdir, "LATEST")))

        archive_dirs = map(os.path.basename, archive_dirs)
        archive_dirs = list(sorted(archive_dirs, key=get_step))
        if not is_sft(fdir) or args.min_step > 0:
            archive_dirs = [d for d in archive_dirs if args.min_step < get_step(d) <= (args.max_step or float("inf"))]
        
        if not is_sft(fdir) and (args.n_ckpts is not None and len(archive_dirs) < args.n_ckpts) and not args.allow_fewer_ckpts:
            print(f"too few ckpts ({len(archive_dirs)} < {args.n_ckpts})")
            continue

        if args.n_ckpts is not None and len(archive_dirs) > args.n_ckpts:
            ind = [min(int(round(f * len(archive_dirs))), len(archive_dirs) - 1) for f in fractions]
            archive_dirs = [archive_dirs[i] for i in ind]

        ckpt_samples_info = []
        do_sample = not args.rewards
       
        continue_ = False
        for ckpt_dir in archive_dirs:
            ckpt_sample_path = get_sample_path(
                sample_dir, fdir, ckpt_dir, args.dataset,
                do_sample=do_sample, sampling_params=sampling_params,
                use_sampling_params=args.use_sampling_params
            )
            if ckpt_sample_path is None:
                continue_ = True
                break

            dataset = args.dataset

            if (os.path.exists(ckpt_sample_path) or args.run_anyways) and args.rewards_on_samples:
                ckpt_sample_path = os.path.abspath(os.path.realpath(ckpt_sample_path))
                dataset = ckpt_sample_path
                ds_id = f"{args.dataset}_local_samples"
                ckpt_sample_path = get_sample_path(
                    sample_dir, fdir, ckpt_dir, ds_id,
                    do_sample=False,
                    sampling_params=sampling_params,
                    use_sampling_params=args.use_sampling_params
                )

            elif args.rewards_on_samples:
                print(ckpt_sample_path)
                print("rewards_on_samples but no sample path found")
                continue
 
            ckpt_samples_info.append((ckpt_dir, ckpt_sample_path, dataset))
            n_to_sample += 1

        if continue_:
            print(f"alpha/beta value mismatch")
            continue

        print(f"found {len(ckpt_samples_info)} after filtering")
        if len(ckpt_samples_info) != 0:
            collected.append((ckpt_samples_info, fdir))

    collected = list(sorted(collected, key=lambda d: len(d[0])))
    print()
    print("=" * 40)
    print(f"collected {len(collected)} dirs, {n_to_sample} ckpts to sample from")
    for ckpt_samples_info, fdir in collected:
        for archive_dir, sample_path, ds in ckpt_samples_info:
            print(f"- DIR: {fdir}, CKPT: {archive_dir}, DATASET: {ds}, SAMPLE: {os.path.basename(sample_path)}")

    sft_archive = None
    if args.rewards or args.rewards_on_samples:
        if args.sft_archive is not None:
            sft_archive = args.sft_archive
        else:
            try:
                sft_archives = list(
                    sorted(
                        filter(lambda p: is_sft(p[1]), collected),
                        key=lambda p: os.path.getsize(os.path.join(args.archive_dir, p[1]))
                    )
                )
                sft_archive = os.path.join(args.archive_dir, sft_archives[0][1], sft_archives[0][0][-1][0])
                if not sft_archive.endswith("policy.pt"):
                    sft_archive = os.path.join(sft_archive, "policy.pt")
            except IndexError:
                raise ValueError("no sft archive found, can't compute rewards")
        print(f"using {sft_archive} as sft archive")

    print("=" * 40)
    first = True

    for ckpt_samples_info, fdir in collected:
        for archive_dir, sample_path, dataset in ckpt_samples_info:
            if os.path.exists(sample_path) and not args.overwrite:
                print(f"skipping existing sample {sample_path}")
                continue
            elif sft_archive is not None and (fdir in sft_archive or "b0-a0" in fdir):
                print(f"skipping sft archive {fdir} for reward computation")
                continue
           
            model_archive = os.path.join(args.archive_dir, fdir, archive_dir, "policy.pt")
            cmd = template.format(
                model_archive=model_archive,
                sample_path=sample_path,
                dataset=dataset,
                dataset_id=args.dataset + ("_samples" if args.rewards_on_samples else ""),
                sft_archive=sft_archive,
                max_len=args.max_len,
                model=args.model,
                batch_size=args.batch_size,
                **sampling_params
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

            if args.dry_run:
                print(f"would write to {sample_path}... dry run complete")
            else:
                os.system(f"sbatch {tmpfile}")
            os.remove(tmpfile)
