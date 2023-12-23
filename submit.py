import sys
import os

if __name__ == "__main__":
    msg = f"submitting from root dir {os.getcwd()}"
    print(msg)

    sample = input("are you sampling (y/N)? ")
    if sample.lower() == "y":
        f = "scripts/sample.sh"

    else:
        dirs_ = [d for d in os.listdir('scripts') if os.path.isdir(os.path.join("scripts", d))]
        ds = input(f"choose dataset {dirs_}: ")
        if ds not in dirs_:
            print("ERROR: selected not in directories")
            sys.exit(1)
       
        model = input(f"choose model (pythia28): ")
        if not model:
            model = "pythia28"

        mod = input("add length mod (y/N)? ")
        f = f"scripts/{ds}/{'base' if not mod else 'mod'}_{model}.sh"
        if not os.path.exists(f):
            print(f"ERROR: {f} does not exist")
            sys.exit(1)

    print("\nprinting requested file below...")
    print("=" * 80)
    with open(f, "r") as fb:
        for line in fb:
            start, end = "\033[1m", "\033[0;0m"
            if line.startswith("#"):
                start, end = "\x1B[3m", "\x1B[0m"
            print(start + line.strip() + end)
            
    print("=" * 80)

    cmd = f"sbatch {f}" 
    msg = f"Execute '{cmd}' (Y/n)? "
    if input(msg).lower() != "n":
        os.system(cmd)
        os.system("watch squeue -u $USER")
