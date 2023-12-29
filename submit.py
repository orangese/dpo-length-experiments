import sys
import os

if __name__ == "__main__":
    msg = f"submitting from root dir {os.getcwd()}"
    print(msg)

    sample = input("are you sampling (y/N)? ")
    dirs_ = [d for d in os.listdir('scripts') if os.path.isdir(os.path.join("scripts", d))]
    ds = input(f"choose dataset {dirs_}: ")
    if ds not in dirs_:
        print("ERROR: selected not in directories")
        sys.exit(1)
    
    if sample.lower() == "y":
        f = f"scripts/{ds}/sample.sh"

    else:   
        model = input(f"choose model (pythia28): ") or "pythia28"
        alpha = input("length penalty (0)? ") or "0.0"
        
        f = f"scripts/{ds}/mod_{alpha[2:]}_{model}.sh"
        if not os.path.exists(f):
            print(f"ERROR: {f} does not exist")
            sys.exit(1)

    print("\nprinting requested file below...")
    print("=" * 80)
    ds_found = set()
    with open(f, "r") as fb:
        for line in fb:
            start, end = "\033[1m", "\033[0;0m"
            if line.startswith("#"):
                start, end = "\x1B[3m", "\x1B[0m"
            print(start + line.strip() + end)
            for dir_ in dirs_:
                if dir_ in line:
                    ds_found.add(dir_)
            
    print("=" * 80)
    print(f"datasets found: {ds_found}")
    
    if len(ds_found) != 1 or list(ds_found)[0] != ds:
        print("ERROR: datasets found in job file != requested ds")
        sys.exit(1)

    cmd = f"sbatch {f}" 
    msg = f"Execute '{cmd}' (Y/n)? "
    if input(msg).lower() != "n":
        os.system(cmd)
        #os.system("watch squeue -u $USER")
