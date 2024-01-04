import os
import json
import curses
import argparse
import textwrap
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from collections import defaultdict
from itertools import zip_longest


def load_hh(silent=False):
    """
    Loads HH dataset test split.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split="test")
    reformatted = {}
    i = 0
    kword = "Assistant:"
    for entry in tqdm(dataset, disable=silent):
        s = entry["chosen"].rfind(kword) + len(kword)
        reformatted[entry["chosen"][:s]] = entry["chosen"][s + 1:]
        i += 1
    if not silent:
        print(f"loaded {i} examples from HH test split")
    return reformatted


def load_shp(silent=False):
    """
    Loads SHP dataset test split.
    """
    dataset = load_dataset("stanfordnlp/SHP", split="test")
    reformatted = {}
    i = 0
    for entry in tqdm(dataset, disable=silent):
        key = "A" if int(entry["labels"]) == 0 else "B"
        prompt = f"\n\nHuman: {entry['history']}\n\nAssistant:"
        reformatted[prompt] = entry[f"human_ref_{key}"]
        i += 1
    if not silent:
        print(f"loaded {i} examples from SHP test split")
    return reformatted


def load_samples(sample_dir, to_process=None):
    """
    Get samples from directory and list of json files.
    Returns a dict with keys corresponding to model names, and values
    corresponding to dict of prompt: response pairs.
    """
    if to_process is None:
        to_process = os.listdir(sample_dir)

    sampled = defaultdict(dict)
    kword = "Assistant:"
    for f in to_process:
        if f.endswith(".json"):
            with open(os.path.join(sample_dir, f), "r") as fi:
                tmp = json.load(fi)
                for prompt, v in tmp.items():
                    v = v[0]
                    response = v[v.rfind(kword) + len(kword) + 1:]
                    sampled[f.replace(".json", "")][prompt] = response

    print(f"loaded {len(sampled)} model sample sets")
    return sampled


def as_cols(args, padding=2, headers=None):
    if isinstance(args, str):
        args = [args]
    if isinstance(headers, str):
        headers = [headers]

    cols = curses.COLS
    zip_args = [textwrap.wrap(s, (cols - padding * (len(args) - 1)) // len(args)) for s in args]
    interleaved = [*zip_longest(*zip_args, fillvalue="")]
    width = min(max(len(word) for row in interleaved for word in row) + padding, cols)

    result = []
    if headers is not None:
        interleaved.insert(0, headers)
        interleaved.insert(1, ["-" * len(h) for h in headers])
    for row in interleaved:
        result.append("".join(word.ljust(width) for word in row)[:cols])
    return result


def write(sampled, access_order, prompts, ptr, add_info=True):
    prompt = prompts[ptr]

    to_write = as_cols(prompt.replace("\n\nHuman: ", "").replace("\n\nAssistant: ", ""))
    to_write.append("")

    if add_info:
        import sys
        sys.path.insert(1, "../eda")
        from all_datasets import get_len
        headers = [f"{k.upper()} ({get_len(sampled[k].get(prompt))}T)" for k in access_order]
    else:
        headers = [f"{k.upper()}" for k in access_order]

    responses = [sampled[k].get(prompt, "").replace("\n\nAssistant: ", "") for k in access_order]
    to_write.extend(as_cols(responses, headers=headers, padding=4))

    cols = curses.COLS - 1
    to_write.extend([""] * max(1, (curses.LINES - len(to_write) - 1)))
    to_write.append((f"({ptr + 1}/{len(prompts)}) j: previous prompt, k: next prompt, q/c: quit".ljust(cols), 1))

    return to_write


def main(stdscr, args):
    curses.start_color()
    curses.curs_set(0)
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)

    stdscr.addstr("initializing sample viewer...\n")
    stdscr.refresh()

    stdscr.nodelay(1)
    stdscr.clearok(True)
    
    sampled = load_samples(args.sample_dir, args.sample_files)
    sampled["truth"] = globals()[f"load_{args.dataset}"]()

    access_order = ["truth"] + list(set(sampled.keys()) - {"truth"})
    prompts = list(sampled["truth"].keys())

    ptr = 0
    current_top_line = 0
    text_lines = write(sampled, access_order, prompts, ptr)
    needs_update = True

    while True:
        if needs_update:
            stdscr.erase()
            height, width = stdscr.getmaxyx()

            for i in range(current_top_line, min(current_top_line + height, len(text_lines))):
                line = text_lines[i][:width - 1]
                color = 0
                if isinstance(line, tuple):
                    line, color = line
                stdscr.addstr(i - current_top_line, 0, line, curses.color_pair(color))

            stdscr.refresh()
            needs_update = False

        ch = stdscr.getch()

        if ch == curses.KEY_UP and current_top_line > 0:
            current_top_line -= 1
            needs_update = True
        elif ch == curses.KEY_DOWN and current_top_line < len(text_lines) - height:
            current_top_line += 1
            needs_update = True
        elif ch == ord(" ") or ch == ord("k"):
            ptr = (ptr + 1) % len(prompts)
            text_lines = write(sampled, access_order, prompts, ptr)
            current_top_line = 0
            needs_update = True
        elif ch == ord("j"):
            ptr = max(0, ptr - 1)
            text_lines = write(sampled, access_order, prompts, ptr)
            current_top_line = 0
            needs_update = True

        if ch == ord("q") or ch == ord("c"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_dir",
        default="../sampled",
        help="directory to sample from"
    )
    parser.add_argument(
        "--sample_files",
        nargs="*",
        help="list of files in sample_dir to load"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="dataset sampled from"
    )
    args = parser.parse_args()

    curses.wrapper(partial(main, args=args))
