# %%
import transformers
import datasets
import torch
import tqdm
import textwrap
import openai
import os
import hashlib
import json
import time
from collections import defaultdict
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import argparse
# %%
cache_dir = '/scr/rypark/'

# output directory for responses
step_idx = 'step_2400.pt'
checkpoint_dir = {
    'base': None,
    'sft': None,
    'ppo': None,
    'ppo_temp0': None,
    'fm': os.path.join('/iris/u/em7/code/easy-rlhf/output/20bf_2023-05-12_00-24-43', step_idx),
    'dpo_bhalf': os.path.join('/iris/u/em7/code/easy-rlhf/output/b391_2023-05-12_00-28-51', step_idx),
    'dpo_b1': os.path.join('/iris/u/em7/code/easy-rlhf/output/63d6_2023-05-12_00-24-29', step_idx)
}

output_dir = '/sailhome/rypark/dpo-length-experiments/evals/gpt4_eval_samples'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'results_new_prompt'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'results_new_ref_prompt'), exist_ok=True)

base_name = 'tldr_gptj_'
all_models = {}
for key in ['sft', 'base', 'ppo', 'ppo_temp0', 'fm', 'dpo_bhalf', 'dpo_b1']:
    all_models[key] = [base_name + key, checkpoint_dir[key], os.path.join(output_dir, base_name + key + '.json')]


# %%
all_keys = ['base', 'sft', 'ppo', 'fm', 'dpo', 'dpo_longer', 'fm_total']
all_temps = [0.01, 0.25, 0.5, 0.75, 1.0]
response_map = {key:{} for key in all_keys} 
for key in all_keys:
    if key in ['dpo_longer', 'fm_total']:
        continue
    if key == 'dpo':
        file_key = 'dpo_bhalf'
    else:
        file_key = key

    for temp in all_temps:
        if temp == 1.0:
            response_path = os.path.join(output_dir, f'{base_name}{file_key}.json')
        else:
            response_path = os.path.join(output_dir, f'{base_name}{file_key}_temp{temp}.json')
        with open(response_path, 'r') as f:
            response_map[key][temp] = json.load(f)

for key in ['dpo_longer', 'fm_total']:
    for temp in all_temps:
        if key == 'dpo_longer':
            with open(os.path.join(output_dir, f'/iris/u/em7/code/easy-rlhf/gpt4_eval_samples/tldr_gptj_dpo_bhalf_3k_temp{temp}.json'), 'r') as f:
                response_map[key][temp] = json.load(f)
        elif key == 'fm_total':
            with open(os.path.join(output_dir, f'/iris/u/em7/code/easy-rlhf/gpt4_eval_samples/tldr_gptj_fm_total_temp{temp}.json'), 'r') as f:
                response_map[key][temp] = json.load(f)
# %%
with open(os.path.join(output_dir, 'tldr_gptj_dpo_bhalf_temp0.01.json'), 'r') as f:
    check_file = json.load(f)
with open(os.path.join(output_dir, 'tldr_gptj_dpo_bhalf_temp0.25.json'), 'r') as f:
    check_file2 = json.load(f)
with open(os.path.join(output_dir, 'tldr_gptj_dpo_bhalf_temp0.75.json'), 'r') as f:
    check_file3 = json.load(f)

assert all([x == y for x, y in zip(check_file, response_map['dpo'][0.01])])
idx = 90
for f in [check_file, check_file2, check_file3]:
    print(f[idx][f[idx].index('TL;DR:'):])

# %%
def _get_response_text(post):
    # return the post itself if there is no TL;DR
    try:
        return post[post.index('TL;DR:') + 6:].strip()
    except:
        return post

def get_rating_prompt(dpo, sft):
    human_query = dpo[dpo.index('POST:') + 5:dpo.index('TL;DR:')].strip()

    prompt = "Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? A good summary is both precise and concise.\n\n"
    # prompt = "Which of the following summaries does a better job of summarizing the most important points in the given forum post?\n\n"
    prompt += f'Post:\n{human_query}\n\n'
    response_a, response_b = dpo, sft
    shuffled = False
    if random.random() < 0.5:
        response_a, response_b = response_b, response_a
        shuffled = True

    prompt += f'Summary A:\n{_get_response_text(response_a)}\n\nSummary B:\n{_get_response_text(response_b)}\n\n'
    prompt += 'FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:\n'
    prompt += 'Comparison: <one-sentence comparison and explanation>\nPreferred: <"A" or "B">'
    return prompt, shuffled

def _cached_function(fn_to_cache, cache_dir='/iris/u/em7/code/easy-rlhf/openai_cache/json'):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    def wrapped(*args, **kwargs):
        no_cache = False
        if 'no_cache' in kwargs:
            no_cache = kwargs['no_cache']
            del kwargs['no_cache']
        if 'seed' in kwargs:
            if kwargs['seed'] is None:
                del kwargs['seed']            

        json_dump_args_kwargs = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
        hash = hashlib.sha256(json_dump_args_kwargs.encode('utf-8')).hexdigest()
        cache_path = os.path.join(cache_dir, hash)
        if os.path.exists(cache_path) and not no_cache:
            with open(cache_path, 'r') as f:
                return json.load(f)
        else:
            if 'seed' in kwargs:
                del kwargs['seed']
            result = fn_to_cache(*args, **kwargs)
            with open(cache_path, 'w') as f:
                json.dump(result, f)
            return result

    return wrapped


_openai_chat_completion = _cached_function(openai.ChatCompletion.create, cache_dir='/iris/u/architsh/code/easy-rlhf/openai_cache/json')


def get_completion(prompt,
                   cache=True,
                   model='gpt-3.5-turbo-0301',
                   system_prompt='You are an evaluation system that compares the quality of the summaries of two automated summarization systems. Given a forum post and two generated summaries, you have to decide which more effectively summarizes the main points of the post.',
                   seed=None):
    c = _openai_chat_completion(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        no_cache=not cache,
        seed=seed,
    )
    if len(c['choices']) == 1:
        return c['choices'][0]['message']['content']
    else:
        return [c_['message']['content'] for c_ in c['choices']]

# %%
re_rank_n = [2, 4, 8, 16, 32, 64, 128, 256]
temperatures = [0.01, 0.25, 0.5, 0.75, 1.0]

for n in re_rank_n:
    response_map[f'rerank_{n}'] = {}
    for temp in temperatures:
        response_map[f'rerank_{n}'][temp] = []

base_path = '/iris/u/architsh/code/easy-rlhf/rerank_samples'
num_samples = 256
for temp in temperatures:
    response_path = os.path.join(base_path, f'tldr_gptj_sft_temp{temp}_nsamples{num_samples}.json')
    score_path = os.path.join(base_path, f'scores/tldr_gptj_sft_temp{temp}_nsamples{num_samples}.json')
    with open(response_path, 'r') as f:
        responses = json.load(f)
    with open(score_path, 'r') as f:
        scores = json.load(f)
    assert len(responses) == len(scores)
    assert len(responses) % num_samples == 0
    for idx in range(0, len(responses), num_samples):
        for n in re_rank_n:
            best_idx = np.argmax(scores[idx:idx+n])
            best_reponse = responses[idx + best_idx]
            response_map[f'rerank_{n}'][temp].append(best_reponse)

def get_reference_completions(n):
    random.seed(0)
    dataset = datasets.load_dataset("CarperAI/openai_summarize_tldr", cache_dir=cache_dir)['test']
    completions = [s.strip() for s in dataset['label']]

    random.shuffle(completions)
    return completions[:n]

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--temperatures', type=str, default='0.01,0.25,0.5,0.75,1.0')
parser.add_argument('--ranks', type=str, default='2,4,8,16,32,64,128,256')
parser.add_argument('--use_test_labels', action='store_true')
args = parser.parse_args()

eval_list = []
temps = [float(t) for t in args.temperatures.split(',')]
# ordering by importance
eval_methods = [
    'ppo',
    'dpo_longer',
    'fm_total',
    'sft',
    'base',
]
for key in eval_methods:
    for temp in temps:
        eval_list.append((key, temp))

for n in args.ranks.split(','):
    for temp in temps:
        eval_list.append((f'rerank_{n}', temp))

eval_list = [('dpo_longer', 0.25), ('ppo', 1.0), ('sft', 0.25)]

print(eval_list)
# final dict with all winrates
final_winrate = {}
reference_completions = get_reference_completions(256)
# %%
api_seed = 0
model = 'gpt-4-0314'

if args.use_test_labels:
    print('using test labels as reference')
    model_b = 'test_labels'
    model_b_responses = reference_completions
    folder_name  = 'results_new_ref_prompt'
else:
    print('using ppo @ 0.01 completions as reference')
    model_b = 'ppo_temp0.01'
    model_b_responses = response_map['ppo'][0.01]
    folder_name  = 'results_new_prompt'

for eval_name in eval_list:
    random.seed(0)
    print(f'evaluating {eval_name}\n')
    results = defaultdict(lambda: defaultdict(list))
    model_a = f'{eval_name[0]}_temp{eval_name[1]}'
    model_a_responses = response_map[eval_name[0]][eval_name[1]]

    results_key = f'{model_a}_{model_b}_{api_seed}'
    for idx, (dpo, sft) in enumerate(zip(model_a_responses, model_b_responses)):
        request, shuffled = get_rating_prompt(dpo, sft)
        with ThreadPoolExecutor(max_workers=1) as executor:
            timeout = 10.0
            while True:
                try:
                    future = executor.submit(get_completion, request, model=model, seed=api_seed)  # starts executing your function
                    judgment = future.result(timeout=timeout).strip(' .')  # 10 is the number of seconds before timeout
                    break
                except TimeoutError as e:
                    print(f"Function execution took longer than {timeout} seconds and was interrupted.")
                    # timeout += 5.0
                    time.sleep(2)
                except Exception as e:
                    print(e)
                    print('Retrying after 2 seconds...')
                    time.sleep(2)
                    continue

        judgment_letter = judgment[-1]
        justification = judgment.split('\n')[0]

        results[results_key]['request'].append(request)
        results[results_key]['letters'].append(judgment_letter)
        results[results_key]['shuffled'].append(shuffled)
        results[results_key]['judgments'].append(judgment)
        results[results_key]['justifications'].append(justification)
        valid = judgment_letter in ['A', 'B']
        results[results_key]['valid'].append(valid)

        if not valid:
            print('invalid judgment')
            print(judgment)
            results[results_key]['method_choices'].append('INVALID')
            continue

        b_win = (shuffled and judgment_letter == 'A') or (not shuffled and judgment_letter == 'B')
        method_choice = model_b if b_win else model_a
        results[results_key]['method_choices'].append(method_choice)

        n_a_win = results[results_key]['method_choices'].count(model_a)
        n_valid = sum(results[results_key]['valid'])
        winrate = n_a_win / n_valid
        stderr = np.sqrt(winrate * (1 - winrate) / n_valid)
        print(f'{idx} {method_choice} {winrate:0.3f} ({stderr:0.3f}) {judgment_letter} {justification}')

    print(f'{model_a} winrate: {winrate:0.3f} ({stderr:0.3f})')
    print(f'Valid judgments: {n_valid}/{len(model_a_responses)} prompts')
    results[results_key]['final_winrate'] = winrate
    results[results_key]['final_stderr'] = stderr
    with open(os.path.join(output_dir, folder_name, model_a + '.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    final_winrate[results_key] = (winrate, stderr)

with open(os.path.join(output_dir, folder_name, 'final_winrate.json'), 'w') as f:
    json.dump(final_winrate, f, indent=2)


