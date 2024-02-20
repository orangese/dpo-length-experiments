import os
import json
import argparse

def parse_model_id(m_id, ds="ultrafeedback"):
    try:
        model, ds, beta, alpha = m_id.split("-")
    except:
        model, beta, alpha = m_id.split("-")

    alpha = float("0." + alpha.replace("a", ""))
    beta = float("0." + beta.replace("b", ""))
    return model, ds, beta, alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mt_bench", type=str, help="data/mt_bench directory", required=True)
    parser.add_argument("--sample_dir", type=str, help="target sample directory", required=True)
    parser.add_argument("--step", type=int, help="last training step", default=59904)
    args = parser.parse_args()

    os.makedirs(args.sample_dir, exist_ok=True)
    with open(os.path.join(args.mt_bench, "question.jsonl")) as f:
        q_list = [json.loads(line) for line in f.readlines()]
    questions = {}
    for question in q_list:
        for turn, prompt in enumerate(question["turns"]):
            questions[(question["question_id"], turn)] = prompt
    print(f"loaded {len(questions)} questions (multi-turn flattened)")

    for mfile in os.listdir(os.path.join(args.mt_bench, "model_answer")):
        with open(os.path.join(args.mt_bench, "model_answer", mfile)) as f:
            a_list = [json.loads(line) for line in f.readlines()]

        pairs = {}
        lens = []
        for answer in a_list:
            model, ds, beta, alpha = parse_model_id(answer["model_id"])
            choice = answer["choices"][0]
            assert len(answer["choices"]) == 1
            for i, response in enumerate(choice["turns"]):
                prompt = questions[(answer["question_id"], i)]
                pairs[prompt] = prompt + response
                lens.append(len(response))

        fname = f"{model}__{ds}__b{beta}__a{alpha}__s{args.step}__samples.json"
        path = os.path.join(args.sample_dir, fname)
        with open(path, "w+") as f:
            json.dump(pairs, f)
        print(f"wrote to '{path}', mean char len: {round(sum(lens) / len(lens), 2)}")

