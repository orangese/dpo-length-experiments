import os

m_id = "mistral7b-{beta}-beam{beam}-ngram{ngram}"
nm_id = "zephyr7b-{beta}-beam{beam}{extra}"
k = "mistral7b-"

for f in os.listdir():
    if not f.endswith(".sh"):
        continue
    lines = []

    with open(f, "r") as b:
        try:
            beam, ngram = f.split("_")
        except:
            print("skipped", f)
            continue

        beam = int(beam.replace("beam", ""))
        ngram = int(ngram.replace("ngram", "").replace(".sh", ""))

        for l in b:
            lines.append(l)
            if "mistral-7b-sft-beta" in l:
                lines[-1] = l.replace("mistral-7b-sft-beta", "alignment-handbook/zephyr-7b-sft-full")
            if "HuggingFaceH4/alignment-handbook/zephyr-7b-sft-full" in l:
                lines[-1] = l.replace("HuggingFaceH4/alignment-handbook/zephyr-7b-sft-full", "alignment-handbook/zephyr-7b-sft-full")

            l = lines[-1]
            i = l.find(k)
            if i != -1:
                beta = l[i + len(k): i + l[i:].find("-beam")]
                model_id = m_id.format(beta=beta, beam=beam, ngram=ngram)
                
                beta = f"beta{beta[1:]}" if beta != "sft" else "sft"
                extra = "" if ngram == 0 else f"-ngram{ngram}"
                new_model_id = nm_id.format(beta=beta, beam=beam, extra=extra)
                lines[-1] = l.replace(model_id, new_model_id)

    print("".join(lines))
    with open(f, "w") as d:
        for line in lines:
            d.write(line)
