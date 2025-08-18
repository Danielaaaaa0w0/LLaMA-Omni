import json
from jiwer import cer
def compute_corpus_cer(jsonl_path):
    refs = []
    hyps = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            refs.append(data["answer"])
            hyps.append(data["prediction"])
    ref_all = "".join(refs)
    hyp_all = "".join(hyps)
    overall_cer = cer(ref_all, hyp_all)
    print(f"Corpus CER: {overall_cer:.4f}")
# 用法範例
compute_corpus_cer("/mnt/md0/user_yuze0w0/dataset/taigi_100h/data_cn/answer.json")