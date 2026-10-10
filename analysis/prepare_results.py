"""从原始运行目录生成 results/ 下随包的结果文件（为本报告新写的清理工具）。

原始运行目录不随包（含模型检查点、全文题面与本机路径）；本脚本记录清理时做了哪些改动，
数据部分一律原样保留：

- probes/s17_transfer.json：逐字节复制 state0 探针（种子 17）的 transfer.json。
- probes/s23_transfer_partial.json：种子 23 的部分张量（6/7 行，由日志抢救）。只改两个
  元数据字段：stopped_reason 换成中性说明，source_log 换成"日志不随包"的说明；rows 原样。
- candidates/C{0..3}-eval-s200{1,2}.json：8 份 test 评价。
  * test_ids 原为 manifest 记录 id（"数据集名:题面全文"），换成 sha256(id) 的前 16 位十六进制，
    顺序不变；重建 manifest 后对每条 id 求同样的哈希即可对上。
  * 去掉 checkpoint 字段（本机绝对路径）。其余字段原样。
- candidates/training_summary.json：各候选训练区间（round 1）的块矩阵与 token 数（取自
  C*/logs.json），路由（取自评价文件），训练与评价的墙钟秒数（取自运行台账）；以及共同起点 W
  那一轮（round 0）的均匀块矩阵与 token 数。不带 W 日志里与本报告无关的诊断字段。
- protocol/decoding_variance_summary.json：解码方差拆分与 k 的选择，只含汇总数字；
  原始输入（一个更早的四专家组合的逐题分数）不随包。可选 --decoding-repeat 补记重复采样
  的种子与 algorithmic 那 80 题的数据集构成（只记计数，不记题目）。
- protocol/manifest_summary.json（可选 --manifest）：数据 manifest 的 sha256、表头里的数据集
  与切分盐、逐来源逐切分的题数、test 切分的数据集构成、排序后前 400 题的数据集构成，并核对
  8 份评价文件的哈希 test_ids 恰为排序后前 400 条 test id 的哈希。manifest 本身不随包。

用法：
    python analysis/prepare_results.py --s17 <s17 transfer.json> --s23 <s23 部分张量 json> \
        --candidates <候选运行目录：C*/logs.json、C*-eval-s*/test_extension.json、台账> \
        --decoding <解码方差汇总 json> [--decoding-repeat <重复采样评价 json>] \
        [--manifest <manifest.jsonl>] [--out results]

运行台账先找 ledger.txt（pipeline/run_candidates.sh 与 run_eval.sh 写的文件），没有再找
ledger.log（产生本报告结果的原始运行目录里，台账用的是这个名字）。
"""
import argparse, collections, hashlib, json, re, shutil
from pathlib import Path

CANDS = ("C0", "C1", "C2", "C3")
SEEDS = (2001, 2002)
NAMES = ["Integral", "Differential", "Abstract", "Linear", "Statistics",
         "algorithmic", "generic"]

S23_STOPPED = ("run stopped before the last row (generic) finished; the six completed rows "
               "were recovered from the run log")
S23_SOURCE = "s23 probe run log (not included in this package)"

ROUND_KEYS = ("blocks", "allocation", "processed_tokens", "completion_tokens",
              "block_tokens_realized")


def hash_id(item_id):
    return hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:16]


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_eval(src):
    d = json.loads(src.read_text(encoding="utf-8"))
    out = {}
    for key, value in d.items():
        if key == "checkpoint":
            continue
        if key == "test_ids":
            value = {k: [hash_id(x) for x in ids] for k, ids in value.items()}
            for k, ids in value.items():
                if len(set(ids)) != len(ids):
                    raise SystemExit(f"{src}: 来源 {k} 的哈希 id 有碰撞")
        out[key] = value
    return out


def ledger_seconds(text):
    train, evals = {}, {}
    for line in text.splitlines():
        m = re.search(r"(C\d) 训练 退出码 0 用时 (\d+)s", line)
        if m:
            train[m.group(1)] = int(m.group(2))
        m = re.search(r"(C\d) 评价 种子 (\d+) 退出码 0 用时 (\d+)s", line)
        if m:
            evals.setdefault(m.group(1), {})[m.group(2)] = int(m.group(3))
    return train, evals


def find_ledger(run):
    """运行台账：本包的启动脚本写 ledger.txt；原始运行目录里的台账名为 ledger.log。"""
    for name in ("ledger.txt", "ledger.log"):
        if (run / name).is_file():
            return run / name
    raise SystemExit(f"{run} 下没有运行台账（ledger.txt 或 ledger.log）")


def training_summary(run, evals):
    logs = {c: json.loads((run / c / "logs.json").read_text(encoding="utf-8")) for c in CANDS}
    train_s, eval_s = ledger_seconds(find_ledger(run).read_text(encoding="utf-8"))
    rounds0 = []
    out = {"note": ("Per-candidate training interval (round 1, forked from the shared warm-up "
                    "state W) and the shared warm-up round (round 0). Blocks are rows = experts, "
                    "columns = sources; routing = expert serving each source at test time. "
                    "Wall-clock seconds are from the run ledger (shell timing, including model "
                    "loading)."),
           "sources": NAMES, "candidates": {}}
    for c in CANDS:
        by_round = {e["round"]: e for e in logs[c] if "blocks" in e}
        r1 = by_round[1]
        rounds0.append({k: by_round[0][k] for k in ROUND_KEYS})
        routing = [evals[(c, s)]["routing"] for s in SEEDS]
        assert routing[0] == routing[1]
        out["candidates"][c] = {**{k: r1[k] for k in ROUND_KEYS},
                                "routing": routing[0],
                                "train_wall_seconds": train_s[c],
                                "eval_wall_seconds": {str(s): eval_s[c][str(s)] for s in SEEDS},
                                "eval_seconds_in_process": {str(s): evals[(c, s)]["seconds"]
                                                            for s in SEEDS},
                                "eval_generations": {str(s): evals[(c, s)]["generated"]
                                                     for s in SEEDS}}
    assert all(r == rounds0[0] for r in rounds0), "四个候选继承的 W 那一轮不一致"
    out["warmup_W_round0"] = {**rounds0[0],
                              "note": "uniform warm-up interval shared by all candidates"}
    return out


def decoding_summary(src, repeat=None):
    d = json.loads(src.read_text(encoding="utf-8"))
    out = {
        "note": ("Design inputs for choosing k (samples per test item), measured before the "
                 "candidate comparison on four-expert portfolios of an earlier comparison, each "
                 "trained from the same warm-up state W for one interval: the decoding split uses "
                 "one of them, and the paired-difference variance uses two of them. Neither is "
                 "one of the candidates in this report. Only summary numbers are recorded here; "
                 "the per-item score files and the script that computed them are not included "
                 "in this package."),
        "decoding_split": {
            "items_per_source": 80,
            "description": ("two independent stochastic decodings of the same checkpoint on the "
                            "same 80 test items per source; decoding variance = mean((y1-y2)^2)/2; "
                            "total variance = mean of the two per-run item variances; between-item "
                            "variance = total - decoding"),
            "per_source": [{"source": r["source"],
                            "two_decoding_disagreement_rate": r["discordant"],
                            "decoding_variance": r["sigma_d2"],
                            "total_variance": r["sigma_t2"],
                            "between_item_variance": r["sigma_b2"],
                            "decoding_share": r["decoding_share"]} for r in d["rows"]],
        },
        "paired_se_extrapolation": {
            "items_per_source": 400,
            "description": ("SE of the paired difference in equal-weight utility between two "
                            "portfolios with k samples per item: V(k) = V_obs - 2*decoding_variance"
                            "*(1 - 1/k) per source, where V_obs is the per-item paired-difference "
                            "variance measured at k = 1 on 400 items per source between two "
                            "portfolios of the earlier comparison; SE(k) = sqrt(sum_j V_j(k)/n_j)/7; "
                            "MDE = 2.8*SE (80% power, two-sided alpha = 0.05)"),
            "by_k": {k: {"se": v["se"], "mde": v["mde"]} for k, v in d["k"].items()},
        },
        "delta_min": 0.02,
        "rule": "smallest k in {1, 2, 4} with MDE <= delta_min (k = 4 if none)",
        "k_chosen": d["k_chosen"],
    }
    if repeat is not None:
        r = json.loads(Path(repeat).read_text(encoding="utf-8"))
        split = out["decoding_split"]
        split["samples"] = {"first": "default CUDA generator state",
                            "second": f"CUDA seed {r['eval_seed']}"}
        split["items_by_dataset"] = {
            NAMES[int(k)]: dict(collections.Counter(x.split(":", 1)[0] for x in ids))
            for k, ids in r["test_ids"].items()}
    return out


def manifest_summary(path, evals):
    """manifest 的计数摘要；顺带核对评价文件的 test_ids。"""
    h = hashlib.sha256()
    counts = collections.defaultdict(collections.Counter)
    by_dataset = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    test_ids = collections.defaultdict(list)
    header = None
    with open(path, "rb") as f:
        for line in f:
            h.update(line)
            rec = json.loads(line)
            if header is None:
                header = rec
                continue
            counts[rec["source"]][rec["split"]] += 1
            by_dataset[rec["source"]][rec["split"]][rec["id"].split(":", 1)[0]] += 1
            if rec["split"] == "test":
                test_ids[rec["source"]].append(rec["id"])
    n = 400
    first = {k: sorted(v)[:n] for k, v in test_ids.items()}
    match = all(ev["test_ids"][str(k)] == [hash_id(x) for x in first[k]]
                for ev in evals.values() for k in range(len(NAMES)))
    dataset = lambda ids: dict(collections.Counter(x.split(":", 1)[0] for x in ids))  # noqa: E731
    splits = sorted({s for c in counts.values() for s in c})
    all_counts = collections.Counter()
    for k, c in counts.items():
        for sp, v in c.items():
            all_counts[sp] += v
    total = sum(all_counts.values())
    return {
        "note": ("Counts from the data manifest built by pipeline/build_manifest_7src.py. The "
                 "manifest itself (problem text and reference solutions) is not included."),
        "sha256": h.hexdigest(),
        "header": {k: header[k] for k in ("dataset_revisions", "code_shards", "tokenizer",
                                          "sources", "split_salt") if k in header},
        "sources": NAMES,
        "split_counts": {NAMES[k]: {sp: counts[k][sp] for sp in splits} for k in sorted(counts)},
        "split_fraction_all_records": {sp: all_counts[sp] / total for sp in splits},
        "split_counts_by_dataset": {NAMES[k]: {sp: dict(by_dataset[k][sp]) for sp in splits}
                                    for k in sorted(by_dataset)},
        "test_by_dataset": {NAMES[k]: dataset(v) for k, v in sorted(test_ids.items())},
        "first_400_test_sorted_by_id_by_dataset": {NAMES[k]: dataset(v)
                                                   for k, v in sorted(first.items())},
        "min_test_count": min(len(v) for v in test_ids.values()),
        "eval_test_ids_equal_hashed_first_400_sorted": match,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s17", required=True, help="种子 17 的探针 transfer.json（不随包的原件）")
    ap.add_argument("--s23", required=True, help="种子 23 的部分张量 json（不随包的原件）")
    ap.add_argument("--candidates", required=True, help="候选运行目录（不随包）")
    ap.add_argument("--decoding", required=True, help="解码方差汇总 json（不随包的原件）")
    ap.add_argument("--decoding-repeat", help="解码方差拆分中第二次采样的评价 json（可选，不随包）")
    ap.add_argument("--manifest", help="数据 manifest.jsonl（可选，不随包）")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)

    (out / "probes").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.s17, out / "probes" / "s17_transfer.json")
    s23 = json.loads(Path(a.s23).read_text(encoding="utf-8"))
    s23["stopped_reason"] = S23_STOPPED
    s23["source_log"] = S23_SOURCE
    dump(out / "probes" / "s23_transfer_partial.json", s23)

    run = Path(a.candidates)
    evals = {}
    for c in CANDS:
        for s in SEEDS:
            evals[(c, s)] = clean_eval(run / f"{c}-eval-s{s}" / "test_extension.json")
            dump(out / "candidates" / f"{c}-eval-s{s}.json", evals[(c, s)])
    dump(out / "candidates" / "training_summary.json", training_summary(run, evals))
    dump(out / "protocol" / "decoding_variance_summary.json",
         decoding_summary(Path(a.decoding), a.decoding_repeat))
    if a.manifest:
        summary = manifest_summary(a.manifest, evals)
        assert summary["eval_test_ids_equal_hashed_first_400_sorted"], "评价 test_ids 与 manifest 不符"
        dump(out / "protocol" / "manifest_summary.json", summary)
    for p in sorted(out.rglob("*.json")):
        print(f"{p.stat().st_size:>8}  {p}")


if __name__ == "__main__":
    main()
