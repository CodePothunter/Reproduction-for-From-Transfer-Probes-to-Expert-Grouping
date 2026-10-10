"""建 7 来源 manifest：迁移探针、热身与候选训练共用的语料。

为什么是这七个来源：要检验「按标签分组」是否会分错，需要**标签说不清**的
来源对。DeepMath 的主题是层级的：Integral / Differential Calculus 同属
Calculus，Abstract / Linear Algebra 同属 Algebra。任何按标签的划分都必须把
每一对归一组；如果同一对之间其实没有迁移，标签划分就是错的。

来源设计（七个，层级上互斥，single_source 才不会判 ambiguous）：

    0  Integral Calculus      ┐ 标签说这两个同属 Calculus
    1  Differential Calculus  ┘
    2  Abstract Algebra       ┐ 标签说这两个同属 Algebra
    3  Linear Algebra         ┘
    4  Statistics               独立分支（Applied Mathematics 下）
    5  algorithmic              OpenCodeInstruct domain
    6  generic                  OpenCodeInstruct domain

**MBPP+ 不做训练来源**，只留作外部评估（整批指到来源 5）。它整批只有约
24.5 万 completion token，在 32768 的块尺度下只够 8 块，一个区间就会用尽。

⚠ 只建 manifest，不加载模型、不跑评估。纯 CPU，但需要联网下载数据集与
tokenizer。--salt 的缺省值决定切分，换了它就是另一份 manifest。

⚠ 本报告用的 manifest 取了三个 OpenCodeInstruct 分片的全部行（不截断），即
`--code-pool 0`；这样重建的 manifest 与 results/protocol/manifest_summary.json 里的
sha256 逐位一致。--code-pool 的缺省值 120000 会截断代码池，得到切分完全不同的另一份
manifest，所以复现时必须显式传 --code-pool 0。
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEEPMATH_SOURCES = {
    "Integral Calculus": 0,
    "Differential Calculus": 1,
    "Abstract Algebra": 2,
    "Linear Algebra": 3,
    "Statistics": 4,
}
CODE_SOURCES = {"algorithmic": 5, "generic": 6}
# 五个键一个都不能少：split_records 要求键集恰好是 SPLITS。换了比例就换了
# 切分，与既有 manifest 不再可比。
SPLIT_RATIOS = {"train": 0.66, "feedback": 0.12, "route": 0.06,
                "audit": 0.10, "test": 0.06}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    p.add_argument("--cache", default="data/raw/hf")
    p.add_argument("--pool", type=int, default=103000,
                   help="DeepMath 取多少条原始记录；七来源要分五份，别取太少")
    p.add_argument("--code-pool", type=int, default=120000,
                   help="OpenCodeInstruct 打散后取多少条原始记录；0 表示全部。"
                        "本报告的 manifest 用的是 0（见模块说明）")
    p.add_argument("--code-shards", type=int, nargs="+", default=[45, 48, 49])
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--salt", default="7src-2026-09-18")
    p.add_argument("--out", default="runs/manifest-7src")
    a = p.parse_args()

    from transformers import AutoTokenizer
    from lwts.datasets import (DEEPMATH, MBPPPLUS, OPENCODEINSTRUCT,
                               deepmath, mbppplus, opencodeinstruct,
                               prepare_records, split_records, validate_manifest)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True)

    def prepare(label, module, dataset, split, limit, sources, shuffle=False,
                drop_conflicts=False, **kw):
        started = time.time()
        data = module.load(cache_dir=a.cache, **kw)[split]
        if drop_conflicts:
            # 同一题面挂到两个候选主题上会让 prepare_records 硬中止。实测只有
            # 0.03%（16/51039，全是 Integral ↔ Differential Calculus），按
            # single_source 的既有规则排除，不任选其一。
            before = data.num_rows
            data = module.drop_topic_conflicts(data, sources)
            print(f"  {label:18} 清除跨主题冲突 {before - data.num_rows} 行", flush=True)
        if shuffle:
            # OpenCodeInstruct 的 domain 在分片内是**连续块**（实测 algorithmic
            # 位于分片 45 的 [41456, 65703]），不先打散再截断会得到零条
            # algorithmic。种子由 salt 经 sha256 导出，跨进程可复现。
            digest = hashlib.sha256(a.salt.encode("utf-8")).digest()
            data = data.shuffle(seed=int.from_bytes(digest[:4], "big"))
        if limit:
            data = data.select(range(min(limit, data.num_rows)))
        stats = {}
        rows = prepare_records(list(data), tokenizer, dataset=dataset, revision="main",
                               original_split=split, max_tokens=a.seq_len,
                               report=stats,
                               **({"topic_sources": sources} if sources else {}))
        print(f"  {label:18} {split:5} 原始 {stats['input']:6} → 保留 {stats['kept']:6}"
              f"  ({time.time() - started:.0f}s)", flush=True)
        return rows

    print("=== 1. 训练语料 ===", flush=True)
    trainable = (
        prepare("DeepMath", deepmath, DEEPMATH, "train", a.pool, DEEPMATH_SOURCES,
                drop_conflicts=True)
        + prepare("OpenCodeInstruct", opencodeinstruct, OPENCODEINSTRUCT, "train",
                  a.code_pool, CODE_SOURCES, shuffle=True,
                  shards=tuple(a.code_shards)))
    records = split_records(trainable, SPLIT_RATIOS, salt=a.salt)

    print("\n=== 2. 外部 test（不进训练）===", flush=True)
    # ⚠ **数学侧没有外部 test。** 常见的外部数学基准（例如 MATH-500）用的是
    # 标准 MATH 分类（Algebra / Intermediate Algebra / Precalculus / ...），与
    # 本设计的五个细分主题没有一个能诚实对应——那里的 "Algebra" 是中学代数，
    # 不是 Abstract Algebra。硬映射等于编数据。因此数学侧的能力只由 DeepMath
    # 自身的 held-out 切分承担，结论要按「内部 held-out」而不是「外部泛化」来陈述。
    #
    # MBPP+：它的 parse 直接读 topic_sources["code"]，整批指到一个代码来源。
    # 它评的是代码侧的**整体**能力，不是来源 5 专属。
    for row in prepare("MBPP+", mbppplus, MBPPPLUS, "test", None, {"code": 5}):
        records.append(dict(row, split="test"))

    manifest = {"schema_version": 1, "status": "pilot_not_frozen",
                "dataset_revisions": {DEEPMATH: "main", OPENCODEINSTRUCT: "main",
                                      MBPPPLUS: "main"},
                "code_shards": list(a.code_shards),
                "tokenizer": {"id": a.model, "revision": "main"},
                "sources": list(range(7)), "split_salt": a.salt, "records": records}

    print("\n=== 3. 校验与统计 ===", flush=True)
    audit = validate_manifest(manifest)
    print(f"  记录 {audit['records']}  题组 {audit['groups']}  状态 {audit['status']}")
    names = {**{v: k for k, v in DEEPMATH_SOURCES.items()},
             **{v: k for k, v in CODE_SOURCES.items()}}
    for split in ("train", "feedback", "audit"):
        counts = audit["counts"][split]
        print(f"  {split}")
        for k in range(7):
            c = counts[str(k)]
            blocks = c["tokens"] // 32768
            print(f"    源 {k} {names[k]:<24} {c['records']:>6} 条  "
                  f"{c['tokens']:>10} token" + (f"  ≈ {blocks:>5} 块" if split == "train" else ""))

    target = out / "manifest.jsonl"
    with target.open("w", encoding="utf-8") as fh:
        head = {k: v for k, v in manifest.items() if k != "records"}
        fh.write(json.dumps(head, ensure_ascii=False) + "\n")
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n写入 {target}（{target.stat().st_size / 2**20:.0f} MB）")


if __name__ == "__main__":
    main()
