"""把被中途停止的 probe_transfer 运行，从日志抢救成机器可读的张量文件。

为什么需要它：probe_transfer.py 只在**七行全部跑完**之后才写 transfer.json。
种子 23 的复现在第 6 行（generic）完成前被停止，已经完成的六行如果只留在
日志里，下游每个分析脚本都得各自去解析日志，容易各自解析出不同的数。

输出刻意**不叫** transfer.json，而是 transfer-partial.json，并带 `complete: false`
和 `rows_complete`。不完整的运行绝不能伪装成完整的：下游任何按完整张量假设做的
聚合（比如 analysis/allocation_sweep.py 要求完整七行的张量）必须能一眼看出这份数据缺行。
"""
import argparse, json, re
from pathlib import Path

NAMES = ["Integral", "Differential", "Abstract", "Linear", "Statistics",
         "algorithmic", "generic"]
K = len(NAMES)


def parse(text):
    """逐行抓「能力差 / 翻转」两段。行号来自结果表首列，不靠出现顺序推断。"""
    rows, current = [], None
    for line in text.splitlines():
        header = re.match(r"^ +([0-6]) +(-?[\d.]+(?: +-?[\d.]+){%d})" % (K - 1), line)
        if header:
            current = {"source": int(header.group(1)),
                       "transfer": [float(x) for x in header.group(2).split()]}
            continue
        if current is None:
            continue
        if "能力差" in line:
            current["delta"] = [float(x) for x in re.findall(r"[-+][\d.]+", line)]
        elif "翻转" in line:
            current["flips"] = [{"gained": int(g), "lost": int(l)}
                                for g, l in re.findall(r"\+(\d+)/-(\d+)", line)]
            current["pvalues"] = [float(p) for p in re.findall(r"p=([\d.e-]+)", line)]
            rows.append(current)
            current = None
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", nargs="+", type=float,
                    default=[0.3750, 0.4625, 0.3500, 0.3875, 0.2000, 0.3625, 0.2000])
    ap.add_argument("--reason", default="run stopped before all rows finished; "
                                        "completed rows were recovered from the run log")
    args = ap.parse_args()

    rows = parse(Path(args.log).read_text(encoding="utf-8", errors="replace"))
    for row in rows:
        row["before"] = args.baseline
    missing = [NAMES[j] for j in range(K) if j not in {r["source"] for r in rows}]

    payload = {
        "complete": len(missing) == 0,
        "rows_complete": sorted(r["source"] for r in rows),
        "rows_missing": missing,
        "stopped_reason": args.reason,
        "source_log": args.log,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"抢救 {len(rows)}/{K} 行 → {args.out}")
    print(f"  已完成: {[NAMES[i] for i in payload['rows_complete']]}")
    print(f"  缺失  : {missing or '无'}")
    for row in rows:
        total = sum(f["gained"] + f["lost"] for f in row["flips"])
        print(f"  行 {row['source']} {NAMES[row['source']]:<12} 翻转合计 {total}")


if __name__ == "__main__":
    main()
