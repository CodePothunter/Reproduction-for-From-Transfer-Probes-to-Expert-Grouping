"""C0（标签参照）的块矩阵由标签树与预算边际唯一确定。（为本报告新写）

两步，都只用随包代码：

1. 树切分唯一：7 个来源的标签树在 M=4 时只有一种「4 个互不包含、覆盖全部叶子」
   的切分——{Algebra, Calculus, Statistics, code}（analysis/allocation_sweep.py 的
   covering_antichains）。所以「在开发数据上挑得分最高的切分」这条规则在 M=4 时
   没有选择余地。
2. 填充唯一：把切分写成「来源 → 专家」映射 [1,1,0,0,3,2,2]，参考矩阵在命中格取 1、
   其余取 1e-6，用 lwts.allocation.mirror（η=0，即只做 Sinkhorn 配平）配到行边际
   1/4、列边际 1/7，再用 lwts.allocation.integer_counts 取整到每专家 14 块、
   每来源 8 块，得到的正是候选训练执行的 C0。
"""
import sys
from pathlib import Path

from lwts.allocation import check_marginals, integer_counts, mirror
from lwts.experiment import _apportion

ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"

M, K = 4, 7
BLOCKS = 1835008 // 32768            # tokens_per_interval // block_tokens = 56
SEMANTIC_MAP = [1, 1, 0, 0, 3, 2, 2]  # 来源 k → 专家
C0 = [[0, 0, 7, 7, 0, 0, 0],
      [7, 7, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 7, 7],
      [1, 1, 1, 1, 8, 1, 1]]


def _sweep():
    if str(ANALYSIS) not in sys.path:
        sys.path.insert(0, str(ANALYSIS))
    import allocation_sweep
    return allocation_sweep


def test_tree_has_a_single_four_group_cut():
    sweep = _sweep()
    cuts = sweep.covering_antichains(M)
    assert cuts == [("algebra", "calculus", "code", "leaf4")]
    groups = [sorted(sweep.TREE[g]) for g in cuts[0]]
    assert groups == [[2, 3], [0, 1], [5, 6], [4]]
    # 映射与切分一致：同组来源归同一个专家，不同组归不同专家
    owners = {tuple(g): {SEMANTIC_MAP[k] for k in g} for g in groups}
    assert all(len(v) == 1 for v in owners.values())
    assert len({next(iter(v)) for v in owners.values()}) == M


def test_cut_counts_for_other_portfolio_sizes():
    """M=2..7 的切分数为 1、1、1、3、3、1：只有 M=4 时 C0 由树唯一确定这件事
    不是一般规律，而是本树在 M=4 的性质。"""
    sweep = _sweep()
    assert [len(sweep.covering_antichains(m)) for m in range(2, 8)] == [1, 1, 1, 3, 3, 1]


def test_c0_block_matrix_follows_from_mirror_and_integer_counts():
    r = [1.0 / M] * M
    p = [1.0 / K] * K
    reference = [[1e-6] * K for _ in range(M)]
    for k, i in enumerate(SEMANTIC_MAP):
        reference[i][k] = 1.0
    q = mirror([[0.0] * K for _ in range(M)], reference, r, p, eta=0.0)
    check_marginals(q, r, p, tolerance=1e-7)
    rows, cols = _apportion(r, BLOCKS), _apportion(p, BLOCKS)
    assert rows == [14] * M and cols == [8] * K
    assert integer_counts(q, rows, cols) == C0


def test_c0_is_what_the_fixed_method_executes():
    """把 C0 作为 fixed 块矩阵交给 Experiment，它必须原样执行（integer_counts 不改动）。"""
    q = [[v / BLOCKS for v in row] for row in C0]
    assert integer_counts(q, [14] * M, [8] * K) == C0
