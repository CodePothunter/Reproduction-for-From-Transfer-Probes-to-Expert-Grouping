"""组合效用与配对收益；所有 competence 值使用 [0, 1] 单位。

competence 记为 c[i][j]，i 是模型，j 是评估来源：

    硬路由效用  U      = Σ_j w_j · max_i c_ij
    软化效用    Ũ_τ    = Σ_j w_j · τ log Σ_i exp(c_ij/τ)
    夹逼关系    U ≤ Ũ_τ ≤ U + τ log M
    配对收益    g      = (Ũ_τ(Θ⁺) − Ũ_τ(Θ⁻)) / n

本包里唯一的调用方是 Experiment.probe：它用 paired_return 给出每个探针格的
portfolio_return 字段。服务时是硬路由，每个请求只由一个专家处理。
本模块不做任何模型调用，只对已经算好的 competence 表做运算。
"""
import math


def _finite(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限实数")
    return float(value)


def _temperature(tau):
    """tau=None 表示硬路由；否则必须是正的有限温度。"""
    if tau is None:
        return None
    tau = _finite(tau, "tau")
    if tau <= 0:
        raise ValueError("tau 必须大于零；硬路由请传 None")
    return tau


def _checked(competence, weights):
    """校验 competence 表与部署权重，返回 (模型数 M, 来源数 K)。"""
    if not competence or not isinstance(competence, (list, tuple)):
        raise ValueError("competence 必须是非空的 M×K 矩阵")
    models = len(competence)
    sources = len(competence[0]) if competence[0] is not None else 0
    if not sources or any(len(row) != sources for row in competence):
        raise ValueError("competence 不能为空或参差不齐")
    for row in competence:
        for value in row:
            if not 0.0 <= _finite(value, "competence") <= 1.0:
                raise ValueError("competence 必须落在 [0, 1]")
    if len(weights) != sources:
        raise ValueError(f"weights 长度需为 {sources}，与 competence 的列数一致")
    total = 0.0
    for weight in weights:
        if _finite(weight, "weight") < 0:
            raise ValueError("weights 不能为负")
        total += weight
    # τ log M 上界以 Σ_j w_j = 1 为前提；不归一化会让夹逼失效。
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("weights 之和必须为 1")
    return models, sources


def _soft_column(column, tau):
    """τ log Σ_i exp(c_i/τ)，按最大值平移避免小 τ 时溢出。

    结果恒等于 max_i c_i + τ log Σ_i exp((c_i − max)/τ)，其中对数项落在
    [0, log M]，正好给出上面的夹逼。
    """
    top = max(column)
    return top + tau * math.log(sum(math.exp((value - top) / tau) for value in column))


def portfolio_utility(competence, weights, tau=None):
    """tau=None 返回硬路由效用；tau>0 返回 log-sum-exp 效用。

    competence[i][j] 是模型 i 在来源 j 上的能力，weights[j] 是部署重要性。
    硬路由对应实际服务（每个来源只由最强专家承担）。
    """
    tau = _temperature(tau)
    _checked(competence, weights)
    columns = zip(*competence)
    if tau is None:
        return sum(weight * max(column) for weight, column in zip(weights, columns))
    return sum(weight * _soft_column(column, tau) for weight, column in zip(weights, columns))


def paired_return(before, after, tokens, weights, tau):
    """返回同一个评估 bank 上的组合效用变化/token；tokens 必须大于零。

    调用方必须保证 before/after 来自同一批评估样本、同一套 weights 与 tau、
    同一解码协议，且中间只隔了这一段来源纯净的训练，否则这个比值不能解释为
    该 cell 的边际收益。

    正负号有意义：训练后组合效用下降会得到负收益，不做截断。
    """
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("tokens 必须是大于零的整数")
    if len(before) != len(after) or any(len(a) != len(b) for a, b in zip(before, after)):
        raise ValueError("before 与 after 的形状必须一致")
    delta = portfolio_utility(after, weights, tau) - portfolio_utility(before, weights, tau)
    return delta / tokens
