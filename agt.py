#!/usr/bin/env python3
"""
Virasoro 4-pt block vs AGT/Nekrasov (SU(2), N_f=4).

Reduced series:
    1 + sum_{k>=1} c_k q^k

Insertions at (0,q,1,∞) with momenta (beta0, alpha0, alpha1, beta2).
Internal momentum beta1 = a + (e1+e2)/2.

By default (numeric, non-symbolic), if you do NOT provide explicit params,
the script samples random rational params, with no fixed seed.
"""

import argparse
import json
import random
from functools import lru_cache

import sympy as sp
from sympy.matrices.common import NonInvertibleMatrixError


# ============================================================
# Parsing helpers
# ============================================================
def sym(x: str):
    return sp.sympify(x, evaluate=True)

def parse_seed(s):
    if s is None:
        return None
    s_low = str(s).strip().lower()
    if s_low in ("none", "null"):
        return None
    return int(s)

def finite(expr):
    return not expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo)


# ============================================================
# Integer partitions (PBW labels)
# ============================================================
@lru_cache(None)
def partitions_int(n: int, max_part=None):
    if max_part is None or max_part > n:
        max_part = n
    if n == 0:
        return [()]
    out = []
    for k in range(min(max_part, n), 0, -1):
        for rest in partitions_int(n - k, k):
            out.append((k,) + rest)
    return out


# ============================================================
# Virasoro Shapovalov via commutators
# ============================================================
def vir_comm(m, n, c):
    out = {(m + n): sp.Integer(m - n)}
    if m + n == 0:
        out["C"] = (c / 12) * sp.Integer(m) * (sp.Integer(m) ** 2 - 1)
    return out

@lru_cache(None)
def reduce_word_on_ket(word, Delta, c):
    terms = {word: sp.Integer(1)}
    changed = True
    while changed:
        changed = False
        new = {}
        for w, coeff in terms.items():
            w = list(w)

            # rightmost positive kills |Delta>
            if w and w[-1] > 0:
                continue

            # eliminate L_0: L_0 L_-n = L_-n (L_0 + n)
            if 0 in w:
                i = w.index(0)
                shift = 0
                for r in w[i + 1:]:
                    shift -= r
                w2 = tuple(w[:i] + w[i + 1:])
                new[w2] = new.get(w2, 0) + coeff * (Delta + shift)
                changed = True
                continue

            # commute adjacent inversion (+ then -)
            inv = None
            for i in range(len(w) - 1):
                if w[i] > 0 and w[i + 1] < 0:
                    inv = i
                    break

            if inv is None:
                new[tuple(w)] = new.get(tuple(w), 0) + coeff
                continue

            i = inv
            a, b = w[i], w[i + 1]
            left, right = w[:i], w[i + 2:]

            # swapped term
            ws = tuple(left + [b, a] + right)
            new[ws] = new.get(ws, 0) + coeff

            # commutator term
            comm = vir_comm(a, b, c)
            for k, cc in comm.items():
                wn = tuple(left + right) if k == "C" else tuple(left + [k] + right)
                new[wn] = new.get(wn, 0) + coeff * cc

            changed = True
        terms = new

    out = {}
    for w, coeff in terms.items():
        w = [x for x in w if x != 0]
        if any(x > 0 for x in w):
            continue
        lam = tuple(sorted([-x for x in w], reverse=True))
        out[lam] = out.get(lam, 0) + sp.simplify(coeff)
    return {k: sp.simplify(v) for k, v in out.items() if v != 0}

def inner_prod(lam, mu, Delta, c):
    word = tuple(list(lam[::-1]) + [-m for m in mu])
    res = reduce_word_on_ket(word, Delta, c)
    return res.get((), sp.Integer(0))

@lru_cache(None)
def gram_matrix(level, Delta, c):
    basis = partitions_int(level)
    G = sp.Matrix([[inner_prod(basis[i], basis[j], Delta, c) for j in range(len(basis))]
                   for i in range(len(basis))])
    return basis, sp.simplify(G)


# ============================================================
# Vertex coefficients (fast)
# ============================================================
def vertex_value_fast(lam, Delta_bra, Delta_ins, Delta_ket):
    p = Delta_bra - Delta_ins - Delta_ket
    coeff = sp.Integer(1)
    shift = 0
    for n in reversed(lam):
        p_prev = p - shift
        coeff *= -(p_prev + (1 - n) * Delta_ins)
        shift += n
    return sp.simplify(coeff)

def _simplify(expr, how: str):
    if how == "none":
        return expr
    if how == "together":
        return sp.together(expr)
    if how == "factor":
        return sp.factor(sp.together(expr))
    if how == "cancel":
        return sp.cancel(expr)
    raise ValueError(f"unknown simplify mode: {how}")

def virasoro_block_coeffs(N, c, Delta_int, D1, D2, D3, D4, simplify="together"):
    coeffs = [sp.Integer(1)]
    for lvl in range(1, N + 1):
        basis, G = gram_matrix(lvl, Delta_int, c)
        Q = G.inv()
        vL = sp.Matrix([vertex_value_fast(lam, D2, D1, Delta_int) for lam in basis])
        vR = sp.Matrix([vertex_value_fast(lam, D4, D3, Delta_int) for lam in basis])
        expr = (vL.T * Q * vR)[0]
        coeffs.append(_simplify(expr, simplify))
    return coeffs


# ============================================================
# Nekrasov SU(2), N_f=4
# ============================================================
def boxes(Y):
    out = []
    for i, r in enumerate(Y, start=1):
        for j in range(1, r + 1):
            out.append((i, j))
    return out

def arm(Y, s):
    i, j = s
    return Y[i - 1] - j

def leg(Y, s):
    i, j = s
    return sum(1 for r in Y if r >= j) - i

def N_factor(x, Y, W, e1, e2):
    prod = sp.Integer(1)
    for s in boxes(Y):
        A = arm(Y, s); L = leg(W, s)
        prod *= (x + e1 * (A + 1) - e2 * L)
    for t in boxes(W):
        A = arm(W, t); L = leg(Y, t)
        prod *= (x - e1 * A + e2 * (L + 1))
    return prod

def Z_vector(a1, a2, Y1, Y2, e1, e2):
    a = [a1, a2]; Y = [Y1, Y2]
    prod = sp.Integer(1)
    for A in (0, 1):
        for B in (0, 1):
            prod *= 1 / N_factor(a[A] - a[B], Y[A], Y[B], e1, e2)
    return prod

def Z_fund(a1, a2, Y1, Y2, mu, e1, e2):
    # shift = e1*(j-1) + e2*(i-1)
    a = [a1, a2]; Y = [Y1, Y2]
    prod = sp.Integer(1)
    for A in (0, 1):
        for (i, j) in boxes(Y[A]):
            prod *= (a[A] + mu + e1 * (j - 1) + e2 * (i - 1))
    return prod

@lru_cache(None)
def all_diags_upto(N):
    out = [[]]
    for n in range(1, N + 1):
        out += [list(p) for p in partitions_int(n)]
    return out

def nekrasov_Z_series(N, a, mus, e1, e2, simplify="together"):
    a1, a2 = a, -a
    diags = all_diags_upto(N)
    Z = [sp.Integer(0)] * (N + 1)
    for Y1 in diags:
        for Y2 in diags:
            k = sum(Y1) + sum(Y2)
            if k > N:
                continue
            w = Z_vector(a1, a2, Y1, Y2, e1, e2)
            for mu in mus:
                w *= Z_fund(a1, a2, Y1, Y2, mu, e1, e2)
            Z[k] += w
    return [_simplify(z, simplify) for z in Z]


# ============================================================
# AGT dictionary + series assembly
# ============================================================
def Delta_from_alpha(alpha, e1, e2):
    e = e1 + e2
    return sp.together(alpha * (e - alpha) / (e1 * e2))

def poch_series(nu, N):
    return [sp.together(sp.rf(nu, k) / sp.factorial(k)) for k in range(N + 1)]

def agt_coeffs(N, e1, e2, a, alpha0, alpha1, beta0, beta2, simplify="together"):
    e = e1 + e2
    c = sp.together(1 + 6 * e**2 / (e1 * e2))

    beta1 = a + e / 2
    Delta_int = Delta_from_alpha(beta1, e1, e2)

    # insertions at (0,q,1,∞): (beta0, alpha0, alpha1, beta2)
    D1 = Delta_from_alpha(alpha0, e1, e2)  # at q
    D2 = Delta_from_alpha(beta0,  e1, e2)  # at 0
    D3 = Delta_from_alpha(alpha1, e1, e2)  # at 1
    D4 = Delta_from_alpha(beta2,  e1, e2)  # at ∞

    vir = virasoro_block_coeffs(N, c, Delta_int, D1, D2, D3, D4, simplify=simplify)

    mu1 = -e / 2 + alpha0 + beta0
    mu2 =  e / 2 + alpha0 - beta0
    mu3 = 3 * e / 2 - alpha1 - beta2
    mu4 =  e / 2 - alpha1 + beta2
    mus = [mu1, mu2, mu3, mu4]

    nu = sp.together(2 * alpha0 * (e - alpha1) / (e1 * e2))

    Z = nekrasov_Z_series(N, a, mus, e1, e2, simplify=simplify)
    U = poch_series(nu, N)

    agt = [sp.Integer(1)]
    for k in range(1, N + 1):
        s = sp.Integer(0)
        for j in range(k + 1):
            s += U[j] * Z[k - j]
        agt.append(_simplify(s, simplify))
    return vir, agt


# ============================================================
# Random parameter sampling (default)
# ============================================================
def rand_rational(num_max, den_max, signed=True, allow_zero=False):
    num = random.randint(0 if allow_zero else 1, num_max)
    den = random.randint(2, den_max)
    if signed and random.randint(0, 1) == 1:
        num = -num
    return sp.Rational(num, den)

def sample_agt_params(args):
    """Return numeric (e1,e2,a,alpha0,alpha1,beta0,beta2) and a dict for printing."""
    vals = {}
    for name in ("e1", "e2", "a", "alpha0", "alpha1", "beta0", "beta2"):
        sval = getattr(args, name)
        if sval is not None:
            vals[name] = sym(sval)

    if len(vals) == 7:
        tup = (vals["e1"], vals["e2"], vals["a"], vals["alpha0"], vals["alpha1"], vals["beta0"], vals["beta2"])
        return tup, vals

    # otherwise: sample all randomly (default behavior)
    for _ in range(args.rand_tries):
        e1 = rand_rational(args.num_max, args.den_max, signed=args.signed, allow_zero=args.allow_zero)
        e2 = rand_rational(args.num_max, args.den_max, signed=args.signed, allow_zero=args.allow_zero)
        a  = rand_rational(args.num_max, args.den_max, signed=args.signed, allow_zero=args.allow_zero)
        alpha0 = rand_rational(args.num_max, args.den_max, signed=args.signed, allow_zero=args.allow_zero)
        alpha1 = rand_rational(args.num_max, args.den_max, signed=args.signed, allow_zero=args.allow_zero)
        beta0  = rand_rational(args.num_max, args.den_max, signed=args.signed, allow_zero=args.allow_zero)
        beta2  = rand_rational(args.num_max, args.den_max, signed=args.signed, allow_zero=args.allow_zero)
        try:
            vir, agt = agt_coeffs(args.N, e1,e2,a, alpha0,alpha1,beta0,beta2, simplify=args.simplify)
        except (NonInvertibleMatrixError, ZeroDivisionError):
            continue
        if not all(finite(x) for x in vir[1:] + agt[1:]):
            continue

        vals = {"e1": e1, "e2": e2, "a": a, "alpha0": alpha0, "alpha1": alpha1, "beta0": beta0, "beta2": beta2}
        return (e1,e2,a, alpha0,alpha1,beta0,beta2), vals

    raise SystemExit(f"Could not find nonsingular random parameters in {args.rand_tries} tries. "
                     "Try increasing --rand-tries or changing ranges.")


# ============================================================
# Output helpers
# ============================================================
def notation_text():
    return (
        "Notation (AGT variables):\n"
        "  e1=ε1, e2=ε2, e=e1+e2.\n"
        "  c = 1 + 6 e^2/(e1 e2).\n"
        "  Δ(alpha) = alpha (e-alpha)/(e1 e2).\n"
        "Insertions at (0,q,1,∞) with momenta (beta0, alpha0, alpha1, beta2).\n"
        "Internal momentum beta1 = a + e/2.\n"
        "U(1) exponent nu = 2 alpha0 (e-alpha1)/(e1 e2).\n"
    )

def print_series_plain(label, coeffs, levels):
    for k in levels:
        print(f"{label}[{k}] = {coeffs[k]}\n")

def print_series_latex(label, coeffs, levels, latex_env=None):
    lines = [sp.latex(sp.Symbol(f"{label}_{k}")) + " = " + sp.latex(coeffs[k]) + r"\\"
             for k in levels]
    body = "\n".join(lines)
    if latex_env:
        print(r"\begin{" + latex_env + "}\n" + body + "\n" + r"\end{" + latex_env + "}\n")
    else:
        print(body + "\n")

def coeffs_to_json(label, coeffs, levels):
    return {f"{label}[{k}]": str(coeffs[k]) for k in levels}


# ============================================================
# CLI
# ============================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="Virasoro 4-pt block vs AGT/Nekrasov (SU(2), N_f=4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--N", type=int, default=4, help="Max level / instanton number.")
    p.add_argument("--levels", type=str, default=None,
                   help="Comma-separated levels to print (e.g. '0,1,2,4'). Default: 0..N.")
    p.add_argument("--show", choices=["vir", "agt", "both", "none"], default="both",
                   help="Which series to print. Use 'none' to print neither.")
    p.add_argument("--mode", choices=["agt", "vir"], default="agt",
                   help="agt: build Vir+AGT from (e1,e2,a,alpha0,alpha1,beta0,beta2). "
                        "vir: compute only Vir from (c,Delta,D1..D4).")

    p.add_argument("--symbolic", action="store_true", help="Use symbolic parameters instead of numeric.")
    p.add_argument("--simplify", choices=["none", "together", "factor", "cancel"], default="together",
                   help="Simplification applied to coefficients.")

    p.add_argument("--output", choices=["plain", "latex", "both", "json"], default="plain",
                   help="Output format.")
    p.add_argument("--latex-env", type=str, default=None, help="LaTeX environment name, e.g. 'align*'.")
    p.add_argument("--explain", action="store_true", help="Print notation legend (AGT mode).")
    p.add_argument("--print-params", action="store_true", help="Print the numeric params used (helpful for random runs).")

    # Random sampler controls (used by default unless all params explicitly provided)
    p.add_argument("--seed", type=str, default=None,
                   help="Seed RNG (int). Default None = do not seed (fresh randomness each run).")
    p.add_argument("--num-max", type=int, default=9, help="Max abs numerator for random rationals.")
    p.add_argument("--den-max", type=int, default=10, help="Max denominator for random rationals (min is 2).")
    p.add_argument("--signed", action="store_true", help="Allow negative numerators in random rationals.")
    p.add_argument("--allow-zero", action="store_true", help="Allow 0 as a random numerator.")
    p.add_argument("--rand-tries", type=int, default=200, help="Max tries to find nonsingular random params.")

    # numeric params for AGT mode (all optional; if not all 7 provided, random sampling is used)
    for nm in ("e1", "e2", "a", "alpha0", "alpha1", "beta0", "beta2"):
        p.add_argument(f"--{nm}", type=str, default=None)

    # numeric params for vir-only mode
    for nm in ("c", "Delta", "D1", "D2", "D3", "D4"):
        p.add_argument(f"--{nm}", type=str, default=None)

    return p


def main():
    args = build_parser().parse_args()

    # levels to print
    if args.levels is None:
        levels = list(range(0, args.N + 1))
    else:
        levels = [int(s.strip()) for s in args.levels.split(",") if s.strip() != ""]
        for k in levels:
            if k < 0 or k > args.N:
                raise SystemExit(f"level {k} outside [0,{args.N}]")

    # seeding: default None means do nothing
    seed = parse_seed(args.seed)
    if seed is not None:
        random.seed(seed)

    if args.explain and args.mode == "agt":
        print(notation_text())

    if args.mode == "agt":
        if args.symbolic:
            e1,e2,a,alpha0,alpha1,beta0,beta2 = sp.symbols("e1 e2 a alpha0 alpha1 beta0 beta2")
            vir, agt = agt_coeffs(args.N, e1,e2,a, alpha0,alpha1,beta0,beta2, simplify=args.simplify)
            params_used = None
        else:
            (e1,e2,a, alpha0,alpha1,beta0,beta2), params_used = sample_agt_params(args)
            vir, agt = agt_coeffs(args.N, e1,e2,a, alpha0,alpha1,beta0,beta2, simplify=args.simplify)

        if args.print_params and params_used is not None:
            print("Parameters used:", {k: str(v) for k, v in params_used.items()})
            print()

        # printing
        def do_plain():
            if args.show in ("vir", "both"):
                print_series_plain("cVir", vir, levels)
            if args.show in ("agt", "both"):
                print_series_plain("cAGT", agt, levels)

        def do_latex():
            if args.show in ("vir", "both"):
                print_series_latex("cVir", vir, levels, latex_env=args.latex_env)
            if args.show in ("agt", "both"):
                print_series_latex("cAGT", agt, levels, latex_env=args.latex_env)

        if args.show != "none":
            if args.output == "plain":
                do_plain()
            elif args.output == "latex":
                do_latex()
            elif args.output == "both":
                do_plain()
                do_latex()
            elif args.output == "json":
                payload = {}
                if args.show in ("vir", "both"):
                    payload.update(coeffs_to_json("cVir", vir, levels))
                if args.show in ("agt", "both"):
                    payload.update(coeffs_to_json("cAGT", agt, levels))
                print(json.dumps(payload, indent=2, sort_keys=True))

    else:  # vir-only mode
        if args.symbolic:
            c, Delta, D1, D2, D3, D4 = sp.symbols("c Delta D1 D2 D3 D4")
        else:
            missing = [nm for nm in ("c", "Delta", "D1", "D2", "D3", "D4") if getattr(args, nm) is None]
            if missing:
                raise SystemExit(f"vir mode requires --c --Delta --D1 --D2 --D3 --D4 (missing: {', '.join(missing)})")
            c, Delta, D1, D2, D3, D4 = map(sym, (args.c, args.Delta, args.D1, args.D2, args.D3, args.D4))

        vir = virasoro_block_coeffs(args.N, c, Delta, D1, D2, D3, D4, simplify=args.simplify)

        if args.show == "none":
            return

        if args.output in ("plain", "both") and args.show in ("vir", "both"):
            print_series_plain("cVir", vir, levels)
        if args.output in ("latex", "both") and args.show in ("vir", "both"):
            print_series_latex("cVir", vir, levels, latex_env=args.latex_env)
        if args.output == "json" and args.show in ("vir", "both"):
            print(json.dumps(coeffs_to_json("cVir", vir, levels), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
