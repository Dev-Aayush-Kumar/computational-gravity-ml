import sympy as sp

x = sp.symbols('x', positive=True)

phi_n = -1/(2*x)

phi_gr = sp.Rational(1,2) * (
    sp.sqrt(1 - 1/x) - 1
)

g_n = sp.simplify(
    -sp.diff(phi_n, x)
)

g_gr = sp.simplify(
    -sp.diff(phi_gr, x)
)

ratio = sp.simplify(
    g_gr / g_n
)

print("\n===== NEWTONIAN FIELD =====\n")
sp.pprint(g_n)

print("\n===== RELATIVISTIC FIELD =====\n")
sp.pprint(g_gr)

print("\n===== FIELD RATIO =====\n")
sp.pprint(ratio)

print("\n===== WEAK FIELD LIMIT =====\n")
print(
    sp.simplify(
        sp.limit(
            ratio,
            x,
            sp.oo
        )
    )
)

print("\n===== HORIZON LIMIT =====\n")
print(
    sp.limit(
        ratio,
        x,
        1,
        dir="+"
    )
)