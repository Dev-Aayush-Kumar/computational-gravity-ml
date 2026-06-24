import sympy as sp

x = sp.symbols('x', positive=True)

phi_n = -1/(2*x)

phi_gr = sp.Rational(1,2) * (
    sp.sqrt(1 - 1/x) - 1
)

signed_error = sp.simplify(
    100 * (phi_gr - phi_n) / phi_gr
)

relative_error = sp.simplify(
    100 * sp.Abs(
        (phi_gr - phi_n) / phi_gr
    )
)

print("\nSIGNED ERROR")

print(
    sp.limit(
        signed_error,
        x,
        1,
        dir="+"
    )
)

print(
    sp.limit(
        signed_error,
        x,
        sp.oo
    )
)

print("\nRELATIVE ERROR")

print(
    sp.limit(
        relative_error,
        x,
        1,
        dir="+"
    )
)

print(
    sp.limit(
        relative_error,
        x,
        sp.oo
    )
)