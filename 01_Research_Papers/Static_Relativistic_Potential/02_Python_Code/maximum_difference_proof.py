import sympy as sp

x = sp.symbols('x', positive=True)

phi_n = -1/(2*x)

phi_gr = sp.Rational(1,2) * (
    sp.sqrt(1 - 1/x) - 1
)

difference = sp.simplify(
    phi_gr - phi_n
)

print("\n===== DIFFERENCE =====\n")
sp.pprint(difference)

derivative = sp.simplify(
    sp.diff(difference, x)
)

print("\n===== DERIVATIVE =====\n")
sp.pprint(derivative)

critical_points = sp.solve(
    derivative,
    x
)

print("\n===== CRITICAL POINTS =====\n")
print(critical_points)

for point in critical_points:

    if point.is_real:

        print("\nPoint =", point)

        print(
            "Difference =",
            sp.simplify(
                difference.subs(x, point)
            )
        )