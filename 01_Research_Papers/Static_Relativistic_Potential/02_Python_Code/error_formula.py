import sympy as sp

x = sp.symbols('x', positive=True)

phi_n = -1/(2*x)

phi_gr = sp.Rational(1,2) * (
    sp.sqrt(1 - 1/x) - 1
)

difference = phi_gr - phi_n

error = sp.simplify(
    100 * sp.Abs(difference / phi_gr)
)

print("\n===== EXACT ERROR FORMULA =====\n")
print(error)