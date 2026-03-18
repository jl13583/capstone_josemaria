# osp_check_normalized.py
import sympy as sp
from math import isclose

sp.init_printing()

# --- 0. User: set your even generators here (3x3) --------------
# Replace these with your exact E,F,H if they differ from the standard ones
E = sp.Matrix([[0,1,0],[0,0,0],[0,0,0]])
F = sp.Matrix([[0,0,0],[1,0,0],[0,0,0]])
H = sp.Matrix([[1,0,0],[0,-1,0],[0,0,0]])

# --- 1. New odd generators you specified -------------------------
Tp = sp.Matrix([[0,0,1],[0,0,0],[0,1,0]])   # T+
Tm = sp.Matrix([[0,0,0],[0,0,-1],[1,0,0]])  # T-

# --- 2. Helper operations ----------------------------------------
def comm(A,B): return A*B - B*A
def anticomm(A,B): return A*B + B*A

# Parity predicate (we keep it simple)
even_list = [E, F, H]
odd_list  = [Tp, Tm]
def is_even_matrix(A):
    return any((A - B).equals(sp.zeros(3)) for B in even_list)

def is_odd_matrix(A):
    return any((A - B).equals(sp.zeros(3)) for B in odd_list)

# Evaluate bracket with parity: if both odd -> anticommutator, else commutator
def super_bracket(A,B):
    if is_odd_matrix(A) and is_odd_matrix(B):
        return anticomm(A,B)
    else:
        return comm(A,B)

# Express a 3x3 matrix M in terms of the even-basis (E,F,H) coordinates.
def coords_in_even_basis(M):
    # We get coefficients c_E,c_F,c_H such that M acting on basis vectors equals the linear combo.
    # Simpler approach: apply M to the basis coordinate vectors [1,0,0],[0,1,0],[0,0,1]
    # and express each column in the E,F,H basis by solving small linear system.
    cols = [M * sp.Matrix([1,0,0]), M * sp.Matrix([0,1,0]), M * sp.Matrix([0,0,1])]
    # But we want a single vector of coefficients (a,b,c) so that M = a*E + b*F + c*H
    # Solve for scalars a,b,c from matrix equality: M = a*E + b*F + c*H
    a,b,c = sp.symbols('a b c')
    expr = a*E + b*F + c*H
    sol = sp.solve(sp.Matrix(expr - M), (a,b,c), dict=True)
    if sol:
        s = sol[0]
        return (sp.simplify(s[a]), sp.simplify(s[b]), sp.simplify(s[c]))
    else:
        return (None,None,None)

# Express a 3x3 matrix N in terms of the odd-basis (Tp,Tm) coordinates: N = u*Tp + v*Tm
def coords_in_odd_basis(N):
    u,v = sp.symbols('u v')
    expr = u*Tp + v*Tm
    sol = sp.solve(sp.Matrix(expr - N), (u,v), dict=True)
    if sol:
        s = sol[0]
        return (sp.simplify(s[u]), sp.simplify(s[v]))
    else:
        return (None,None)

# Pretty print helper
def fmt_even_coords(c):
    if c[0] is None: return "not expressible in E,F,H basis"
    terms = []
    names = ['E','F','H']
    for coeff,name in zip(c,names):
        if coeff != 0:
            terms.append(f"({sp.simplify(coeff)})*{name}")
    return ' + '.join(terms) if terms else '0'

def fmt_odd_coords(c):
    if c[0] is None: return "not expressible in T+,T- basis"
    terms = []
    names = ['T+','T-']
    for coeff,name in zip(c,names):
        if coeff != 0:
            terms.append(f"({sp.simplify(coeff)})*{name}")
    return ' + '.join(terms) if terms else '0'

# --- 3. Compute raw brackets -------------------------------------
print("=== Raw (observed) brackets ===")
pairs = [
    ('[H,E]', H, E), ('[H,F]', H, F), ('[E,F]', E, F),
    ('[H,T+]', H, Tp), ('[H,T-]', H, Tm), ('[E,T-]', E, Tm), ('[F,T+]', F, Tp),
    ('{T+,T+}', Tp, Tp), ('{T-,T-}', Tm, Tm), ('{T+,T-}', Tp, Tm)
]

for label, A, B in pairs:
    br = super_bracket(A,B)
    # decide expected parity of result: if both odd -> even else (if any even) parity = odd? Actually:
    # if both even -> even; if one even & one odd -> odd; if both odd -> even.
    if is_odd_matrix(A) and is_odd_matrix(B):
        # even output
        coords = coords_in_even_basis(sp.simplify(br))
        print(f"{label} -> {fmt_even_coords(coords)}")
    else:
        # output lives in odd subspace
        coords = coords_in_odd_basis(sp.simplify(br))
        print(f"{label} -> {fmt_odd_coords(coords)}")

# --- 4. Automatic rescaling (optional) ---------------------------
# Suppose we want to normalize to target relations:
# target: [H,T+] = 1*T+, [H,T-] = -1*T-, {T+,T-} = 1*H, {T+,T+} = 2*E, {T-,T-} = -2*F
# We'll compute observed scalars and propose a scaling s_T for Tp, t_T for Tm, and r_H for H (if needed).

print("\n=== Observed scalar summary (for possible automatic normalization) ===")

# get observed scalars
def observed_scalar_for_comm(A,B, expected_basis='odd'):
    br = super_bracket(A,B)
    if expected_basis == 'odd':
        coords = coords_in_odd_basis(sp.simplify(br))
        return coords  # returns tuple (u,v) coefficients for (T+,T-)
    else:
        coords = coords_in_even_basis(sp.simplify(br))
        return coords  # returns (a,b,c) for (E,F,H)

# Observed {T+,T-} in even basis
obs_AQpm = observed_scalar_for_comm(Tp, Tm, expected_basis='even')
print("{T+,T-} observed (coeffs of E,F,H):", obs_AQpm)

# Observed [H,T+] in odd basis
obs_H_Tp = observed_scalar_for_comm(H, Tp, expected_basis='odd')
print("[H,T+] observed (coeffs of T+,T-):", obs_H_Tp)

# Observed {T+,T+}
obs_QpQp = observed_scalar_for_comm(Tp, Tp, expected_basis='even')
print("{T+,T+} observed (coeffs of E,F,H):", obs_QpQp)

# Observed {T-,T-}
obs_QmQm = observed_scalar_for_comm(Tm, Tm, expected_basis='even')
print("{T-,T-} observed (coeffs of E,F,H):", obs_QmQm)

# --- 5. Example: if factors are uniform, suggest simple scaling ----------
# (This will not solve arbitrary mixed scalings; it's a best-effort helper.)
print("\n=== Suggested simple scaling (best-effort) ===")
# If {T+,T-} = c*H with c != 0 then scaling Tp <- s*Tp, Tm <- s*Tm, H <- r*H with r = desired/observed
# For typical desire desired_c = 1
desired_c = 1
if obs_AQpm[2] is not None and obs_AQpm[2] != 0:
    observed_c = sp.simplify(obs_AQpm[2])
    r = sp.simplify(desired_c / observed_c)         # scaling for H: H' = r * H
    # choose s so that s^2 * observed_c / r = desired_c  -> trivial if s=1 works when r chosen as above
    print("Observed {T+,T-} coeff on H:", observed_c, "; set H' = (1/{} )*H to normalize".format(sp.simplify(observed_c)))
else:
    print("Could not read a single H coefficient for {T+,T-}; manual analysis needed.")
