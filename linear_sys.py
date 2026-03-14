import sympy as sp
import os
import random 
from textwrap import shorten
from lss_1_op import assoc as assoc_lss1
from lss_2_op import assoc as assoc_lss2
from sympy.polys.polytools import groebner
from sympy.polys.domains import GF
import csv
from collections import Counter
import math 
from sympy import factor_list
# --------------------------------------------------------
# Helper function: Reducing polynomial coefficients mod p
# --------------------------------------------------------

def preprocess_polys_in_char_p(expr, p, gens, invert_pairs=None):
    """
    Prepare polynomial system over GF(p).
    - expr: list of expressions f = 0
    - gens: list of poly generators 
    - invert_pairs: list of pairs (a,b) to replace a with b^{-1} mod p (e.g. for g and ginv)
    
    Returns: list of polynomial expressions reduced mod p
    """
    polys = list(expr)

    # Enforce invertibility ONLY if requested
    invertible_syms = set()
    if invert_pairs:
        for g, ginv in invert_pairs:
            polys.append(g*ginv - 1)
            invertible_syms |= {g, ginv}

    out = []
    for f in polys:
        f = sp.expand(f)
        num, den = sp.fraction(sp.together(f))

        if den != 1:
            # If denominator uses ONLY invertible symbols (e.g. g, ginv),
            # we can clear it safely in the localized ring.
            den_syms = den.free_symbols
            if invertible_syms and den_syms.issubset(invertible_syms):
                # equation num/den = 0 is equivalent to num = 0 on den ≠ 0
                f = num
            else:
                raise ValueError(f"Non-polynomial expression remains: denominator {den}")

        P = sp.Poly(sp.expand(f), *gens, modulus=p)
        out.append(P.as_expr())

    return out

def rewrite_inverses(expr, g, ginv):
    """
    Rewrite any negative powers so expression becomes polynomial in g and ginv,
    assuming ginv = g^{-1}.

    Rules:
    g^(-n)     -> ginv^n
    ginv^(-n)  -> g^n
    """
    expr = sp.together(expr)

    def repl_pow(e):
        base, exp = e.base, e.exp
        if exp.is_Integer and exp < 0:
            n = int(-exp)
            if base == g:
                return ginv**n
            if base == ginv:
                return g**n
        return e

    expr = expr.replace(lambda e: isinstance(e, sp.Pow), repl_pow)

    # Catch any remaining 1/g or 1/ginv that might appear as Mul/Div forms
    expr = expr.subs({1/g: ginv, 1/ginv: g})

    return sp.expand(expr)

# --------------------------------------------------------
# 1. Define symbols for unknown coefficients
# --------------------------------------------------------
# odd generators: T+, T-
# even generators: E, F, H
even = ['E','F','H']
odd  = ['Tp','Tm']
gens = even + odd

# --- Choosing Burde's structures ---
structure_version = 2  # or 2

if structure_version == 1:
    u, v = sp.symbols('u v')
    w = v - u
    lamE = sp.Matrix([[0, u, 1], 
                [0, w, -1], 
                [1, w**2 +1, -w]])
    lamF = sp.Matrix([[u, u - v*w**2, v*w], 
                [w, v, 1], 
                [w**2, -v*w, -u - v]])
    lamH = sp.Matrix([[0, v*w, -v],
                [-1, -1, 0],
                [-w, -u - v, 1]])
elif structure_version == 2:
    g = sp.symbols('g')
    ginv = sp.symbols('ginv')
    lamE = sp.Matrix([[0, 0, g],
                    [0, 0, 0],
                    [0, -1 - ginv, 0]])
    lamF = sp.Matrix([[0, 0, 0],
                    [0, 0, g], 
                    [1 - ginv, 0, 0]])
    lamH = sp.Matrix([[g - 1, 0, 0], 
                    [0, g + 1, 0],
                    [0, 0, g]])
    
# --- Defining containers if missing ---
if 'beta' not in locals():
    beta = {}
if "full_sym" not in locals():
    full_sym = {}
    
# --- Inject Burde's even-even structure into beta/full_sym ---
lams = {'E': lamE, 'F': lamF, 'H': lamH}
# standard basis vectors k^3
basis_vecs = {'E': sp.Matrix([1,0,0]), 
            'F': sp.Matrix([0,1,0]), 
            'H': sp.Matrix([0,0,1])
}

for X in even: 
    for Y in even: 
        vec = lams[X] * basis_vecs[Y]
        for i, Z in enumerate(even):
            beta[(X, Y, Z)] = sp.simplify(vec[i])
            full_sym[beta[(X, Y, Z)]] = vec[i]

print("Burde even-even products initialized:")
for X in even:
    for Y in even:
        prod = {Z: full_sym[beta[(X, Y, Z)]] for Z in even}
        print(f"{X} * {Y} =", prod)
        
beta_even = beta.copy() # save for later 

# print('Contents of beta_even')
# for key, val in beta_even.items():
#     print(f"{key} : {val}")
    

# Unknown structure constants:
# beta[x,y,z] means coefficient of z in (x*y)
beta = {} # reinitializing beta 
for (x, y, z), val in beta_even.items():
    beta[(x, y, z)] = val # copying even-even products

# print('Contents of beta after copying even-even products') 
# for key, val in beta.items():
#     print(f"{key} : {val}")
    
for x in gens:
    for y in gens:
        # skip even-even (Burde already did this)
        if (x in even and y in even):
            for z in even:
                beta[(x,y,z)] = beta_even[(x,y,z)]
            continue
            
        # determining output basis by parity
        if (x in even and y in even) or (x in odd and y in odd):
            out_basis = even
        else: 
            out_basis = odd
            
        for z in out_basis:
            # add symbol only if it doesn't exist yet in beta
            if (x, y, z) not in beta:
                beta[(x,y,z)] = sp.Symbol(f'b_{x}_{y}_{z}')

# --------------------------------------------------------
# 2. Parity map
# --------------------------------------------------------
parity = {}
for x in even: parity[x] = 0
for x in odd:  parity[x] = 1

# --------------------------------------------------------
# 3. Define the osp(1|2) super-bracket table
# --------------------------------------------------------
# Each entry [x,y] = linear combination dict {basis : coeff}
bracket = { (x,y): {} for x in even+odd for y in even+odd }

# sl2(k) part
"""bracket[('H','E')] = {'E': 2}
bracket[('E','H')] = {'E': -2}
bracket[('H','F')] = {'F': -2}
bracket[('F','H')] = {'F': 2}
bracket[('E','F')] = {'H': 1}
bracket[('F','E')] = {'H': -1}"""

# even–odd
bracket[('H','Tp')] = {'Tp': 1}
bracket[('Tp','H')] = {'Tp': -1}
bracket[('H','Tm')] = {'Tm': -1}
bracket[('Tm','H')] = {'Tm': 1}
bracket[('E','Tm')] = {'Tp': -1}
bracket[('Tm','E')] = {'Tp': 1}
bracket[('F','Tp')] = {'Tm': -1}
bracket[('Tp','F')] = {'Tm': 1}

# odd–odd (anticommutators)
bracket[('Tp','Tp')] = {'E': 2}
bracket[('Tm','Tm')] = {'F': -2}
bracket[('Tp','Tm')] = {'H': 1}
bracket[('Tm','Tp')] = {'H': 1}

# --------------------------------------------------------
# 4. Build the linear system
# --------------------------------------------------------
eqs = []
unknowns = [val for key, val in beta.items() if key not in beta_even]

pairs = []
pairs_seen = set()

for x in even+odd:
    for y in even+odd:
        # Skip pairs with trivial bracket
        if bracket[(x,y)] == {}:
            continue

        # Avoid redundant opposite pairs (since [x,y] = -(-1)^{|x||y|}[y,x])
        if (y,x) in pairs_seen:
            continue
        pairs_seen.add((x,y))
        pairs.append((x,y))
        
        sign = (-1)**(parity[x]*parity[y])

        # Compute x*y and y*x using unknown betas
        prod_xy = {z: beta[(x,y,z)] for z in even+odd if (x,y,z) in beta}
        prod_yx = {z: beta[(y,x,z)] for z in even+odd if (y,x,z) in beta}

        # graded commutator: [x,y] = x*y - (-1)^{|x||y|} y*x
        for z in even+odd:
            if z in bracket[(x,y)]:
                rhs = bracket[(x,y)][z]
            else:
                rhs = 0
            lhs = prod_xy.get(z,0) - sign * prod_yx.get(z,0)
            eq = sp.Eq(lhs, rhs)
            eqs.append(eq)

# Simplify all equations
eqs = [sp.simplify(e) for e in eqs] 

# Printing unordered pairs for verification 
print("\nGenerated equations for the following generator pairs:")
for (x,y) in pairs: 
    print(f"    Pair {x, y} --> scalar eqs: {len(gens)}  (components: {gens})")
    
print(f"\nNumber of equations: {len(eqs)}")
print(f"Number of unknowns: {len(unknowns)}")

# --------------------------------------------------------
# 5. Solve the linear system symbolically
# --------------------------------------------------------
solution = sp.solve(eqs, unknowns, dict=True, simplify=True)

if solution:
    print(f"Solutions found: {len(solution)}")
    for sol in solution:
        for k,v in sol.items():
            if v != 0:
                print(f"{k} = {sp.simplify(v)}")
else:
    print("No exact symbolic solution — equations may be dependent or inconsistent.")
    # Optional diagnostic: check if any equations are identically true or false
    for e in eqs[:10]:  # print a few samples
        print(e, "->", e.lhs.simplify() == e.rhs.simplify())
        
# --------------------------------------------------------
# 6. Spot-test solutions with random assignments to free variables
# --------------------------------------------------------

# For example: solution = sp.solve(eqs, unknowns, dict=True)
if not solution:
    raise RuntimeError("No solution dictionary available (sol is empty).")

sol0 = solution[0]   # dictionary mapping some unknowns -> expressions (possibly in other unknowns)

# 1) Determine free symbols (those in unknowns not solved for directly)
sol_keys = set(sol0.keys())
free_syms = [u for u in unknowns if u not in sol_keys]

print("Num unknowns:", len(unknowns))
print("Num solved symbols:", len(sol_keys))
print("Num free symbols:", len(free_syms))
print("Free symbols sample:", free_syms[:])

print("\n=== Full parametrization of all 36 unknowns ===")
for s in unknowns: 
    if s in sol0: 
        print(f"{s} = {sol0[s]}")
    else: 
        # It is free
        print(f"{s} = free parameter")

# === Build full table of coefficients ===
# latex_rows = []
# for s in list(beta.values()):
#     name = sp.latex(s)
#     if s in sol0:
#         expr = sp.latex(sp.simplify(sol0[s]))
#         status = "Dependent"
#     else:
#         expr = "\\text{free parameter}"
#         status = "Free"
#     latex_rows.append(f"{name} & {expr} & {status}\\\\")
    
# latex_table = (
#     "\\begin{table}[H]\n"
#     "\\centering\n"
#     "\\begin{tabular}{lll}\\hline\n"
#     "Coefficient & Expression & Status\\\\\\hline\n" +
#     "\n".join(latex_rows) +
#     "\\hline\n\\end{tabular}\n\\caption{Stage 1 solutions for the $b_{x,y}^z$ coefficients in the extended $osp(1|2)$ product.}\n\\end{table}"
# )

# print("\n===== Copy this LaTeX table to your report =====\n")
# print(latex_table)
# --------------------------------------------------------

# 2) Function to construct a full substitution mapping for all unknowns
def build_full_substitution(free_assign):
    """
    free_assign: dict {Symbol: value} giving concrete values for each free symbol.
    Returns: dict mapping every unknown symbol -> concrete numeric value (Sympy ints).
    """
    full = {}
    # assign free symbols
    for s in free_syms:
        if s in free_assign:
            full[s] = sp.Integer(free_assign[s])
        else:
            # default choice 0
            full[s] = sp.Integer(0)
    # now compute solved symbols from sol0, substituting free values
    for k, expr in sol0.items():
        # expr may be symbolic in free_syms; substitute then evaluate
        full[k] = sp.simplify(sp.nsimplify(expr.subs(full)))
    return full

# 3) Function to evaluate the graded commutator for a pair using the full subs
def lhs_minus_rhs_for_pair(x, y, full_subs):
    """
    Return a dict of residual coefficients for each basis element z:
    residual_z = coeff_in(x*y - (-1)^{|x||y|} y*x) - target_coeff
    using numeric values from full_subs.
    """
    sign = (-1)**(parity[x] * parity[y])
    residuals = {}
    # compute numeric coefficient of basis z in x*y: look up beta[(x,y,z)] if present
    for z in (list(even) + list(odd)):  # even/odd lists of earlier
        coeff_xy = 0
        coeff_yx = 0
        if (x,y,z) in beta:
            coeff_xy = sp.Integer(full_subs[beta[(x,y,z)]])
        if (y,x,z) in beta:
            coeff_yx = sp.Integer(full_subs[beta[(y,x,z)]])
        lhs_coeff = coeff_xy - sign * coeff_yx
        rhs_coeff = bracket[(x,y)].get(z, 0)
        residuals[z] = sp.Integer(lhs_coeff - rhs_coeff)
    return residuals

# 4) Test function: run multiple random assignments for free symbols
def run_spot_tests(num_trials=10, rng_seed=0, modp=None):
    random.seed(rng_seed)
    for t in range(num_trials):
        # choose random small integers for free symbols (e.g. -2..2)
        assign = {}
        for s in free_syms:
            assign[s] = random.randint(-2,2)
        full = build_full_substitution(assign)
        # if modulus requested, reduce all entries mod p into 0..p-1
        if modp is not None:
            for k in list(full.keys()):
                full[k] = int(full[k]) % modp

        # Check all nontrivial pairs (those with bracket target nonzero)
        bad_count = 0
        for x in gens:
            for y in gens:
                if bracket[(x,y)] == {}:
                    continue
                resid = lhs_minus_rhs_for_pair(x,y, full)
                # For parity, only some z will be relevant: target keys + whatever lhs produced
                # Check that all residuals are zero (or zero mod p if modp provided)
                for z, val in resid.items():
                    if modp is not None:
                        val = int(val) % modp
                    if val != 0:
                        bad_count += 1
                        # optionally print first few failures
                        if bad_count <= 10:
                            print(f"Trial {t}: Failure for pair ({x},{y}), basis {z}: residual={val}")
                        break
                if bad_count:
                    break
            if bad_count:
                break
        if bad_count == 0:
            print(f"Trial {t}: PASS (assignment {assign})")
        else:
            print(f"Trial {t}: FAIL (assignment {assign}) with {bad_count} nonzero residuals")
            # Optionally break early if you want
            # break

# 5) Run some tests (example: 20 random trials, also a deterministic one with zeros)
print("\n-- Deterministic test: set all free symbols to 0 --")
full0 = build_full_substitution({s:0 for s in free_syms})
fails = 0
for x in gens:
    for y in gens:
        if bracket[(x,y)] == {}:
            continue
        r = lhs_minus_rhs_for_pair(x,y, full0)
        if any(v != 0 for v in r.values()):
            print("Deterministic test FAIL for pair", (x,y), " residuals:", r)
            fails += 1
if fails == 0:
    print("Deterministic zero-assignment PASSED all bracket equations.")

print("\n-- Randomized spot-tests (10 trials) --")
run_spot_tests(num_trials=10, rng_seed=1, modp=None)

# Optional: run tests mod 3 if that's your base field
print("\n-- Randomized spot-tests mod 3 --")
run_spot_tests(num_trials=5, rng_seed=2, modp=3)

# --------------------------------------------------------
# 7. Graded left-symmetric identity checks
# --------------------------------------------------------

# === Stage 2: graded left-symmetric identity checks ===

# === Stage 2 symbolic solver for graded left-symmetric identity ===
print("\n=== Stage 2 symbolic solving for graded left-symmetric identity ===")

# Use symbolic dictionary from Stage1: keep free symbols symbolic
full_sym = {b: b for b in beta.values()} 

for s in beta.values():
    if s in sol0:
        full_sym[s] = sp.simplify(sol0[s])
    else:
        full_sym[s] = s  # leave free

# helper to build multiplication in symbolic form
def star_sym(x, y):
    out = {z: 0 for z in gens}
    for z in gens:
        if (x,y,z) in beta:
            out[z] = full_sym[beta[(x,y,z)]]
    return out

def assoc_sym(x, y, z):
    if x in even and y in even and z in even: 
        # mapping 'E', 'F', 'H' to 0,1,2
        index_map = {'E': 0, 'F': 1, 'H': 2} # using imported assoc_lss1 or assoc_lss2
        if structure_version == 1:
            result = assoc_lss1(index_map[x], index_map[y], index_map[z])
        elif structure_version == 2:
            result = assoc_lss2(index_map[x], index_map[y], index_map[z])
        else: 
            raise ValueError(f"Unsupported structure_version: {structure_version}")
        return {"E": result[0], "F": result[1], "H": result[2], "Tp": 0, "Tm": 0}
    
    # sign_xy = (-1)**(parity[x]*parity[y])
    xy = star_sym(x, y)
    yz = star_sym(y, z)

    term1 = {u: 0 for u in even+odd}
    for t,c in xy.items():
        for u,val in star_sym(t, z).items():
            term1[u] += c*val

    term2 = {u: 0 for u in even+odd}
    for t,c in yz.items():
        for u,val in star_sym(x, t).items():
            term2[u] += c*val

    return {u: sp.simplify(term1[u] - term2[u]) for u in gens}

eqs2 = []
for x in even+odd:
    for y in even+odd:
        for z in even+odd:
            sign_xy = (-1)**(parity[x]*parity[y])
            a1 = assoc_sym(x, y, z)
            a2 = assoc_sym(y, x, z)
            for u in gens:
                eq = sp.Eq(sp.simplify(a1[u] - sign_xy*a2[u]), 0)
                if isinstance(eq, sp.Equality):
                    if not sp.simplify(eq.lhs - eq.rhs) == 0:
                        eqs2.append(eq)
                elif eq is not True and eq is not False:
                    eqs2.append(eq)

print(f"Total graded LSS equations generated: {len(eqs2)}")

# Try to simplify and remove duplicates
eqs2 = list({str(e): e for e in eqs2}.values())

# Solve for the free parameters symbolically
unknowns2 = list(set(s for s in beta.values() if s not in sol0))
sol2 = sp.solve(eqs2, unknowns2, dict=True, simplify=True)

print(f"\nNumber of unknowns in Stage 2: {len(unknowns2)}")
print(f"Unknowns: {unknowns2}")

print(f"Found {len(sol2)} symbolic solution sets for the free parameters.")
if sol2:
    for k,v in sol2[0].items():
        print(k, "=", v)
else:
    print("No full symbolic solution found (possibly multiple or dependent constraints).")
    
print(f"Collected {len(eqs2)} nontrivial graded LSS equations in Stage 2.")

# Inspecting some of the nonzero graded LSS associator differences
print("\n=== Sample graded associator differences (nonzero before solving) ===")

printed = 0
for x in gens:
    for y in gens:
        for z in gens:
            sign_xy = (-1)**(parity[x]*parity[y])
            a1 = assoc_sym(x, y, z)
            a2 = assoc_sym(y, x, z)
            diff = {u: sp.simplify(a1[u] - sign_xy*a2[u]) for u in gens}

            # Filter: show only if some components are nonzero
            if any(v != 0 for v in diff.values()):
                printed += 1
                print(f"({x},{y},{z}) ->", {k: v for k, v in diff.items() if v != 0})
                if printed >= 10:  # limit to 10 printed triples
                    break
        if printed >= 10:
            break
    if printed >= 10:
        break

# --- Testing specific products after Stage 1 ---
# print("\nTest even-even products:")
# print("E * F =", {Z: beta[('E','F', Z)] for Z in even})

# --- Debugging specific associator computation ---
# print("\nDebugging assoc_sym for (E, E, F):")
# a1 = assoc_sym('E', 'H', 'E')
# a2 = assoc_sym('H', 'E', 'E')
# print("a1:", a1)
# print("a2:", a2)

# print("\n=== Debugging eqs2 ===")
# for i, eq in enumerate(eqs2[:10]):  # Print the first 10 equations
#     print(f"[{i}] {eq}")

# --- B: Linear Reduction and Groebner over GF(3) or QQ ---

import sympy as sp
from sympy.polys.polytools import groebner
from sympy.polys.domains import GF

# --- ensuring eqs2 is a list of polynomial expressions (lhs) ---
polys = []
for eq in eqs2:
    # case 1: equation Eq(lhs, rhs)
    if isinstance(eq, sp.Equality):
        poly = sp.simplify(eq.lhs - eq.rhs)
        
    # case 2: plain boolean  or truth value
    elif isinstance(eq, bool):
        if eq is True:
            continue
        elif eq is False:
            polys.append(sp.Integer(1))  # inconsistent
        else: 
            continue
    # case 3: plain expression already (Expr)
    elif isinstance(eq, sp.Expr):
        poly = sp.simplify(eq)
    
    # anything else
    else:
        continue
    
    # keeping only nonzero polynomials
    if poly != 0:
        polys.append(sp.expand(poly))
polys = [p for p in polys]  # list of SymPy expressions representing polynomial = 0

print("Total nontrivial scalar polynomials (polys):", len(polys))

# --- unknowns2: the free parameters not fixed by sol0 ---
# IMPORTANT: Burde's matrices contain bare parameter symbols (u, v for structure 1;
# g, ginv for structure 2) as matrix entries, which end up as values in beta.
# These must be excluded — they are known parameters, not unknowns.
if structure_version == 1:
    _param_syms = {sp.Symbol('u'), sp.Symbol('v')}
elif structure_version == 2:
    _param_syms = {sp.Symbol('g'), sp.Symbol('ginv')}
else:
    _param_syms = set()

unknowns2 = sorted(
    [s for s in set(beta.values()) 
     if s not in sol0 and isinstance(s, sp.Symbol) and s not in _param_syms],
    key=str
)
print("Free unknowns (count):", len(unknowns2))
print("Free unknowns:", [str(s) for s in unknowns2])

# Substituting ginv back to 1/g in polynomials if structure_version == 2
if structure_version == 2:
    polys = [p.subs(ginv, 1/g) for p in polys]
    
# --- Separating linear vs nonlinear polynomials ---
linear_polys = []
nonlinear_polys = []
for p in polys:
    if unknowns2:
        # exclude 'g' only if structure_version == 2
        if structure_version == 2:
            polyobj = sp.Poly(sp.expand(p), *[u for u in unknowns2 if u != g], domain=sp.EX)
        else: 
            polyobj = sp.Poly(sp.expand(p), *unknowns2, domain=sp.EX)
        deg = polyobj.total_degree()
    else:
        deg = 0
    if deg <= 1:
        linear_polys.append(p)
    else:
        nonlinear_polys.append(p)

print("Linear polys:", len(linear_polys))
print("Nonlinear polys:", len(nonlinear_polys))

# --- Solve linear part (symbolically using sp.solve to preserve b variable names) ---
# NOTE: We use sp.solve instead of linsolve because linsolve introduces
# tau0, tau1, ... symbols for free parameters. sp.solve returns a dict
# {b_solved: expr_in_other_b_vars} which keeps the original variable names.
lin_subs = {}
if linear_polys:
    print(f"Solving {len(linear_polys)} linear equations for {len(unknowns2)} unknowns...")
    lin_sol = sp.solve([sp.Eq(p, 0) for p in linear_polys], unknowns2, dict=True, simplify=True)
    
    if lin_sol:
        lin_subs = lin_sol[0]  # dict: {b_solved: expr in remaining free b's and params}
        print(f"Linear subsystem solved: {len(lin_subs)} variables expressed in terms of others.")
        
        # Show the solved variables
        print("\n=== Linear reduction (solved b variables) ===")
        for k, v in sorted(lin_subs.items(), key=lambda x: str(x[0])):
            print(f"  {k} = {v}")
        
        # Identify remaining free b variables
        solved_vars = set(lin_subs.keys())
        free_b_vars = sorted([s for s in unknowns2 if s not in solved_vars], key=str)
        print(f"\nFree b variables after linear reduction: {len(free_b_vars)}")
        print(f"  {[str(s) for s in free_b_vars]}")
    else:
        print("No linear solution (inconsistent linear subsystem or no equations).")
    
# -------------------------------------------------------
# --- Substitute linear solution into nonlinear polys ---
# -------------------------------------------------------
if lin_subs:
    reduced_polys = [sp.simplify(p.subs(lin_subs)) for p in nonlinear_polys]
else:
    reduced_polys = nonlinear_polys[:]

# clear any polynomials that became 0
reduced_polys = [sp.expand(p) for p in reduced_polys if not sp.simplify(p) == 0]
print("Remaining nonlinear polys after linear sub:", len(reduced_polys))

print("\n=== POLYNOMIAL ANALYSIS STAGE ===")

# Step 1: Inspect what nonlinear equations we actually have
print(f"Total reduced scalar equations (nonlinear polys): {len(reduced_polys)}")
for i, p in enumerate(reduced_polys[:10]):   # print only first 10
    print(f"[{i}] {sp.simplify(p)}")

# -----------------------------------------------------------
# -------- Saving polynomial equations to text file ---------
# -----------------------------------------------------------

# Step 2: Save *all* polynomial equations to a text file
# export_path = "osp1_2_LSS_equations_lss2.txt"
# with open(export_path, "w") as f:
#     f.write("=== All polynomial equations (expanded) ===\n\n")
#     for i, p in enumerate(reduced_polys):
#         f.write(f"[{i}] {sp.expand(p)}\n")
# print(f"\nExported all {len(reduced_polys)} equations to file: {export_path}")

# Collect unique τ symbols from reduced_polys
# tau_syms = {s for eq in reduced_polys for s in eq.free_symbols if str(s).startswith("tau")}
# tau_syms = sorted(list(tau_syms), key=lambda s: int(''.join(ch for ch in str(s) if ch.isdigit())))

# print("\n=== Unique τ symbols found in reduced polynomials ===")
# for tau in tau_syms:
#     print(tau)

# tau_dependencies = {}
# for tau in tau_syms:
#     for eq in reduced_polys:
#         if tau in eq.free_symbols:
#             if str(tau) not in tau_dependencies:
#                 tau_dependencies[str(tau)] = []
#             tau_dependencies[str(tau)].append(eq)
            
# mapping_file = "/Users/josemarialoza/capstone/tau_dependencies.txt"
# with open(mapping_file, "w") as f:
#     f.write("τ Dependencies\n")
#     f.write("===========================\n\n")
#     for tau, eqs in tau_dependencies.items():
#         f.write(f"{tau}:\n")
#         for eq in eqs:
#             f.write(f"  {eq}\n")
#         f.write("\n")

# print(f"Dependencies exported successfully to: {mapping_file}")

# ------------------------------------------------------------------
# --- Counting frequencies of free coefficients in reduced_polys ---
# ------------------------------------------------------------------

print("\n=== Frequency of free coefficients in reduced polynomials ===") 

freq = Counter()
for p in reduced_polys:
    for s in p.free_symbols:
        freq[str(s)] += 1
        
# Showing top 5 most frequent free coefficients 
top = freq.most_common(5)
print('Top 5 most frequent free coefficients (symbol, count):')
for symb, count in top:
    print(symb, count)
    
# --- End frequency analysis ---

# Top 5 most frequent free coefficients (symbol, count):
# tau2 163 # numerical
# tau5 82 # in terms of v
# tau9 62
# tau17 62
# tau16 60

# --------------------------------------------------------
# --- Test fixing one parameter (most frequent) ---
# --------------------------------------------------------

def try_fix_parameter(param_name, candidates=None, field='QQ', tau_map=None, rewrite_map=None, unknowns=None, polys=None, export_dir='/Users/josemarialoza/capstone', g=None, ginv=None):
    """_summary_

    Args:
        param_name : str e.g. 'tau2', or 'b_E_Tp_Tm'
        candidates : iterable of values to try. Defaults to None.
            - field=='QQ': candidates = [-2, -1, 0, 1, 2]
            - field='GF3': candidates = [0, 1, 2]
        field : 'QQ' or 'GF3'. Defaults to 'QQ'.
        tau_map : dictionary mapping tau symbol (string or Sym) -> list of b_variables (Sym) or expression. Defaults to None.
        rewrite_map : dictionary mapping tau Sym to b Sym or expressions for rewriting polys. Defaults to None.
        unknowns : list of Sym. Defaults to None.
        polys : list of Sym expressions (reduced polys). Defaults to None.
    """
    if polys is None:
        polys = reduced_polys
    
    if unknowns is None: 
        unknowns = unknowns2
        
    # ensuring g and ginv are provided
    if g is None and ginv is None:
        raise ValueError('Both g and ginv must be provided for structure_version == 2')

    # parsing parameter as sym
    param_sym = sp.Symbol(param_name) if isinstance(param_name, str) else param_name
    
    if candidates is None: 
        if field == 'GF3': 
            candidates = [0, 1, 2]
        else: 
            candidates = [-2, -1, 0, 1, 2]
            
    # Building a rewrite mapping for taus if provided (so polys are expressed in b_X_Y_Z)
    if rewrite_map is None: 
        rewrite_map = {}
        if tau_map is not None: 
            # tau_map: dict(tauSym --> list of b_Syms) or tauSym -> expression
            for t, val in tau_map.items():
                # if val is list of b symbols, replace tau by single corresponding b symbol if unique 
                if isinstance(val, (list, tuple)) and len(val) == 1: 
                    rewrite_map[t] = val[0]
                elif isinstance(val, (list, tuple)) and len(val) > 1:
                    # if we cannot directly map one tau to multiple bs, choose first b
                    primary = val[0]
                    rewrite_map[t] = primary
                    
                    # rewrite all other b's to that same symbol
                    for other in val[1:]:
                        rewrite_map[other] = primary
                else: 
                    # val may be an expression (string or sym)
                    rewrite_map[t] = val
    
    # rewriting polys
    polys_b = [sp.expand(p.xreplace(rewrite_map)) for p in polys]
    
    # ensuring substitution of 1/g back to ginv if structure_version == 2
    if structure_version == 2:
        polys_b = [p.subs(1/g, ginv) for p in polys_b]
        
    # Exporting rewritten polynomials using only b_coefficients 
    export_polys_file = f"{export_dir}/polynomials_in_b_{param_name}_lss_{structure_version}.txt"
    with open(export_polys_file, 'w') as f: 
        f.write("Rewritten polynomial system (only b_X_Y_Z coefficients):\n\n")
        for p in polys_b:
            f.write(str(p) + "\n")
    print(f"Rewritten polynomials exported to {export_polys_file}")
    
    results = []
    for v in candidates: 
        # build substitution mapping 
        subs = {}
        
        # now substituting candidate value for parameter
        subs[param_sym] = sp.Integer(v) if field=='GF3' else sp.Rational(v)
        polys_spec = [sp.expand(p.subs(subs)) for p in polys_b]
        
        # ensure no stray extra symbols remain other than the unknowns; record what is leftover
        leftover = sorted({str(s) for p in polys_spec for s in p.free_symbols if str(s) not in {str(u) for u in unknowns}})
        
        print(f"\nTrying {param_name} = {v} (leftover symbols after substituting: {len(leftover)} ; sample: {leftover[:10]})")
        
        # if field GF3, try Groebner mod 3, else QQ
        try: 
            if field == 'GF3':
                G = groebner(polys_spec, *unknowns, order='lex', modulus=3)
                
            else: 
                G = groebner(polys_spec, *unknowns, order='lex', domain=sp.QQ)
                
            inconsistent = any(g == 1 for g in G)
            results.append((v, not inconsistent, G if not inconsistent else None))
            print(f"    Groebner computed (size {len(G)}). Consistent? {not inconsistent}")
            
        except Exception as e: 
            print("     Groebner computation failed:", e)
            results.append((v, False, None))
            
    # exporting results summary
    outtxt = f"{export_dir}/fix_param_{param_name}_{field}_results.txt"
    with open(outtxt, 'w') as f: 
        f.write(f"Fix parameter {param_name} results field={field}\n")
        for v, ok, G in results: 
            f.write(f"Value {v} --> consistent: {ok}\n")
            if ok and G is not None: 
                f.write("Sample Groebner basis elements:\n")
                for g in list(G)[:10]:
                    f.write(str(g) + "\n")
                f.write('\n')
    print(f"Results exported to {outtxt}")
    return results

# Replacing 1/g back to ginv if structure_version == 2
if structure_version == 2: 
    polys = [p.subs(1/g, ginv) for p in polys]
    reduced_polys = [p.subs(1/g, ginv) for p in reduced_polys]
    
# calling function
#try_fix_parameter('b_Tp_H_Tm', field='QQ', tau_map=None, rewrite_map=None, unknowns=unknowns2, polys=reduced_polys, g=g, ginv=ginv)

# --- Trying Groebner Solver ---:
# We use reduced_polys (after linear pre-reduction) for performance,
# and recompute the variable lists from the actual free symbols present
# in reduced_polys. Since we used sp.solve (not linsolve), all variables
# are still in b_X_Y_Z form — no tau symbols.

use_char3 = True   # set to True if working over GF(3); otherwise False for QQ
    
if reduced_polys:
    # --- Recompute unknowns and params from actual free symbols in reduced_polys ---
    all_free = sorted(
        set().union(*[f.free_symbols for f in reduced_polys]),
        key=lambda s: s.name
    )
    
    if structure_version == 1:
        param_names = {'u', 'v'}
    elif structure_version == 2:
        param_names = {'g', 'ginv'}
    else:
        param_names = set()
    
    # unknowns: b_X_Y_Z symbols only
    gb_unknowns = [s for s in all_free if s.name not in param_names]
    # parameters: u,v or g,ginv
    gb_params = [s for s in all_free if s.name in param_names]
    
    # For structure 2: preprocess_polys_in_char_p will add g*ginv - 1, so both
    # g and ginv MUST be in gens_all even if one is absent from reduced_polys.
    # (e.g., if all g terms were eliminated during linear reduction, but ginv remains,
    # the invertibility relation still needs g as a ring generator.)
    if structure_version == 2:
        gb_param_names = {s.name for s in gb_params}
        if 'g' not in gb_param_names:
            gb_params.append(g)
        if 'ginv' not in gb_param_names:
            gb_params.append(ginv)
        gb_params = sorted(gb_params, key=lambda s: s.name)
    
    # Put unknowns first for lex order elimination behavior
    gens_all = gb_unknowns + gb_params
    
    print(f"\nGroebner computation setup:")
    print(f"  Unknowns ({len(gb_unknowns)}): {[s.name for s in gb_unknowns]}")
    print(f"  Parameters ({len(gb_params)}): {[s.name for s in gb_params]}")
    print(f"  Total ring generators: {len(gens_all)}")
    print(f"  Total polynomials: {len(reduced_polys)}")

    if use_char3:
        print("\nReducing coefficients mod 3 and computing Groebner basis over GF(3)...")

        try:
            if structure_version == 1:
                polys_for_gb = list(reduced_polys)
                polys_mod3 = preprocess_polys_in_char_p(polys_for_gb, 3, gens_all, invert_pairs=None)

            elif structure_version == 2:
                polys_for_gb = list(reduced_polys) 
                polys_mod3 = preprocess_polys_in_char_p(polys_for_gb, 3, gens_all, invert_pairs=[(g, ginv)])
                
            else:
                raise ValueError("Unknown structure; expected 1 or 2")
            
            # Sanity check: no remaining denominators
            for i, p in enumerate(polys_mod3):
                num, den = sp.fraction(sp.together(p))
                if den != 1:
                    print(f"WARNING: denominator {den} in equation {i}: {p}")
                    break
            
            # Compute Groebner basis over GF(3)
            G = groebner(polys_mod3, *gens_all, order='lex', domain=sp.GF(3), method='buchberger')

            print("Groebner basis size:", len(G))
            inconsistent = any(gb == 1 for gb in G)
            print("Inconsistent (1 in basis)?", inconsistent)
            print(G)

        except Exception as e:
            print("Groebner(mod 3) failed:", e)

    else:
        print("\nComputing Groebner basis over QQ ...")
        try:
            G = groebner(reduced_polys, *gens_all, order='lex', domain=sp.EX, method='buchberger')
            print("Groebner basis size:", len(G))
            print(G)
        except Exception as e:
            print("Groebner(QQ) failed:", e)

#----------------------------------------------------------------------------------------------------        
# --- Exporting polynomials for SageMath verification ---
#----------------------------------------------------------------------------------------------------

# Since we use sp.solve (not linsolve), all variables remain in b_X_Y_Z form.
# No tau → b rewriting is needed. We export polys_mod3 directly.

print("\n=== Exporting polynomials for SageMath ===")

# Use polys_mod3 from the Groebner block if available, otherwise reduced_polys
if 'polys_mod3' in dir() and polys_mod3:
    polys_for_export = polys_mod3
elif reduced_polys:
    polys_for_export = reduced_polys
else:
    polys_for_export = polys

# Collect free symbols
all_free_syms = sorted(
    set().union(*[p.free_symbols for p in polys_for_export]),
    key=lambda s: s.name
)

print(f"Free symbols present: {[str(s) for s in all_free_syms]}")

if structure_version == 1:
    param_syms   = [s for s in all_free_syms if str(s) in ('u', 'v')]
    unknown_syms = [s for s in all_free_syms if str(s) not in ('u', 'v')]
elif structure_version == 2:
    param_syms   = [s for s in all_free_syms if str(s) in ('g', 'ginv')]
    unknown_syms = [s for s in all_free_syms if str(s) not in ('g', 'ginv')]

var_names_export = [str(s) for s in param_syms + unknown_syms]

print(f"Parameters ({len(param_syms)}):  {[str(s) for s in param_syms]}")
print(f"Unknowns  ({len(unknown_syms)}): {[str(s) for s in unknown_syms]}")
print(f"Total ring variables for Sage: {len(var_names_export)}")

# --- Export ---
export_path = f'/Users/josemarialoza/capstone/polys_files_sage/polys_for_sage_lss{structure_version}.py'
with open(export_path, 'w') as f:
    f.write(f"# Auto-generated for SageMath — Structure {structure_version}\n")
    f.write(f"# All variables in b_X_Y_Z form (no tau symbols).\n")
    f.write(f"# Parameters: {[str(s) for s in param_syms]}\n")
    f.write(f"# Unknowns:   {[str(s) for s in unknown_syms]}\n\n")
    f.write(f"var_names = {repr(var_names_export)}\n\n")
    f.write("polys_raw = [\n")
    for p in polys_for_export:
        f.write(f"    {repr(str(p))},\n")
    f.write("]\n")

print(f"\nExported to {export_path}")

#----------------------------------------------------------------------------------------------------
            
# --- Fallback: try randomized finite-field search if GF(3) and number of free vars smallish ---
if use_char3 and reduced_polys:
    K = len(unknowns2)
    if K <= 12:
        import itertools
        print("\nAttempting exhaustive search over GF(3) (this may be slow if K>10)")
        found = False
        for vals in itertools.product(range(3), repeat=K):
            subs = {sym: sp.Integer(v) for sym, v in zip(unknowns2, vals)}
            ok = True
            for p in reduced_polys:
                val = sp.expand(p.subs(subs))  
                val_mod3 = sp.Mod(val, 3)
                if sp.simplify(val_mod3) != 0:
                    ok = False
                    break
            if ok:
                print("Found GF(3) solution:", subs)
                found = True
                break
        if not found:
            print("No GF(3) solution found by exhaustive search.")
    else:
        print("K too large for brute force; try random trials with limited budget.")
        import random
        found = False
        for t in range(500):
            subs = {sym: random.randint(0,2) for sym in unknowns2}
            ok = True
            for p in reduced_polys:
                val = sp.expand(p.subs(subs))
                val_mod3 = sp.Mod(val, 3)
                if sp.simplify(val_mod3) != 0:
                    ok = False
                    break
            if ok:
                print("Random GF(3) solution found:", subs)
                found = True
                break
        if not found:
            print("No random GF(3) solution found in 500 trials.")

# print("\nDebugging star_sym for (F, E):")
# fe = star_sym('F', 'E')
# print("F * E =", fe)

# --- Debugging star_sym for (E, t) for each t in F * E ---
# print("\nDebugging star_sym for (E, t) for each t in F * E:")
# for t, coeff in fe.items():
#     et = star_sym('E', t)
#     print(f"E * {t} (coeff={coeff}):", et)
    
# print("\nBeta values for (F, E, z):")
# for z in gens:
#     print(f"beta[('F', 'E', '{z}')]: {beta.get(('F', 'E', z), 'Not Found')}")

# --------------------------------------------------------------------------
# ------------ Factor analysis of reduced polynomials ---------------
# --------------------------------------------------------------------------

# Defining helper function
def factor_in_b_only(expr, b_vars):
    """
    Factor expr into factors involving only b_vars.
    All other symbols that are not variables (e.g., g, ginv) are treated as constants.
    """
    
    polynomial = sp.Poly(expr, *b_vars, domain=sp.EX)
    coeff, factors = polynomial.factor_list()
    return coeff, factors

print('\n=== Factor analysis of reduced polynomials ===')

factor_sets = []
for poly in reduced_polys: 
    try: 
        coeff, factors = factor_in_b_only(poly, unknowns2)
        factor_sets.append([f for f, _ in factors])
    except Exception as e: 
        print('Factoring failed for:', poly)
        print('Reason:', e)
    
# Inspecting factors
print("\nSample factorizations:")
for i, factors in enumerate(factor_sets[:5]):  # print first 10
    print(f"[{i}] Factors:")
    for f in factors:
        print("   ", f)
        
# printing some elements of unknowns2 for inspection
print("\nSample unknowns2 (b variables):")
for u in unknowns2[:10]:  # print first 10
    print("   ", u)
    
print(f"\n=== Structure {structure_version}: Sample reduced polynomials ===")
for i, p in enumerate(reduced_polys[:10]):
    print(f"  [{i}] {sp.expand(p)}")