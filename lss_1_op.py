import sympy as sp 

u, v = sp.symbols('u v')
w = v - u

# defining Burde's LSS 1
lamE = sp.Matrix([[0, u, 1], 
                [0, w, -1], 
                [1, w**2 +1, -w]])

lamF = sp.Matrix([[u, u - v*w**2, v*w], 
                [w, v, 1], 
                [w**2, -v*w, -u - v]])

lamH = sp.Matrix([[0, v*w, -v],
                [-1, -1, 0],
                [-w, -u - v, 1]])

# store items in list for indexing

lams = [lamE, lamF, lamH]
names = ['E', 'F', 'H']

# standard basis vectors k^3
basis_vecs = [sp.Matrix([1,0,0]), sp.Matrix([0,1,0]), sp.Matrix([0,0,1])]

# LS product: x*y = λ_x * (basis vector of y or coordinate vector)
def star(x, y):
    return lams[x] * basis_vecs[y]

# associator (X, Y, Z)
def assoc(x, y, z, verbose=False):
    xy = star(x, y)
    if verbose: 
        print('xy =', xy)
    # (x*y)*z
    term1 = sum((xy[i] * star(i,z) for i in range(3)), sp.zeros(3,1))
    if verbose:
        print('term1 =', term1)
    yz = star(y, z)
    if verbose:
        print('yz =', yz)
    # x*(y*z)
    term2 = sum((yz[i] * star(x,i) for i in range(3)), sp.zeros(3,1))
    if verbose:
        print('term2 =', term2)
    return sp.simplify(term1 - term2)

# difference D(X, Y, Z) = (X, Y, Z) - (Y, X, Z)
def D(x, y, z):
    return sp.simplify(assoc(x, y, z) - assoc(y, x, z))

# running checks for all triples 
cnt = 0
# for x in range(3):
#     for y in range(3):
#         for z in range(3):
#             cnt += 1
#             print(f'Check {cnt}:')
#             print(f'Checking D({names[x]}, {names[y]}, {names[z]})')
#             d = D(x, y, z)
#             print("(x,y,z) =", assoc(x, y, z))
#             print("(y,x,z) =", assoc(y, x, z)) 
#             print('Result =', d)
#             print('Is zero:', d == sp.Matrix([0,0,0]))
#             print(f"Total checks so far: {cnt}")
#             print('---')