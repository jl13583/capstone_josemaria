if reduced_polys:
    if use_char3:
        print("\nComputing Groebner basis over sp.EX ... (may take time)")
        try:
            G = groebner(reduced_polys, *unknowns2, order='lex', domain=sp.EX, method='buchberger')
            print("Groebner basis size:", len(G))
            # check for 1 in basis => inconsistent
            if any(g == 1 for g in G):
                print("Groebner contains 1: system inconsistent over sp.EX — NO solution.")
                print(G)
            else:
                print("Groebner basis (sample):")
                for i,g in enumerate(G[:min(len(G),8)]):
                    print(i, g)
        except Exception as err:
            print("Groebner failed:", err)
    else:
        print("Computing Groebner basis over QQ ... (may be heavy)")
        try:
            G = groebner(reduced_polys, *unknowns2, order='lex', domain=sp.QQ, method='buchberger')
            print("Groebner basis size:", len(G))
            if any(g == 1 for g in G):
                print("Groebner contains 1: system inconsistent over QQ.")
            else:
                print("Groebner basis (sample):")
                for i,g in enumerate(G[:min(len(G),8)]):
                    print(i, g)
        except Exception as err:
            print("Groebner failed:", err)
else:
    print("No nonlinear polys remaining to Groebner (all constraints linear and solved).")
