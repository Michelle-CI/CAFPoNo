import cdt.causality.pairwise as cdt

def RECI(X, Y, args=None):
    model = cdt.RECI(degree=args.degree)
    X, Y = map(lambda x: x.reshape(-1, 1), (X, Y))
    score_XY = model.b_fit_score(X, Y)
    score_YX = model.b_fit_score(Y, X)
    return score_XY, score_YX
