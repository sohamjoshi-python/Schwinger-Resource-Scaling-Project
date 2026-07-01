"""Classical baselines for the QCL-vs-classical comparison.

Two families:
  1-D (input = single scalar, e.g. mass or time):  classical_baselines(data)
  2-D (input = (t, m) vectors):                     classical_baselines_2d(X, y, ...)

Every model reports (n_params, test_mse) so it can be overlaid on the QCL
Pareto frontier on equal footing. The point (per Ikeda): does a classical model
of EQUAL cost match QCL? If yes, QCL is "just fitting."
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# ============================ 1-D baselines ============================
def _xy(pairs):
    m = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([np.real(p[1]) for p in pairs], dtype=float)
    return m, y


def _poly_row(train, test, degree):
    m_tr, y_tr = _xy(train); m_te, y_te = _xy(test)
    coeffs = np.polyfit(m_tr, y_tr, degree)
    mse = float(np.mean((np.polyval(coeffs, m_te) - y_te) ** 2))
    return {"model": f"poly_deg{degree}", "n_params": degree + 1,
            "test_mse_mean": mse, "test_mse_std": 0.0}


def _nn_row(train, test, hidden, seeds):
    m_tr, y_tr = _xy(train); m_te, y_te = _xy(test)
    X_tr, X_te = m_tr.reshape(-1, 1), m_te.reshape(-1, 1)
    mses = []
    for s in seeds:
        mdl = MLPRegressor(hidden_layer_sizes=hidden, max_iter=5000, random_state=s)
        mdl.fit(X_tr, y_tr)
        mses.append(np.mean((mdl.predict(X_te) - y_te) ** 2))
    h = hidden[0]
    n_params = (1 * h + h) + (h * 1 + 1)
    return {"model": f"nn_h{h}", "n_params": n_params,
            "test_mse_mean": float(np.mean(mses)), "test_mse_std": float(np.std(mses))}


def classical_baselines(data, test_size=0.2, random_state=42,
                        poly_degrees=range(1, 9),
                        nn_hidden_sizes=((3,), (4,), (6,)), nn_seeds=(0, 1, 2)):
    """1-D baselines on (m, condensate) data. Returns list of result rows."""
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    rows = [_poly_row(train, test, d) for d in poly_degrees]
    rows += [_nn_row(train, test, h, nn_seeds) for h in nn_hidden_sizes]
    return rows


# ============================ 2-D baselines ============================
# For the multi-parameter dynamics target f(t, m). Take pre-split arrays so the
# SAME split (interpolation / parameter-extrap / time-extrap) is shared with QCL.

def poly2d(X_tr, y_tr, X_te, y_te, degree):
    pf = PolynomialFeatures(degree)
    A_tr = pf.fit_transform(X_tr)
    reg = LinearRegression().fit(A_tr, y_tr)
    mse = float(np.mean((reg.predict(pf.transform(X_te)) - y_te) ** 2))
    return {"model": f"poly2d_d{degree}", "n_params": A_tr.shape[1],
            "test_mse_mean": mse, "test_mse_std": 0.0}


def fourier2d(X_tr, y_tr, X_te, y_te, K=4, w0_t=0.6, w0_m=2.0):
    """Fixed-frequency 2-D Fourier features. Expected to struggle because the
    true frequencies move with m -- the honest structure-mismatched foil."""
    def feats(X):
        t, m = X[:, 0], X[:, 1]
        cols = [np.ones_like(t)]
        for k in range(1, K + 1):
            cols += [np.cos(k * w0_t * t), np.sin(k * w0_t * t),
                     np.cos(k * w0_m * m), np.sin(k * w0_m * m)]
        return np.column_stack(cols)
    A_tr = feats(X_tr)
    reg = LinearRegression().fit(A_tr, y_tr)
    mse = float(np.mean((reg.predict(feats(X_te)) - y_te) ** 2))
    return {"model": f"fourier2d_K{K}", "n_params": A_tr.shape[1],
            "test_mse_mean": mse, "test_mse_std": 0.0}


def nn2d(X_tr, y_tr, X_te, y_te, hidden=(16, 16), seeds=(0, 1, 2)):
    """Small neural net -- the SERIOUS competitor on the 2-D oscillatory target."""
    mses = []
    for s in seeds:
        mdl = MLPRegressor(hidden_layer_sizes=hidden, max_iter=8000, random_state=s)
        mdl.fit(X_tr, y_tr)
        mses.append(np.mean((mdl.predict(X_te) - y_te) ** 2))
    sizes = [2] + list(hidden) + [1]
    n_params = sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))
    return {"model": f"nn2d_{'x'.join(map(str, hidden))}", "n_params": n_params,
            "test_mse_mean": float(np.mean(mses)), "test_mse_std": float(np.std(mses))}


def classical_baselines_2d(X_tr, y_tr, X_te, y_te,
                           poly_degrees=(3, 4, 6), fourier_K=(3, 5),
                           nn_hidden_sizes=((8, 8), (16, 16))):
    """Run all 2-D baselines on a GIVEN split. Returns list of result rows."""
    rows = [poly2d(X_tr, y_tr, X_te, y_te, d) for d in poly_degrees]
    rows += [fourier2d(X_tr, y_tr, X_te, y_te, K=k) for k in fourier_K]
    rows += [nn2d(X_tr, y_tr, X_te, y_te, hidden=h) for h in nn_hidden_sizes]
    return rows