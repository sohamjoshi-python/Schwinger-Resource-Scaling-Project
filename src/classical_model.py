"""Classical baselines for the fitting-vs-prediction comparison.

The point (per Ikeda's critique): does a classical model of EQUAL parameter cost
predict the condensate as well as the QCL circuit? If yes, QCL is "just fitting."

Single entry point: classical_baselines(data) -> list of result rows
(model, n_params, test_mse_mean, test_mse_std), directly comparable to the QCL
phase2.csv schema so it can be overlaid on the same Pareto plot.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


def _xy(pairs):
    m = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([np.real(p[1]) for p in pairs], dtype=float)
    return m, y


def _poly_row(train, test, degree):
    m_tr, y_tr = _xy(train)
    m_te, y_te = _xy(test)
    coeffs = np.polyfit(m_tr, y_tr, degree)
    preds = np.polyval(coeffs, m_te)
    mse = float(np.mean((preds - y_te) ** 2))
    # a degree-d polynomial has d+1 free coefficients
    return {"model": f"poly_deg{degree}", "n_params": degree + 1,
            "test_mse_mean": mse, "test_mse_std": 0.0}


def _nn_row(train, test, hidden, seeds):
    m_tr, y_tr = _xy(train)
    m_te, y_te = _xy(test)
    X_tr, X_te = m_tr.reshape(-1, 1), m_te.reshape(-1, 1)
    mses = []
    for s in seeds:
        model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=5000,
                             random_state=s)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        mses.append(np.mean((preds - y_te) ** 2))
    # single hidden layer of size h: (1*h weights + h biases) + (h*1 weights + 1 bias)
    h = hidden[0]
    n_params = (1 * h + h) + (h * 1 + 1)
    return {"model": f"nn_h{h}", "n_params": n_params,
            "test_mse_mean": float(np.mean(mses)),
            "test_mse_std": float(np.std(mses))}


def classical_baselines(data, test_size=0.2, random_state=42,
                        poly_degrees=range(1, 9),
                        nn_hidden_sizes=((3,), (4,), (6,)),
                        nn_seeds=(0, 1, 2)):
    """Run all classical baselines on the (m, condensate) dataset.

    Uses the SAME split convention as the QCL sweep (test_size=0.2,
    random_state=42) so the comparison is fair.

    Returns a list of dict rows with keys:
        model, n_params, test_mse_mean, test_mse_std
    """
    train, test = train_test_split(data, test_size=test_size,
                                   random_state=random_state)
    rows = []
    for d in poly_degrees:
        rows.append(_poly_row(train, test, d))
    for hidden in nn_hidden_sizes:
        rows.append(_nn_row(train, test, hidden, nn_seeds))
    return rows