from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
DEMO_PATH = ROOT / "data" / "ventas_ade_demo.csv"

def load_demo():
    return pd.read_csv(DEMO_PATH)

def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def clean_model_data(df, y, xs):
    cols = [y] + list(xs)
    out = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    return out

def fit_ols(df, y, xs, robust=False):
    dat = clean_model_data(df, y, xs)
    X = sm.add_constant(dat[list(xs)], has_constant="add")
    model = sm.OLS(dat[y], X).fit()
    if robust:
        model = model.get_robustcov_results(cov_type="HC1")
    return model, dat, X

def coef_table(model):
    names = list(model.model.exog_names)
    conf = np.asarray(model.conf_int())
    return pd.DataFrame({
        "Variable": names,
        "Coeficiente": np.asarray(model.params),
        "Error típico": np.asarray(model.bse),
        "t": np.asarray(model.tvalues),
        "p-valor": np.asarray(model.pvalues),
        "IC 95% inf.": conf[:, 0],
        "IC 95% sup.": conf[:, 1],
    })

def equation_text(model, y):
    names = list(model.model.exog_names)
    params = np.asarray(model.params)
    terms = []
    for name, b in zip(names, params):
        if name == "const":
            terms.append(f"{b:.3f}")
        else:
            sign = "+" if b >= 0 else "-"
            terms.append(f" {sign} {abs(b):.3f}·{name}")
    return f"{y} = " + "".join(terms)

def gretl_script(filename, y, xs, robust=False):
    regressors = " ".join(xs)
    robust_line = "\nmodtest --white" if robust else ""
    return f"""# Script generado por Econometría Interactiva
open {filename}

# Descriptivos
summary {y} {regressors}
corr {y} {regressors}

# MCO
ols {y} const {regressors}

# Diagnóstico básico
modtest --normality
modtest --squares
modtest --white
vif
{robust_line}

# Guarda el script con extensión .inp para reproducibilidad
"""
