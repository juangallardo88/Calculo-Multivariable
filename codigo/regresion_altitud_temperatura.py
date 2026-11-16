"""
Proyecto de Cálculo Multivariable — UCC 2026B
Regresión lineal y optimización mediante funciones de varias variables

Problema real: ¿cómo cambia la temperatura media anual con la altitud
en las ciudades de Colombia?

    x = altitud de la estación meteorológica (km sobre el nivel del mar)
    y = temperatura media anual (°C), normales climatológicas del IDEAM

Modelo: y = a·x + b
Función de error: S(a, b) = Σ (y_i − a·x_i − b)²

Este programa:
  1. carga los datos,
  2. calcula a* y b* con NUESTRO PROPIO método (ecuaciones normales que
     dedujimos a mano a partir de ∇S = 0),
  3. calcula S(a*, b*),
  4. compara con funciones de librería (numpy y scipy),
  5. construye todas las gráficas (dispersión, rectas, superficie, trazas,
     curvas de nivel, residuos, histograma),
  6. calcula y verifica numéricamente el Hessiano,
  7. evalúa la calidad del ajuste (RMSE y R²),
  8. hace los análisis adicionales A, B, C y D.

Ejecutar:  python regresion_altitud_temperatura.py
Requiere:  numpy, pandas, matplotlib, scipy
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # genera las imágenes sin abrir ventanas
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats

CARPETA = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(CARPETA, "figuras")
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------
# Estilo de las gráficas (colores consistentes en todo el proyecto)
# ----------------------------------------------------------------------------
AZUL, NARANJA, VERDE, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
TINTA, TINTA2, GRIS = "#0b0b0b", "#52514e", "#b9b8b3"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 11,
    "axes.edgecolor": GRIS, "axes.labelcolor": TINTA, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "legend.frameon": False,
})


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, nombre), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================================
# 1. CARGAR LOS DATOS
# ============================================================================
datos = pd.read_csv(os.path.join(CARPETA, "..", "datos", "datos_ciudades.csv"))
ciudades = datos["ciudad"].tolist()
x = (datos["altitud_m"] / 1000).to_numpy()    # metros -> kilómetros
y = datos["temperatura_C"].to_numpy()
n = len(x)

print("=" * 70)
print("1. DATOS  (n =", n, ")")
print("=" * 70)
print(datos[["ciudad", "altitud_m", "temperatura_C"]].to_string(index=False))


# ============================================================================
# 2. MÉTODO PROPIO: ecuaciones normales obtenidas con ∇S(a, b) = 0
# ============================================================================
#   ∂S/∂a = −2 Σ x_i (y_i − a x_i − b) = 0   →   a Σx² + b Σx = Σxy
#   ∂S/∂b = −2 Σ   (y_i − a x_i − b) = 0   →   a Σx  + b n  = Σy
#
#   Sistema 2×2 que resolvemos con la regla de Cramer:
#        D = n Σx² − (Σx)²
#        a* = (n Σxy − Σx Σy) / D
#        b* = (Σy − a* Σx) / n

def sumas(x, y):
    """Calcula las sumas que aparecen en las ecuaciones normales (con un ciclo,
    igual que se hace a mano en la tabla)."""
    Sx = Sy = Sxx = Sxy = 0.0
    for xi, yi in zip(x, y):
        Sx += xi
        Sy += yi
        Sxx += xi * xi
        Sxy += xi * yi
    return Sx, Sy, Sxx, Sxy


def ajuste_propio(x, y):
    """Devuelve (a*, b*) resolviendo el sistema de ecuaciones normales."""
    m = len(x)
    Sx, Sy, Sxx, Sxy = sumas(x, y)
    D = m * Sxx - Sx ** 2                  # determinante del sistema
    if D == 0:
        raise ValueError("Todos los x son iguales: no hay una única recta.")
    a = (m * Sxy - Sx * Sy) / D
    b = (Sy - a * Sx) / m
    return a, b


def S(a, b, x=x, y=y):
    """Función de error cuadrático total S(a, b) = Σ (y_i − a x_i − b)²."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    forma = (-1,) + (1,) * a.ndim          # permite evaluar S en una malla de (a, b)
    r = y.reshape(forma) - (a * x.reshape(forma) + b)
    return (r ** 2).sum(axis=0)


def gradiente(a, b):
    """∇S = (∂S/∂a, ∂S/∂b), con las fórmulas deducidas a mano."""
    r = y - a * x - b
    return np.array([-2 * np.sum(x * r), -2 * np.sum(r)])


Sx, Sy, Sxx, Sxy = sumas(x, y)
Syy = float(np.sum(y ** 2))
a_cod, b_cod = ajuste_propio(x, y)
S_cod = float(S(a_cod, b_cod))

print("\n" + "=" * 70)
print("2. MÉTODO PROPIO (ecuaciones normales)")
print("=" * 70)
print(f"Σx = {Sx:.4f}   Σy = {Sy:.4f}   Σx² = {Sxx:.4f}   Σxy = {Sxy:.4f}   Σy² = {Syy:.4f}")
print(f"Sistema:  {Sxx:.4f} a + {Sx:.4f} b = {Sxy:.4f}")
print(f"          {Sx:.4f} a + {n} b = {Sy:.4f}")
print(f"a* = {a_cod:.6f} °C/km     b* = {b_cod:.6f} °C     S(a*,b*) = {S_cod:.6f}")
print(f"Gradiente en (a*, b*): {gradiente(a_cod, b_cod)}  (debe ser ≈ 0)")


# ============================================================================
# 3. HESSIANO: analítico y verificación numérica
# ============================================================================
#   S_aa = 2 Σx²   S_ab = 2 Σx   S_bb = 2n   (constantes: S es cuadrática)
H = np.array([[2 * Sxx, 2 * Sx], [2 * Sx, 2 * n]])
detH = float(np.linalg.det(H))
autovalores = np.linalg.eigvalsh(H)


def hessiano_numerico(f, a0, b0, h=1e-3):
    """Hessiano por diferencias finitas centrales (no usa las fórmulas)."""
    faa = (f(a0 + h, b0) - 2 * f(a0, b0) + f(a0 - h, b0)) / h ** 2
    fbb = (f(a0, b0 + h) - 2 * f(a0, b0) + f(a0, b0 - h)) / h ** 2
    fab = (f(a0 + h, b0 + h) - f(a0 + h, b0 - h)
           - f(a0 - h, b0 + h) + f(a0 - h, b0 - h)) / (4 * h ** 2)
    return np.array([[faa, fab], [fab, fbb]])


H_num = hessiano_numerico(lambda a, b: float(S(a, b)), a_cod, b_cod)

print("\n" + "=" * 70)
print("3. HESSIANO")
print("=" * 70)
print("Analítico:\n", np.round(H, 4))
print("Numérico (diferencias finitas):\n", np.round(H_num, 4))
print(f"S_aa = {H[0,0]:.4f} > 0   y   det(H) = {detH:.4f} > 0  →  MÍNIMO")
print(f"Autovalores: {autovalores[0]:.4f}, {autovalores[1]:.4f} (ambos > 0 → definida positiva)")


# ============================================================================
# 4. COMPARACIÓN CON LIBRERÍAS
# ============================================================================
a_np, b_np = np.polyfit(x, y, 1)                       # numpy.polyfit
reg = stats.linregress(x, y)                           # scipy.stats.linregress
a_sp, b_sp = reg.slope, reg.intercept
A = np.column_stack([x, np.ones(n)])                   # numpy.linalg.lstsq
(a_ls, b_ls), *_ = np.linalg.lstsq(A, y, rcond=None)

# Valores del procedimiento manual (calculados a mano con 4 decimales)
a_man, b_man, S_man = -6.1677, 29.9003, 8.8252

tabla = pd.DataFrame({
    "Parámetro": ["a (°C/km)", "b (°C)", "S_min (°C²)"],
    "Analítico (a mano)": [a_man, b_man, S_man],
    "Código propio": [a_cod, b_cod, S_cod],
    "numpy.polyfit": [a_np, b_np, float(S(a_np, b_np))],
    "scipy.linregress": [a_sp, b_sp, float(S(a_sp, b_sp))],
})
cols = ["Analítico (a mano)", "Código propio", "numpy.polyfit", "scipy.linregress"]
tabla["Diferencia máxima"] = tabla[cols].max(axis=1) - tabla[cols].min(axis=1)
dif_cod_lib = max(abs(a_cod - a_np), abs(b_cod - b_np), abs(a_cod - a_sp), abs(b_cod - b_sp))

print("\n" + "=" * 70)
print("4. COMPARACIÓN DE MÉTODOS")
print("=" * 70)
with pd.option_context("display.float_format", "{:.6f}".format):
    print(tabla.to_string(index=False))
print(f"numpy.linalg.lstsq: a = {a_ls:.6f}, b = {b_ls:.6f}")
print(f"Diferencia máxima código propio vs librerías: {dif_cod_lib:.2e}")


# ============================================================================
# 5. RESIDUOS Y CALIDAD DEL AJUSTE
# ============================================================================
y_pred = a_cod * x + b_cod
e = y - y_pred
RMSE = float(np.sqrt(np.mean(e ** 2)))
SST = float(np.sum((y - y.mean()) ** 2))
R2 = 1 - S_cod / SST
r_pearson = float(np.corrcoef(x, y)[0, 1])

tabla_res = pd.DataFrame({"ciudad": ciudades, "x (km)": x, "y obs (°C)": y,
                          "y pred (°C)": y_pred, "residuo e (°C)": e})
print("\n" + "=" * 70)
print("5. RESIDUOS Y CALIDAD")
print("=" * 70)
with pd.option_context("display.float_format", "{:.3f}".format):
    print(tabla_res.to_string(index=False))
print(f"Suma de residuos = {e.sum():.2e} (≈ 0, consecuencia de ∂S/∂b = 0)")
print(f"Σ x_i e_i = {np.sum(x * e):.2e} (≈ 0, consecuencia de ∂S/∂a = 0)")
print(f"RMSE = {RMSE:.4f} °C    R² = {R2:.4f}    r = {r_pearson:.4f}")


# ============================================================================
# 6. GRÁFICAS
# ============================================================================
xs = np.linspace(0, 3.1, 200)

# --- Fig. 1: diagrama de dispersión -----------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, s=55, color=AZUL, edgecolor="white", linewidth=1.5, zorder=3)
# desplazamientos de etiquetas para que no se monten unas sobre otras
desp = {"Ibagué": (-44, 6), "Cali": (6, 2), "Bucaramanga": (6, -12),
        "Tunja": (-38, 4), "Pasto": (6, -4)}
for c, xi, yi in zip(ciudades, x, y):
    ax.annotate(c, (xi, yi), xytext=desp.get(c, (6, 4)), textcoords="offset points",
                fontsize=9, color=TINTA2)
ax.set_xlabel("Altitud de la estación, x (km)")
ax.set_ylabel("Temperatura media anual, y (°C)")
ax.set_title("Diagrama de dispersión: temperatura vs. altitud (13 ciudades)")
ax.set_xlim(-0.1, 3.3)
guardar(fig, "fig01_dispersion.png")

# --- Fig. 2: recta óptima vs otras rectas -----------------------------------
otras = [(-5.0, 28.0, VERDE), (-7.5, 31.0, VIOLETA), (-6.17, 27.5, GRIS)]
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, s=55, color=TINTA, edgecolor="white", linewidth=1.5, zorder=4,
           label="Datos observados")
ax.plot(xs, a_cod * xs + b_cod, color=AZUL, lw=2.5, zorder=3,
        label=f"Óptima: y = {a_cod:.3f}x + {b_cod:.3f}   S = {S_cod:.2f}")
for aa, bb, col in otras:
    ax.plot(xs, aa * xs + bb, color=col, lw=1.6, ls="--",
            label=f"Otra: y = {aa}x + {bb}   S = {float(S(aa, bb)):.2f}")
ax.set_xlabel("Altitud, x (km)")
ax.set_ylabel("Temperatura media anual, y (°C)")
ax.set_title("La recta óptima tiene el menor error S(a, b)")
ax.legend(fontsize=9, loc="upper right")
guardar(fig, "fig02_rectas_comparadas.png")

# --- malla para superficie y curvas de nivel --------------------------------
aa = np.linspace(-10, -2.3, 220)
bb = np.linspace(23, 37, 220)
AA, BB = np.meshgrid(aa, bb)
ZZ = S(AA, BB)

# --- Fig. 3: superficie de error 3D -----------------------------------------
fig = plt.figure(figsize=(9, 6.5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(AA, BB, ZZ, cmap=cm.Blues_r, alpha=0.6, linewidth=0,
                rstride=4, cstride=4)
ax.contour(AA, BB, ZZ, levels=[20, 60, 150, 300], zdir="z", offset=0,
           colors=TINTA2, linewidths=0.8)
ax.scatter([a_cod], [b_cod], [S_cod], color=NARANJA, s=70, depthshade=False, zorder=10)
ax.scatter([a_cod], [b_cod], [0], color=NARANJA, s=30, depthshade=False)
ax.plot([a_cod, a_cod], [b_cod, b_cod], [0, S_cod], color=NARANJA, lw=1.5)
ax.text2D(0.02, 0.93, f"● Mínimo: a* = {a_cod:.3f},  b* = {b_cod:.3f},  S(a*, b*) = {S_cod:.2f}",
          transform=ax.transAxes, color=NARANJA, fontsize=10, fontweight="bold")
ax.set_xlabel("a (pendiente, °C/km)")
ax.set_ylabel("b (intercepto, °C)")
ax.set_zlabel("S(a, b)")
ax.set_title("Superficie de error z = S(a, b): un paraboloide elíptico")
ax.view_init(elev=28, azim=-58)
guardar(fig, "fig03_superficie.png")

# --- Fig. 4: trazas (cortes por planos verticales) --------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for b_fijo, col in [(b_cod, AZUL), (b_cod - 2, VERDE), (b_cod + 2, VIOLETA)]:
    lab = "b = b* = %.2f" % b_fijo if b_fijo == b_cod else "b = %.2f" % b_fijo
    axs[0].plot(aa, S(aa, b_fijo), color=col, lw=2.2 if b_fijo == b_cod else 1.5, label=lab)
axs[0].scatter([a_cod], [S_cod], color=NARANJA, s=50, zorder=5, label="(a*, S mín)")
axs[0].set_xlabel("a (°C/km)"); axs[0].set_ylabel("S")
axs[0].set_title("Trazas con b fijo:  S(a, b₀)")
axs[0].set_ylim(0, 400); axs[0].legend(fontsize=9)
for a_fijo, col in [(a_cod, AZUL), (a_cod - 1, VERDE), (a_cod + 1, VIOLETA)]:
    lab = "a = a* = %.2f" % a_fijo if a_fijo == a_cod else "a = %.2f" % a_fijo
    axs[1].plot(bb, S(a_fijo, bb), color=col, lw=2.2 if a_fijo == a_cod else 1.5, label=lab)
axs[1].scatter([b_cod], [S_cod], color=NARANJA, s=50, zorder=5, label="(b*, S mín)")
axs[1].set_xlabel("b (°C)"); axs[1].set_ylabel("S")
axs[1].set_title("Trazas con a fijo:  S(a₀, b)")
axs[1].set_ylim(0, 400); axs[1].legend(fontsize=9)
guardar(fig, "fig04_trazas.png")

# --- Fig. 5: secciones transversales sobre la superficie --------------------
fig = plt.figure(figsize=(9, 6.5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(AA, BB, ZZ, cmap=cm.Blues_r, alpha=0.35, linewidth=0, rstride=5, cstride=5)
ax.plot(aa, np.full_like(aa, b_cod), S(aa, b_cod), color=AZUL, lw=3, label="corte b = b*")
ax.plot(np.full_like(bb, a_cod), bb, S(a_cod, bb), color=VERDE, lw=3, label="corte a = a*")
ax.scatter([a_cod], [b_cod], [S_cod], color=NARANJA, s=70, depthshade=False)
ax.set_xlabel("a (°C/km)"); ax.set_ylabel("b (°C)"); ax.set_zlabel("S(a, b)")
ax.set_title("Secciones transversales: los dos cortes pasan por el mínimo")
ax.legend(loc="upper left")
ax.view_init(elev=24, azim=-50)
guardar(fig, "fig05_secciones_3d.png")

# --- Descenso por gradiente (solo para ilustrar el significado de ∇S) -------
lam_max = autovalores.max()
paso = 1.0 / lam_max
p = np.array([-3.0, 25.0])
camino = [p.copy()]
for _ in range(400):
    p = p - paso * gradiente(*p)
    camino.append(p.copy())
camino = np.array(camino)

# --- Fig. 6: curvas de nivel ------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6))
niveles = [10, 12, 15, 20, 30, 50, 80, 120, 180, 260, 360]
cs = ax.contour(AA, BB, ZZ, levels=niveles, cmap=cm.Blues_r, linewidths=1.4)
ax.clabel(cs, fmt="%g", fontsize=8)
ax.plot(camino[:, 0], camino[:, 1], color=VIOLETA, lw=1.4, marker="o", ms=2.5,
        markevery=8, label="Camino de descenso por gradiente (−∇S)")
ax.scatter([a_cod], [b_cod], color=NARANJA, s=90, zorder=5, edgecolor="white",
           linewidth=1.5, label=f"Mínimo (a*, b*) = ({a_cod:.3f}, {b_cod:.3f})")
ax.set_xlabel("a (pendiente, °C/km)"); ax.set_ylabel("b (intercepto, °C)")
ax.set_title("Curvas de nivel S(a, b) = k: elipses alrededor del mínimo")
ax.legend(loc="upper right", fontsize=9)
guardar(fig, "fig06_curvas_nivel.png")

# --- Fig. 8: recta ajustada con residuos dibujados --------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xs, a_cod * xs + b_cod, color=AZUL, lw=2.5, label="Recta ajustada")
for xi, yi, ypi in zip(x, y, y_pred):
    ax.plot([xi, xi], [yi, ypi], color=NARANJA, lw=1.3)
ax.scatter(x, y, s=55, color=TINTA, edgecolor="white", linewidth=1.5, zorder=4, label="Datos")
idx_max = int(np.argmax(np.abs(e)))
for i in np.argsort(-np.abs(e))[:3]:
    ax.annotate(f"{ciudades[i]} ({e[i]:+.2f} °C)", (x[i], y[i]), xytext=(8, 6),
                textcoords="offset points", fontsize=9, color=TINTA)
ax.set_xlabel("Altitud, x (km)"); ax.set_ylabel("Temperatura, y (°C)")
ax.set_title("Recta ajustada y residuos (segmentos naranjas)")
ax.legend()
guardar(fig, "fig08_recta_y_residuos.png")

# --- Fig. 9: residuos vs x --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(0, color=TINTA2, lw=1)
ax.axhspan(-RMSE, RMSE, color=AZUL, alpha=0.08, label=f"± RMSE = ±{RMSE:.2f} °C")
ax.vlines(x, 0, e, color=GRIS, lw=1)
ax.scatter(x, e, s=55, color=AZUL, edgecolor="white", linewidth=1.5, zorder=3)
desp_r = {"Ibagué": (-40, 6), "Cali": (4, 6), "Bogotá": (-42, -6), "Tunja": (-6, -14), "Pasto": (5, 4)}
for c, xi, ei in zip(ciudades, x, e):
    ax.annotate(c, (xi, ei), xytext=desp_r.get(c, (5, 4 if ei >= 0 else -12)),
                textcoords="offset points", fontsize=8, color=TINTA2)
ax.set_xlabel("Altitud, x (km)"); ax.set_ylabel("Residuo e (°C)")
ax.set_title("Residuos en función de x")
ax.legend(loc="upper right")
guardar(fig, "fig09_residuos_vs_x.png")

# --- Fig. 10: histograma de residuos -----------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.hist(e, bins=np.arange(-1.5, 3.0, 0.5), color=AZUL, edgecolor="white", linewidth=2)
ax.axvline(0, color=TINTA2, lw=1)
ax.set_xlabel("Residuo e (°C)"); ax.set_ylabel("Número de ciudades")
ax.set_title("Histograma de los residuos")
ax.grid(axis="x", visible=False)
guardar(fig, "fig10_histograma_residuos.png")


# ============================================================================
# 7. ANÁLISIS ADICIONALES
# ============================================================================
print("\n" + "=" * 70)
print("7. ANÁLISIS ADICIONALES")
print("=" * 70)

# --- Opción B: influencia de cada dato (quitar uno y recalcular) ------------
infl = []
for i in range(n):
    m = np.arange(n) != i
    ai, bi = ajuste_propio(x[m], y[m])
    infl.append((ciudades[i], ai, bi, ai - a_cod, bi - b_cod))
infl = pd.DataFrame(infl, columns=["sin", "a", "b", "Δa", "Δb"])
print("Opción B — quitando una ciudad a la vez:")
with pd.option_context("display.float_format", "{:.3f}".format):
    print(infl.to_string(index=False))
i_med = ciudades.index("Medellín")
m = np.arange(n) != i_med
a_sm, b_sm = ajuste_propio(x[m], y[m])
e_sm = y[m] - (a_sm * x[m] + b_sm)
RMSE_sm = float(np.sqrt(np.mean(e_sm ** 2)))
R2_sm = 1 - np.sum(e_sm ** 2) / np.sum((y[m] - y[m].mean()) ** 2)
print(f"Sin Medellín: a = {a_sm:.4f}, b = {b_sm:.4f}, RMSE = {RMSE_sm:.4f}, R² = {R2_sm:.4f}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
orden = np.argsort(infl["Δa"].to_numpy())
colores = [NARANJA if infl["sin"][k] in ("Medellín", "Santa Marta") else AZUL for k in orden]
axs[0].barh(infl["sin"].to_numpy()[orden], infl["Δa"].to_numpy()[orden], color=colores, height=0.7)
axs[0].axvline(0, color=TINTA2, lw=1)
axs[0].set_xlabel("Cambio en la pendiente Δa (°C/km)")
axs[0].set_title("Opción B: ¿cuánto cambia a si quitamos cada ciudad?")
axs[0].grid(axis="y", visible=False)
axs[1].scatter(x, y, s=45, color=TINTA, edgecolor="white", zorder=4)
axs[1].scatter([x[i_med]], [y[i_med]], s=110, facecolor="none", edgecolor=NARANJA, lw=2, zorder=5)
axs[1].annotate("Medellín", (x[i_med], y[i_med]), xytext=(8, 6), textcoords="offset points", color=NARANJA)
axs[1].plot(xs, a_cod * xs + b_cod, color=AZUL, lw=2.2, label=f"Con las 13: a = {a_cod:.3f}")
axs[1].plot(xs, a_sm * xs + b_sm, color=NARANJA, lw=1.8, ls="--", label=f"Sin Medellín: a = {a_sm:.3f}")
axs[1].set_xlabel("Altitud, x (km)"); axs[1].set_ylabel("Temperatura (°C)")
axs[1].set_title("Recta con y sin Medellín"); axs[1].legend(fontsize=9)
guardar(fig, "fig11_opcionB_influencia.png")

# --- Opción C: comparación con un modelo cuadrático -------------------------
c2, c1, c0 = np.polyfit(x, y, 2)
yq = c2 * x ** 2 + c1 * x + c0
Sq = float(np.sum((y - yq) ** 2))
RMSEq = float(np.sqrt(Sq / n))
R2q = 1 - Sq / SST
# R² ajustado: penaliza el parámetro extra (p = número de variables explicativas)
R2adj_lin = 1 - (1 - R2) * (n - 1) / (n - 2)
R2adj_q = 1 - (1 - R2q) * (n - 1) / (n - 3)
# Prueba F para saber si el término x² mejora de forma significativa
F = ((S_cod - Sq) / 1) / (Sq / (n - 3))
pF = 1 - stats.f.cdf(F, 1, n - 3)
print(f"\nOpción C — cuadrático: y = {c2:.4f}x² + {c1:.4f}x + {c0:.4f}")
print(f"  S lineal = {S_cod:.4f}  S cuadrático = {Sq:.4f}")
print(f"  RMSE lin = {RMSE:.4f}  RMSE cuad = {RMSEq:.4f}")
print(f"  R² lin = {R2:.4f}  R² cuad = {R2q:.4f}  | R² ajust. lin = {R2adj_lin:.4f}  cuad = {R2adj_q:.4f}")
print(f"  Prueba F = {F:.3f}, p = {pF:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, s=50, color=TINTA, edgecolor="white", zorder=4, label="Datos")
ax.plot(xs, a_cod * xs + b_cod, color=AZUL, lw=2.4, label=f"Lineal (RMSE = {RMSE:.3f} °C)")
ax.plot(xs, c2 * xs ** 2 + c1 * xs + c0, color=NARANJA, lw=2, ls="--",
        label=f"Cuadrático (RMSE = {RMSEq:.3f} °C)")
ax.set_xlabel("Altitud, x (km)"); ax.set_ylabel("Temperatura (°C)")
ax.set_title("Opción C: modelo lineal vs. modelo cuadrático")
ax.legend()
guardar(fig, "fig12_opcionC_cuadratico.png")

# --- Opción A: sensibilidad a pequeñas modificaciones de los datos ----------
rng = np.random.default_rng(2026)
pert = 0.3                                    # ±0.3 °C, del orden del redondeo/variación
sim = np.array([ajuste_propio(x, y + rng.uniform(-pert, pert, n)) for _ in range(2000)])
print(f"\nOpción A — 2000 perturbaciones uniformes de ±{pert} °C en y:")
print(f"  a: media = {sim[:,0].mean():.4f}, desv. = {sim[:,0].std():.4f}, rango = [{sim[:,0].min():.3f}, {sim[:,0].max():.3f}]")
print(f"  b: media = {sim[:,1].mean():.4f}, desv. = {sim[:,1].std():.4f}, rango = [{sim[:,1].min():.3f}, {sim[:,1].max():.3f}]")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
axs[0].hist(sim[:, 0], bins=35, color=AZUL, edgecolor="white", linewidth=0.8)
axs[0].axvline(a_cod, color=NARANJA, lw=2, label=f"a* original = {a_cod:.3f}")
axs[0].set_xlabel("a (°C/km)"); axs[0].set_ylabel("Frecuencia"); axs[0].legend(fontsize=9)
axs[0].set_title("Opción A: pendiente con datos perturbados ±0,3 °C")
axs[1].hist(sim[:, 1], bins=35, color=VERDE, edgecolor="white", linewidth=0.8)
axs[1].axvline(b_cod, color=NARANJA, lw=2, label=f"b* original = {b_cod:.3f}")
axs[1].set_xlabel("b (°C)"); axs[1].legend(fontsize=9)
axs[1].set_title("Opción A: intercepto con datos perturbados")
for a_ in axs:
    a_.grid(axis="x", visible=False)
guardar(fig, "fig13_opcionA_sensibilidad.png")

# --- Opción D: efecto del número de datos -----------------------------------
tam = list(range(4, n + 1))
resD = []
for k in tam:
    est = np.array([ajuste_propio(*(lambda idx: (x[idx], y[idx]))(rng.choice(n, k, replace=False)))
                    for _ in range(1500)])
    resD.append((k, est[:, 0].mean(), est[:, 0].std(), np.percentile(est[:, 0], 5),
                 np.percentile(est[:, 0], 95)))
resD = pd.DataFrame(resD, columns=["n", "a_media", "a_desv", "p5", "p95"])
print("\nOpción D — subconjuntos aleatorios de k ciudades (1500 por tamaño):")
with pd.option_context("display.float_format", "{:.3f}".format):
    print(resD.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.fill_between(resD["n"], resD["p5"], resD["p95"], color=AZUL, alpha=0.15,
                label="90 % de los subconjuntos")
ax.plot(resD["n"], resD["a_media"], color=AZUL, lw=2, marker="o", label="Pendiente promedio")
ax.axhline(a_cod, color=NARANJA, lw=1.5, ls="--", label=f"a* con los 13 datos = {a_cod:.3f}")
ax.set_xlabel("Número de ciudades usadas en el ajuste")
ax.set_ylabel("Pendiente a (°C/km)")
ax.set_title("Opción D: con más datos, la pendiente se estabiliza")
ax.legend(fontsize=9)
guardar(fig, "fig14_opcionD_numero_datos.png")


# ============================================================================
# 8. GUARDAR RESULTADOS (para el informe)
# ============================================================================
resultados = dict(
    n=n, Sx=Sx, Sy=Sy, Sxx=Sxx, Sxy=Sxy, Syy=Syy, D=n * Sxx - Sx ** 2,
    a=a_cod, b=b_cod, S_min=S_cod, a_np=a_np, b_np=b_np, a_sp=a_sp, b_sp=b_sp,
    a_ls=float(a_ls), b_ls=float(b_ls), dif_cod_lib=dif_cod_lib,
    H=H.tolist(), H_num=H_num.tolist(), detH=detH, autovalores=autovalores.tolist(),
    RMSE=RMSE, R2=R2, r=r_pearson, SST=SST, suma_residuos=float(e.sum()),
    residuos=dict(zip(ciudades, e.round(4).tolist())),
    y_pred=dict(zip(ciudades, y_pred.round(4).tolist())),
    sin_medellin=dict(a=a_sm, b=b_sm, RMSE=RMSE_sm, R2=float(R2_sm)),
    influencia=infl.round(4).to_dict(orient="records"),
    cuadratico=dict(c2=c2, c1=c1, c0=c0, S=Sq, RMSE=RMSEq, R2=R2q,
                    R2adj_lin=R2adj_lin, R2adj_q=R2adj_q, F=F, p=pF),
    sensibilidad=dict(a_std=float(sim[:, 0].std()), b_std=float(sim[:, 1].std()),
                      a_min=float(sim[:, 0].min()), a_max=float(sim[:, 0].max())),
    num_datos=resD.round(4).to_dict(orient="records"),
    descenso_gradiente=dict(paso=paso, iteraciones=len(camino) - 1,
                            final=camino[-1].tolist()),
)
with open(os.path.join(CARPETA, "resultados.json"), "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2, default=float)
# Fig. 7 (diagrama de flujo) se genera con Graphviz:  dot -Tpng diagrama_flujo.dot -o figuras/fig07_diagrama_flujo.png
print("\nListo. Figuras en la carpeta 'figuras/' y resultados en 'resultados.json'.")
