# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# %%

np.random.seed(42)

# %%

# y lambda_t = efecto_anual + efecto_mensual + efecto_diario.

efecto_anual = {2023: 1000, 2024: 1500, 2025: 2000}


efecto_mensual = {
    1: 1000,   # Enero
    2: 1500,   # Febrero
    3: 2000,   # Marzo
    4: 2000,   # Abril
    5: 2500,   # Mayo
    6: 2500,   # Junio
    7: 3000,   # Julio
    8: 2500,   # Agosto
    9: 2500,   # Septiembre
    10: 2000,  # Octubre
    11: 1500,  # Noviembre
    12: 1000,  # Diciembre
}


efecto_diario = {
    6: 1000,  # Domingo
    0: 2000,  # Lunes
    1: 3000,  # Martes
    2: 3500,  # Miércoles
    3: 3000,  # Jueves
    4: 2000,  # Viernes
    5: 1000,  # Sábado
}

# %%

fechas = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")


lambdas = np.array([
    efecto_anual[f.year] + efecto_mensual[f.month] + efecto_diario[f.dayofweek]
    for f in fechas
])


ventas_francisco = np.random.poisson(lambdas)
ventas_miguel = np.random.poisson(lambdas)


df = pd.DataFrame({
    "fecha": fechas,
    "lambda_t": lambdas,
    "ventas_francisco": ventas_francisco,
    "ventas_miguel": ventas_miguel,
})
df["año"] = df["fecha"].dt.year


print(f"Total de días simulados: {len(df)}")
print(f"\nPrimeras filas:")
print(df.head(10).to_string(index=False))
print(f"\nEstadísticas por año (Don Francisco):")
print(df.groupby("año")["ventas_francisco"].describe().to_string())
print(f"\nEstadísticas por año (Don Miguel):")
print(df.groupby("año")["ventas_miguel"].describe().to_string())

# %%
# incisos 2 y3
# funcion empirica y densidad 
años = [2023, 2024, 2025]
almacenes = {
    "Don Francisco": "ventas_francisco",
    "Don Miguel":    "ventas_miguel",
}

# Colores por año
colores_año = {2023: "#378ADD", 2024: "#1D9E75", 2025: "#D85A30"}

for nombre, col in almacenes.items():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{nombre} — Distribucion", fontsize=14, fontweight="bold")

    for j, año in enumerate(años):
        datos = np.sort(df[df["año"] == año][col].values)
        n = len(datos)
        color = colores_año[año]

        
        # datos ordenados con la prob asignada
        ecdf_x = datos
        ecdf_y = np.arange(1, n + 1) / n

        ax_ecdf = axes[0, j]
        ax_ecdf.step(ecdf_x, ecdf_y, where="post", color=color, linewidth=1.8)
        ax_ecdf.set_title(f"{año}  (n={n})", fontsize=11)
        ax_ecdf.set_xlabel("Ventas diarias")
        ax_ecdf.set_ylabel("F̂(x)")
        ax_ecdf.set_ylim(0, 1.05)
        ax_ecdf.grid(True, alpha=0.3)

        
        for q, label in zip([0.25, 0.50, 0.75], ["Q1", "Med", "Q3"]):
            xq = np.quantile(datos, q)
            ax_ecdf.axvline(xq, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            ax_ecdf.text(xq, q + 0.04, label, fontsize=8, ha="center", color="gray")

        
        k = 10  # cantidad de clases
        h = (datos[-1] - datos[0]) / k  # ancho de cada clase

        # la densidad de cada barra es f_i / (n * h), asi el area total da 1
        ax_dens = axes[1, j]
        ax_dens.hist(datos, bins=k, density=True, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax_dens.set_xlabel("Ventas diarias")
        ax_dens.set_ylabel("f̂(x) = fᵢ / (n·h)")
        ax_dens.set_title(f"k={k} clases,  h={h:.0f}", fontsize=10)
        ax_dens.grid(True, alpha=0.3)

        
        counts, edges = np.histogram(datos, bins=k)
        print(f"\n{nombre} {año} — k={k}, h={h:.1f}, n={n}")
        print(f"{'Intervalo':>20}  {'fᵢ':>5}  {'f̂ᵢ = fᵢ/(n·h)':>18}")
        print("-" * 50)
        for i in range(len(counts)):
            densidad = counts[i] / (n * h)
            print(f"[{edges[i]:7.0f}, {edges[i+1]:7.0f})  {counts[i]:>5}  {densidad:.8f}")
        print(f"Verificación área total: {np.sum(counts / (n * h)) * h:.4f}  (debe ser ≈ 1.0)")

    plt.tight_layout()
    plt.show()


# inciso 4
# densidad empirica conjunta
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Densidad empírica conjunta — Don Francisco (X) vs Don Miguel (Y)", fontsize=13, fontweight="bold")

for j, año in enumerate(años):
    df_año = df[df["año"] == año]
    x = df_año["ventas_francisco"].values
    y = df_año["ventas_miguel"].values
    n = len(x)

    # Misma cantidad de clases para ambos ejes
    kx = 10
    ky = 10
    hx = (x.max() - x.min()) / kx
    hy = (y.max() - y.min()) / ky

    # normalizacion
    H, xedges, yedges = np.histogram2d(x, y, bins=[kx, ky])
    densidad_conjunta = H / (n * hx * hy)

    ax = axes[j]
    im = ax.imshow(
        densidad_conjunta.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="Blues"
    )
    plt.colorbar(im, ax=ax, label="f̂(x,y)")
    ax.set_title(f"{año}  (n={n}, kx=ky={kx})", fontsize=10)
    ax.set_xlabel("Ventas Francisco (X)")
    ax.set_ylabel("Ventas Miguel (Y)")

    # tiene que sumar 1
    verificacion = np.sum(densidad_conjunta) * hx * hy
    print(f"\nDensidad conjunta {año}: hx={hx:.1f}, hy={hy:.1f}")
    print(f"Verificación ∑ f̂(x,y)·hx·hy = {verificacion:.4f}  (debe ser ≈ 1.0)")

plt.tight_layout()

plt.show()


