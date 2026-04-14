import numpy as np
import pandas as pd

# ===============================================
# PARAMETRY
# ===============================================
window = 0.005          # szerokość okna czasowego (sekundy)
a = 15.41256
b = -0.01433

# ===============================================
# ŚCIEŻKI DO PLIKÓW
# ===============================================

files = [
    "tenso1.txt",
    "tenso2.txt",
    "tenso3.txt"
]

# ===============================================
# WCZYTANIE DANYCH
# ===============================================

dfs = []
for f in files:
    df = pd.read_csv(f, sep=None, engine='python', header=None)
    df.columns = ["t", "value"]
    dfs.append(df)

# ===============================================
# TWORZENIE WSPÓLNEJ OSI CZASU
# ===============================================

t_min = max(df["t"].min() for df in dfs)
t_max = min(df["t"].max() for df in dfs)

# siatka czasu co 1 ms (możesz zmienić)
t_common = np.arange(t_min, t_max, 0.001)

# interpolacja danych na wspólną siatkę
interp_values = []
for df in dfs:
    interp = np.interp(t_common, df["t"].values, df["value"].values)
    interp_values.append(interp)

interp_values = np.array(interp_values)    # shape: (3, N)

# ===============================================
# OBLICZANIE ŚREDNICH W OKNIE CZASOWYM
# ===============================================

window_means = []
half_w = window / 2

for t in t_common:
    idx = np.where((t_common >= t - half_w) & (t_common <= t + half_w))[0]

    # średnia z każdego tensometru
    means_per_file = interp_values[:, idx].mean(axis=1)

    # suma średnich
    total_mean = means_per_file.sum()

    # liniowa transformacja
    y = a * total_mean + b

    window_means.append([t, total_mean, y])

# DataFrame
result = pd.DataFrame(window_means, columns=["t", "sum_mean", "linear_output"])

# ===============================================
# ZAPIS DO TXT (BEZ NAGŁÓWKÓW, FORMAT 1:1)
# ===============================================

output_file = r"tenso_processed.txt"

np.savetxt(
    output_file,
    result[["t", "linear_output"]].values,
    fmt="%.10f",         # precyzja — można zmienić
    delimiter="\t"       # TAB jak w tensometrach
)

print(f"Zapisano plik: {output_file}")
