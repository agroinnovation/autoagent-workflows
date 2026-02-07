import pandas as pd, matplotlib, matplotlib.pyplot as plt, seaborn as sns, os
matplotlib.use("Agg")
os.makedirs("plots", exist_ok=True)

# ----- load data -------------------------------------------------
df = pd.read_csv("power_hourly.csv")
df["created"] = pd.to_datetime(df["created"])
df = df.set_index("created").sort_index()

# ----- add helper columns ----------------------------------------
df["hour"]     = df.index.hour
df["weekday"]  = df.index.day_name()
df["month"]    = df.index.month_name()

# ----- 1) Hour‑of‑day ------------------------------------------------
hourly = df.groupby("hour")[["watts"]].mean()
plt.figure(figsize=(10,3))
hourly["watts"].plot(kind="bar",color="steelblue")
plt.title("Average watts by hour of day"); plt.xlabel(""); plt.ylabel("Watts")
plt.tight_layout(); plt.savefig("plots/hourly_watts.png"); plt.close()

# ----- 2) 7‑day rolling --------------------------------------------
df["watts_roll7d"] = df["watts"].rolling(window=168, min_periods=1).mean()
plt.figure(figsize=(12,4))
df[["watts","watts_roll7d"]].plot(alpha=0.5)
plt.title("Watts vs 7‑day rolling average"); plt.ylabel("Watts")
plt.tight_layout(); plt.savefig("plots/rolling_7d.png"); plt.close()

# ----- 3) Monthly -------------------------------------------------
monthly = df.groupby("month")["watts"].mean()
month_order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
monthly = monthly.reindex(month_order)
plt.figure(figsize=(10,3))
monthly.plot(kind="bar",color="darkorange")
plt.title("Average watts by month"); plt.ylabel("Watts")
plt.tight_layout(); plt.savefig("plots/monthly_watts.png"); plt.close()

# ----- 4) Δwatts -------------------------------------------------
df["watts_diff"] = df["watts"].diff()
plt.figure(figsize=(12,3))
df["watts_diff"].plot(color="crimson")
plt.title("Hour‑to‑hour change in watts"); plt.axhline(0,color="gray",linestyle="--"); plt.ylabel("Δ Watts")
plt.tight_layout(); plt.savefig("plots/watts_diff.png"); plt.close()

# ----- 5) Correlation heatmap ------------------------------------
corr = df[["amperage","voltage","watts"]].corr()
plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix")
plt.tight_layout(); plt.savefig("plots/correlation.png"); plt.close()
