import pandas as pd
import matplotlib
matplotlib.use("Agg")                # head‑less mode for script execution
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv("power_hourly.csv")

# Ensure the timestamp column exists and set it as the index
df["created"] = pd.to_datetime(df["created"])
df = df.set_index("created").sort_index()

# -------------------------------------------------
# Define seasons
# -------------------------------------------------
df["month"] = df.index.month
df["season"] = df["month"].apply(
    lambda m: "summer" if m in (6, 7, 8) else ("spring" if m in (3, 4, 5) else "other")
)

# -------------------------------------------------
# Helper: compute hourly average watts for a given subset
# -------------------------------------------------
def hourly_avg(df_subset):
    return (
        df_subset
        .assign(hour=lambda x: x.index.hour)
        .groupby("hour")["watts"]
        .mean()
    )

# -------------------------------------------------
# Compute hourly averages for summer and spring
# -------------------------------------------------
summer_hourly = hourly_avg(df[df["season"] == "summer"])
spring_hourly = hourly_avg(df[df["season"] == "spring"])

# -------------------------------------------------
# Plot summer vs. spring on the same chart (hour‑by‑hour)
# -------------------------------------------------
plt.figure(figsize=(10, 5))

plt.plot(summer_hourly.index, summer_hourly.values,
         label="Summer (Jun‑Aug)", color="orange", marker='o')
plt.plot(spring_hourly.index, spring_hourly.values,
         label="Spring (Mar‑May)", color="green", marker='o')

plt.fill_between(
    summer_hourly.index,
    spring_hourly,
    summer_hourly,
    where=summer_hourly >= spring_hourly,
    interpolate=True,
    color="red",
    alpha=0.3,
    label="Excess (Summer – Spring)"
)

plt.title("Hourly Power Usage: Summer vs. Spring")
plt.xlabel("Hour of Day (0‑23)")
plt.ylabel("Average Watts")
plt.xticks(range(0, 24))
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/summer_vs_spring.png")
plt.close()

# -------------------------------------------------
# 📊  Monthly summary table
# -------------------------------------------------
# Average wattage per month (overall)
monthly_avg = (
    df
    .assign(hour=lambda x: x.index.hour)
    .groupby(["month"])["watts"]
    .mean()
    .reset_index()
    .rename(columns={"watts": "avg_watts"})
)

# Map summer months to their “partner” spring month
partner_map = {6: 3, 7: 4, 8: 5}

# Pull the partner month average and compute excess
monthly_avg["partner_month"] = monthly_avg["month"].map(partner_map)
monthly_avg["partner_avg_watts"] = monthly_avg.apply(
    lambda row: monthly_avg.loc[monthly_avg["month"] == row["partner_month"], "avg_watts"].values[0]
    if pd.notnull(row["partner_month"]) else pd.NA,
    axis=1
)

monthly_avg["excess_vs_partner"] = (
    monthly_avg["avg_watts"] - monthly_avg["partner_avg_watts"]
)

# Re‑order columns for readability
summary = monthly_avg[[
    "month",
    "avg_watts",
    "partner_month",
    "partner_avg_watts",
    "excess_vs_partner"
]]

# Save as CSV
summary_path = "plots/monthly_summary.csv"
summary.to_csv(summary_path, index=False)

# Also print a pretty version to the console
print("\n=== Monthly Power‑Usage Summary ===")
print(summary.to_string(index=False, float_format="{:.2f}".format))
print(f"\nSummary saved to: {summary_path}")