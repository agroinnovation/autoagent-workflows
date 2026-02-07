import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                # head‑less mode for script execution
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# Configuration
# -------------------------------------------------
CSV_PATH = "climate.csv"               # input file name
OUTPUT_DIR = "plots"                   # where PNGs will be saved
HOURLY_FREQ = "H"                      # hourly aggregation
WEEKLY_FREQ = "W-MON"                  # weekly aggregation – week starts on Monday
OUTLIER_STD_THRESHOLD = 4              # drop rows > threshold σ from mean

# -------------------------------------------------
# 1️⃣ Load raw CSV and parse timestamps
# -------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        parse_dates=["created"],
        date_parser=lambda x: pd.to_datetime(
            x, format="%Y-%m-%d %H:%M:%S", errors="coerce"
        ),
    )
    # Remove rows where timestamp could not be parsed (NaT → 1970 fallback)
    df = df.dropna(subset=["created"])

    # Set datetime as index and sort chronologically
    df = df.set_index("created").sort_index()

    # Optional: keep only data from 2023‑01‑01 onward (adjust if needed)
    df = df.loc["2023-01-01":]

    return df

# -------------------------------------------------
# 2️⃣ Drop extreme sensor outliers
# -------------------------------------------------
def drop_outliers(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    numeric_cols = ["temperature", "pressure", "humidity"]
    z = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    mask = (z < threshold).all(axis=1)
    cleaned = df[mask]

    removed = len(df) - len(cleaned)
    if removed:
        print(f"Removed {removed} outlier row(s) (> {threshold}σ).")
    else:
        print("No extreme outliers detected.")
    return cleaned

# -------------------------------------------------
# 3️⃣ Plot helper (shared by hourly & weekly)
# -------------------------------------------------
def save_plot(
    df_resampled: pd.DataFrame,
    y_col: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_resampled, x=df_resampled.index, y=y_col, label=y_col.capitalize())

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.xlim(df_resampled.index.min(), df_resampled.index.max())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

# -------------------------------------------------
# 4️⃣ Main workflow
# -------------------------------------------------
def main():
    # ---- Load & clean -------------------------------------------------
    df = load_data(CSV_PATH)
    df = drop_outliers(df, OUTLIER_STD_THRESHOLD)

    print(f"Data range after cleaning: {df.index.min()} → {df.index.max()}")
    print(f"Total valid rows: {len(df)}")

    # ---- Hourly aggregation --------------------------------------------
    df_hourly = df.resample(HOURLY_FREQ).mean()

    # Hourly temperature
    save_plot(
        df_resampled=df_hourly,
        y_col="temperature",
        title="Hourly Temperature",
        ylabel="Temperature (°C)",
        filename="hourly_temperature.png",
    )
    # Hourly pressure
    save_plot(
        df_resampled=df_hourly,
        y_col="pressure",
        title="Hourly Pressure",
        ylabel="Pressure (hPa)",
        filename="hourly_pressure.png",
    )
    # Hourly humidity
    save_plot(
        df_resampled=df_hourly,
        y_col="humidity",
        title="Hourly Humidity",
        ylabel="Humidity (%)",
        filename="hourly_humidity.png",
    )

    # ---- Weekly aggregation (less noisy) --------------------------------
    df_weekly = df.resample(WEEKLY_FREQ).mean()

    # Weekly temperature
    save_plot(
        df_resampled=df_weekly,
        y_col="temperature",
        title="Weekly Temperature (Mean)",
        ylabel="Temperature (°C)",
        filename="weekly_temperature.png",
    )
    # Weekly pressure
    save_plot(
        df_resampled=df_weekly,
        y_col="pressure",
        title="Weekly Pressure (Mean)",
        ylabel="Pressure (hPa)",
        filename="weekly_pressure.png",
    )
    # Weekly humidity
    save_plot(
        df_resampled=df_weekly,
        y_col="humidity",
        title="Weekly Humidity (Mean)",
        ylabel="Humidity (%)",
        filename="weekly_humidity.png",
    )

    print(f"All plots saved to ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    main()