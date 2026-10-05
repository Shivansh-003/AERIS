"""AERIS Exploratory Data Analysis & Quality Audit Engine."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure non-interactive backend for headless execution
matplotlib.use("Agg")

# Set standard publication styling
plt.rcParams.update(
    {
        "figure.autolayout": True,
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.family": "sans-serif",
    }
)

POLLUTANT_COLUMNS: List[str] = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene",
]
NUMERIC_COLUMNS: List[str] = POLLUTANT_COLUMNS + ["AQI"]


def compute_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level dataset structural statistics."""
    unique_cities = sorted([str(c) for c in df["City"].dropna().unique().tolist()])
    valid_dates = df["Date"].dropna()

    obs_per_city = {
        str(k): int(v) for k, v in df["City"].value_counts().sort_index().items()
    }

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "data_types": {col: str(df[col].dtype) for col in df.columns},
        "city_count": len(unique_cities),
        "cities": unique_cities,
        "date_range": {
            "min_date": valid_dates.min().strftime("%Y-%m-%d")
            if not valid_dates.empty
            else None,
            "max_date": valid_dates.max().strftime("%Y-%m-%d")
            if not valid_dates.empty
            else None,
            "total_days_span": int((valid_dates.max() - valid_dates.min()).days) + 1
            if not valid_dates.empty
            else 0,
        },
        "observations_per_city": obs_per_city,
    }


def compute_missing_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute missing value diagnostics across columns and cities."""
    total_rows = len(df)
    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    missing_pcts = {
        col: round((cnt / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
        for col, cnt in missing_counts.items()
    }

    city_missingness: Dict[str, Dict[str, Any]] = {}
    for city, grp in df.groupby("City"):
        c_rows = len(grp)
        c_missing = {
            col: round((int(grp[col].isna().sum()) / c_rows) * 100.0, 2)
            for col in df.columns
            if col not in ["City", "Date"]
        }
        city_missingness[str(city)] = {
            "row_count": c_rows,
            "missing_percentages": c_missing,
        }

    return {
        "missing_counts": missing_counts,
        "missing_percentages": missing_pcts,
        "aqi_missing_count": missing_counts.get("AQI", 0),
        "aqi_missing_percentage": missing_pcts.get("AQI", 0.0),
        "pollutant_missing_percentages": {
            pol: missing_pcts.get(pol, 0.0) for pol in POLLUTANT_COLUMNS
        },
        "city_missingness": city_missingness,
    }


def compute_duplicate_analysis(df: pd.DataFrame) -> Dict[str, int]:
    """Compute duplicate row counts."""
    return {
        "duplicate_rows_count": int(df.duplicated().sum()),
        "duplicate_city_date_count": int(df.duplicated(subset=["City", "Date"]).sum()),
    }


def compute_outlier_analysis(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute IQR-based outlier statistics for numeric variables."""
    outlier_stats: Dict[str, Dict[str, Any]] = {}
    for col in NUMERIC_COLUMNS:
        series = df[col].dropna()
        if series.empty:
            continue
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round((outlier_count / len(series)) * 100.0, 2)

        outlier_stats[col] = {
            "valid_count": len(series),
            "q25": round(q25, 4),
            "q75": round(q75, 4),
            "iqr": round(iqr, 4),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_pct,
        }
    return outlier_stats


def compute_city_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute per-city air quality summary statistics."""
    city_stats: Dict[str, Dict[str, Any]] = {}
    for city, grp in df.groupby("City"):
        city_str = str(city)
        aqi_s = grp["AQI"].dropna()
        c_rows = len(grp)
        c_aqi_cnt = len(aqi_s)

        if not aqi_s.empty:
            city_stats[city_str] = {
                "observation_count": c_rows,
                "aqi_valid_count": c_aqi_cnt,
                "aqi_missing_count": c_rows - c_aqi_cnt,
                "aqi_mean": round(float(aqi_s.mean()), 2),
                "aqi_median": round(float(aqi_s.median()), 2),
                "aqi_std": round(float(aqi_s.std()), 2)
                if not np.isnan(aqi_s.std())
                else 0.0,
                "aqi_min": round(float(aqi_s.min()), 2),
                "aqi_max": round(float(aqi_s.max()), 2),
            }
        else:
            city_stats[city_str] = {
                "observation_count": c_rows,
                "aqi_valid_count": 0,
                "aqi_missing_count": c_rows,
                "aqi_mean": None,
                "aqi_median": None,
                "aqi_std": None,
                "aqi_min": None,
                "aqi_max": None,
            }
    return city_stats


def compute_pollutant_summaries(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute detailed summary statistics for each pollutant."""
    summaries: Dict[str, Dict[str, Any]] = {}
    for col in POLLUTANT_COLUMNS:
        s = df[col].dropna()
        if s.empty:
            continue
        desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        summaries[col] = {
            "valid_count": int(desc["count"]),
            "mean": round(float(desc["mean"]), 4),
            "std": round(float(desc["std"]), 4) if not np.isnan(desc["std"]) else 0.0,
            "min": round(float(desc["min"]), 4),
            "p5": round(float(desc["5%"]), 4),
            "p25": round(float(desc["25%"]), 4),
            "median": round(float(desc["50%"]), 4),
            "p75": round(float(desc["75%"]), 4),
            "p95": round(float(desc["95%"]), 4),
            "p99": round(float(desc["99%"]), 4),
            "max": round(float(desc["max"]), 4),
            "skewness": round(float(s.skew()), 4),
        }
    return summaries


def compute_aqi_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute AQI distribution statistics and bucket breakdown."""
    aqi_s = df["AQI"].dropna()
    desc = aqi_s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

    bucket_counts: Dict[str, int] = {}
    if "AQI_Bucket" in df.columns:
        for b, count in df["AQI_Bucket"].value_counts(dropna=False).items():
            key = str(b) if pd.notna(b) else "Missing"
            bucket_counts[key] = int(count)

    return {
        "valid_count": int(desc["count"]),
        "missing_count": int(df["AQI"].isna().sum()),
        "mean": round(float(desc["mean"]), 2),
        "median": round(float(desc["50%"]), 2),
        "std": round(float(desc["std"]), 2),
        "min": round(float(desc["min"]), 2),
        "p5": round(float(desc["5%"]), 2),
        "p25": round(float(desc["25%"]), 2),
        "p50": round(float(desc["50%"]), 2),
        "p75": round(float(desc["75%"]), 2),
        "p90": round(float(desc["90%"]), 2),
        "p95": round(float(desc["95%"]), 2),
        "p99": round(float(desc["99%"]), 2),
        "max": round(float(desc["max"]), 2),
        "iqr": round(float(desc["75%"] - desc["25%"]), 2),
        "skewness": round(float(aqi_s.skew()), 4),
        "bucket_distribution": bucket_counts,
    }


def compute_temporal_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute yearly, monthly, and seasonal AQI aggregations."""
    df_temp = df.copy()
    df_temp["Year"] = df_temp["Date"].dt.year
    df_temp["Month"] = df_temp["Date"].dt.month

    yearly_stats: Dict[str, Dict[str, Any]] = {}
    for yr, grp in df_temp.groupby("Year"):
        s = grp["AQI"].dropna()
        yearly_stats[str(yr)] = {
            "count": len(s),
            "mean": round(float(s.mean()), 2) if not s.empty else None,
            "median": round(float(s.median()), 2) if not s.empty else None,
            "std": round(float(s.std()), 2) if not s.empty else None,
            "min": round(float(s.min()), 2) if not s.empty else None,
            "max": round(float(s.max()), 2) if not s.empty else None,
        }

    monthly_stats: Dict[str, Dict[str, Any]] = {}
    for mo, grp in df_temp.groupby("Month"):
        s = grp["AQI"].dropna()
        monthly_stats[str(mo)] = {
            "count": len(s),
            "mean": round(float(s.mean()), 2) if not s.empty else None,
            "median": round(float(s.median()), 2) if not s.empty else None,
            "std": round(float(s.std()), 2) if not s.empty else None,
        }

    return {
        "yearly_aqi_stats": yearly_stats,
        "monthly_aqi_stats": monthly_stats,
    }


def compute_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute Pearson and Spearman correlation matrices and pairings with AQI."""
    numeric_df = df[NUMERIC_COLUMNS]

    pearson_matrix = numeric_df.corr(method="pearson").round(4).to_dict()
    spearman_matrix = numeric_df.corr(method="spearman").round(4).to_dict()

    aqi_correlations: Dict[str, Dict[str, Any]] = {}
    for pol in POLLUTANT_COLUMNS:
        pair_df = df[[pol, "AQI"]].dropna()
        if not pair_df.empty:
            p_r = float(pair_df[pol].corr(pair_df["AQI"], method="pearson"))
            s_rho = float(pair_df[pol].corr(pair_df["AQI"], method="spearman"))
            aqi_correlations[pol] = {
                "pearson_r": round(p_r, 4),
                "spearman_rho": round(s_rho, 4),
                "pairwise_observations": len(pair_df),
            }
        else:
            aqi_correlations[pol] = {
                "pearson_r": None,
                "spearman_rho": None,
                "pairwise_observations": 0,
            }

    return {
        "aqi_correlations": aqi_correlations,
        "pearson_matrix": pearson_matrix,
        "spearman_matrix": spearman_matrix,
    }


def generate_eda_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Compile the entire EDA report into a single dictionary."""
    overview = compute_dataset_overview(df)
    missing = compute_missing_analysis(df)
    duplicates = compute_duplicate_analysis(df)
    outliers = compute_outlier_analysis(df)
    city_stats = compute_city_statistics(df)
    pollutant_stats = compute_pollutant_summaries(df)
    aqi_stats = compute_aqi_summary(df)
    temporal_stats = compute_temporal_statistics(df)
    corr_stats = compute_correlations(df)

    return {
        "report_metadata": {
            "title": "AERIS Exploratory Data Analysis & Quality Report",
            "version": "1.0.0",
            "source_dataset": "data/processed/cleaned_city_day.parquet",
        },
        "dataset_overview": overview,
        "missing_value_analysis": missing,
        "duplicate_analysis": duplicates,
        "outlier_analysis": outliers,
        "city_level_statistics": city_stats,
        "pollutant_summary_statistics": pollutant_stats,
        "aqi_distribution_summary": aqi_stats,
        "temporal_statistics": temporal_stats,
        "correlation_analysis": corr_stats,
    }


def save_eda_report(report: Dict[str, Any], output_path: Path | str) -> Path:
    """Save EDA report as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def plot_all_figures(df: pd.DataFrame, output_dir: Path | str) -> List[Path]:
    """Generate and save all 9 required EDA visualizations."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    # 1. AQI Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    aqi_clean = df["AQI"].dropna()
    sns.histplot(
        aqi_clean,
        kde=True,
        color="#2563eb",
        bins=60,
        ax=ax,
        edgecolor="none",
        alpha=0.6,
    )
    ax.axvline(
        aqi_clean.mean(),
        color="#dc2626",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean ({aqi_clean.mean():.1f})",
    )
    ax.axvline(
        aqi_clean.median(),
        color="#16a34a",
        linestyle="-.",
        linewidth=1.8,
        label=f"Median ({aqi_clean.median():.1f})",
    )
    ax.set_title("Distribution of Air Quality Index (AQI)")
    ax.set_xlabel("AQI")
    ax.set_ylabel("Frequency")
    ax.legend()
    p1 = out_dir / "aqi_distribution.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p1)

    # 2. Pollutant Distributions (Multi-panel histograms)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    for idx, pol in enumerate(POLLUTANT_COLUMNS):
        ax = axes[idx]
        s = df[pol].dropna()
        sns.histplot(
            s,
            kde=True,
            ax=ax,
            color="#0284c7",
            bins=40,
            edgecolor="none",
            alpha=0.5,
        )
        ax.set_title(f"{pol} (N={len(s):,})")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
    plt.suptitle(
        "Empirical Distributions of 12 Monitored Pollutants",
        fontsize=14,
        y=1.02,
    )
    p2 = out_dir / "pollutant_distributions.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p2)

    # 3. Correlation Matrix Heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df[NUMERIC_COLUMNS].corr(method="pearson")
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-0.2,
        vmax=1.0,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix (Pollutants & AQI)")
    p3 = out_dir / "correlation_matrix.png"
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p3)

    # 4. AQI Over Time (National Monthly Mean & Selected Major Cities)
    fig, ax = plt.subplots(figsize=(12, 6))
    df_ts = df.dropna(subset=["AQI"]).copy()
    monthly_national = df_ts.groupby(pd.Grouper(key="Date", freq="ME"))["AQI"].mean()
    ax.plot(
        monthly_national.index,
        monthly_national.values,
        color="#1e293b",
        linewidth=2.5,
        label="National Monthly Average",
    )
    for city, color in [
        ("Delhi", "#ef4444"),
        ("Ahmedabad", "#f59e0b"),
        ("Bengaluru", "#10b981"),
    ]:
        city_sub = df_ts[df_ts["City"] == city]
        city_m = city_sub.groupby(pd.Grouper(key="Date", freq="ME"))["AQI"].mean()
        ax.plot(
            city_m.index,
            city_m.values,
            linestyle="--",
            alpha=0.7,
            label=f"{city} Monthly Avg",
            color=color,
        )
    ax.set_title("Historical AQI Trajectory Across Time (2015 - 2020)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Monthly AQI")
    ax.legend()
    p4 = out_dir / "aqi_over_time.png"
    plt.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p4)

    # 5. City-wise AQI Comparison (Horizontal Bar Chart sorted by Mean AQI)
    fig, ax = plt.subplots(figsize=(10, 10))
    city_means = df.groupby("City")["AQI"].mean().dropna().sort_values(ascending=True)
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(city_means)))
    ax.barh(
        city_means.index,
        city_means.values,
        color=colors,
        edgecolor="none",
        height=0.65,
    )
    ax.set_title("City-Wise Historical Mean AQI Ranking")
    ax.set_xlabel("Mean AQI")
    ax.set_ylabel("City")
    p5 = out_dir / "city_wise_aqi.png"
    plt.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p5)

    # 6. Missing Values Visualization (Column-wise missingness percentages)
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_pcts = (df.isna().sum() / len(df) * 100.0).sort_values(ascending=False)
    missing_pcts = missing_pcts[missing_pcts > 0]
    sns.barplot(
        x=missing_pcts.values,
        y=missing_pcts.index,
        color="#e11d48",
        ax=ax,
    )
    ax.set_title("Missing Value Proportion by Feature (%)")
    ax.set_xlabel("Missing Percentage (%)")
    for i, v in enumerate(missing_pcts.values):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=8.5)
    ax.set_xlim(0, 70)
    p6 = out_dir / "missing_values.png"
    plt.savefig(p6, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p6)

    # 7. PM2.5 vs AQI Scatter / Regression
    fig, ax = plt.subplots(figsize=(8, 6))
    sub_pm25 = df[["PM2.5", "AQI"]].dropna()
    p_r25 = sub_pm25["PM2.5"].corr(sub_pm25["AQI"], method="pearson")
    s_r25 = sub_pm25["PM2.5"].corr(sub_pm25["AQI"], method="spearman")
    sns.regplot(
        data=sub_pm25,
        x="PM2.5",
        y="AQI",
        ax=ax,
        scatter_kws={"alpha": 0.15, "color": "#2563eb", "s": 12},
        line_kws={"color": "#dc2626", "linewidth": 2},
    )
    ax.set_title(f"PM2.5 vs AQI (Pearson r = {p_r25:.2f}, Spearman ρ = {s_r25:.2f})")
    ax.set_xlabel("PM2.5 (µg/m³)")
    ax.set_ylabel("AQI")
    p7 = out_dir / "pm25_vs_aqi.png"
    plt.savefig(p7, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p7)

    # 8. PM10 vs AQI Scatter / Regression
    fig, ax = plt.subplots(figsize=(8, 6))
    sub_pm10 = df[["PM10", "AQI"]].dropna()
    p_r10 = sub_pm10["PM10"].corr(sub_pm10["AQI"], method="pearson")
    s_r10 = sub_pm10["PM10"].corr(sub_pm10["AQI"], method="spearman")
    sns.regplot(
        data=sub_pm10,
        x="PM10",
        y="AQI",
        ax=ax,
        scatter_kws={"alpha": 0.15, "color": "#0d9488", "s": 12},
        line_kws={"color": "#dc2626", "linewidth": 2},
    )
    ax.set_title(f"PM10 vs AQI (Pearson r = {p_r10:.2f}, Spearman ρ = {s_r10:.2f})")
    ax.set_xlabel("PM10 (µg/m³)")
    ax.set_ylabel("AQI")
    p8 = out_dir / "pm10_vs_aqi.png"
    plt.savefig(p8, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p8)

    # 9. NO2 vs AQI Scatter / Regression
    fig, ax = plt.subplots(figsize=(8, 6))
    sub_no2 = df[["NO2", "AQI"]].dropna()
    p_r_no2 = sub_no2["NO2"].corr(sub_no2["AQI"], method="pearson")
    s_r_no2 = sub_no2["NO2"].corr(sub_no2["AQI"], method="spearman")
    sns.regplot(
        data=sub_no2,
        x="NO2",
        y="AQI",
        ax=ax,
        scatter_kws={"alpha": 0.15, "color": "#9333ea", "s": 12},
        line_kws={"color": "#dc2626", "linewidth": 2},
    )
    ax.set_title(f"NO2 vs AQI (Pearson r = {p_r_no2:.2f}, Spearman ρ = {s_r_no2:.2f})")
    ax.set_xlabel("NO2 (µg/m³)")
    ax.set_ylabel("AQI")
    p9 = out_dir / "no2_vs_aqi.png"
    plt.savefig(p9, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p9)

    return saved_paths


def run_eda_pipeline(
    parquet_path: Path | str = "data/processed/cleaned_city_day.parquet",
    report_output: Path | str = "outputs/eda/eda_report.json",
    figures_output: Path | str = "outputs/eda/figures",
) -> Tuple[Dict[str, Any], List[Path]]:
    """Execute end-to-end EDA analysis, saving report and figures."""
    p_path = Path(parquet_path)
    if not p_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {p_path}")

    df = pd.read_parquet(p_path)
    report = generate_eda_report(df)
    save_eda_report(report, report_output)
    saved_figs = plot_all_figures(df, figures_output)

    return report, saved_figs
