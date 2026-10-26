"""Generate a small, explainable Food Delivery ETA dashboard.

The generated data is for demonstration only. Replace it with a documented,
approved dataset before making real operational decisions.
"""

from __future__ import annotations

import argparse
import csv
import html
import random
import webbrowser
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
DATA_FILE = OUTPUT / "synthetic_delivery_data.csv"
REPORT_FILE = OUTPUT / "delivery_dashboard.html"
TRAFFIC_EFFECT = {"low": 0.0, "medium": 6.0, "high": 15.0}
WEATHER_EFFECT = {"clear": 0.0, "rain": 7.0}


def make_demo_data(rows: int = 600, seed: int = 42) -> list[dict[str, float | str]]:
    """Create reproducible demonstration records with an understandable pattern."""
    random.seed(seed)
    records: list[dict[str, float | str]] = []
    for _ in range(rows):
        distance = round(random.uniform(1, 15), 1)
        items = random.randint(1, 8)
        traffic = random.choices(["low", "medium", "high"], weights=[2, 5, 3])[0]
        weather = random.choices(["clear", "rain"], weights=[8, 2])[0]
        noise = random.gauss(0, 3.2)
        eta = max(
            12,
            11 + 2.25 * distance + 0.85 * items + TRAFFIC_EFFECT[traffic] + WEATHER_EFFECT[weather] + noise,
        )
        records.append(
            {
                "distance_km": distance,
                "items": items,
                "traffic": traffic,
                "weather": weather,
                "actual_eta_min": round(eta, 1),
            }
        )
    return records


def features(distance: float, items: int, traffic: str, weather: str) -> list[float]:
    return [1.0, distance, items, float(traffic == "medium"), float(traffic == "high"), float(weather == "rain")]


def train(records: list[dict[str, float | str]]) -> tuple[np.ndarray, float, list[float]]:
    x = np.array(
        [features(float(r["distance_km"]), int(r["items"]), str(r["traffic"]), str(r["weather"])) for r in records],
        dtype=float,
    )
    y = np.array([float(r["actual_eta_min"]) for r in records], dtype=float)
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    predictions = x @ coefficients
    mae = float(np.mean(np.abs(y - predictions)))
    return coefficients, mae, predictions.tolist()


def predict(coefficients: np.ndarray, distance: float, items: int, traffic: str, weather: str) -> float:
    return float(np.dot(np.array(features(distance, items, traffic, weather)), coefficients))


def svg_bar_chart(averages: dict[str, float]) -> str:
    max_value = max(averages.values())
    bars = []
    for index, (label, value) in enumerate(averages.items()):
        height = round(200 * value / max_value)
        x = 55 + index * 105
        bars.append(
            f'<rect x="{x}" y="{235 - height}" width="64" height="{height}" rx="6" fill="#2563eb" />'
            f'<text x="{x + 32}" y="258" text-anchor="middle">{html.escape(label.title())}</text>'
            f'<text x="{x + 32}" y="{225 - height}" text-anchor="middle">{value:.1f} min</text>'
        )
    return f'<svg viewBox="0 0 400 280" role="img" aria-label="Average ETA by traffic level">{"".join(bars)}</svg>'


def create_report(records: list[dict[str, float | str]], coefficients: np.ndarray, mae: float, predictions: list[float]) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        groups[str(record["traffic"])].append(float(record["actual_eta_min"]))
    averages = {traffic: sum(values) / len(values) for traffic, values in groups.items()}
    sample_rows = []
    for record, prediction in list(zip(records, predictions))[:12]:
        sample_rows.append(
            "<tr>"
            f"<td>{record['distance_km']}</td><td>{record['items']}</td><td>{record['traffic']}</td>"
            f"<td>{record['weather']}</td><td>{record['actual_eta_min']}</td><td>{prediction:.1f}</td>"
            "</tr>"
        )
    names = ["Base time", "Distance (km)", "Items", "Medium traffic", "High traffic", "Rain"]
    coefficient_rows = "".join(f"<li><strong>{name}:</strong> {value:+.2f} minutes</li>" for name, value in zip(names, coefficients))
    typical_eta = sum(float(r["actual_eta_min"]) for r in records) / len(records)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Food Delivery ETA Dashboard</title><style>
body{{font-family:Arial,sans-serif;background:#f8fafc;color:#172033;margin:0;padding:32px;line-height:1.5}}
main{{max-width:1000px;margin:auto}}h1{{margin-bottom:4px}}.muted{{color:#5b6472}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px;margin:24px 0}}
.card,section{{background:white;border-radius:12px;padding:18px;box-shadow:0 1px 4px #cbd5e155}}.metric{{font-size:1.8rem;font-weight:bold;color:#1d4ed8}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}td,th{{padding:8px;border-bottom:1px solid #e5e7eb;text-align:left}}th{{background:#eff6ff}}svg{{max-width:100%;height:auto}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Food Delivery ETA Dashboard</h1><p class="muted">Demonstration data - use real, approved data before operational use.</p>
<div class="cards"><div class="card"><div class="muted">Orders analysed</div><div class="metric">{len(records)}</div></div><div class="card"><div class="muted">Model MAE</div><div class="metric">{mae:.1f} min</div></div><div class="card"><div class="muted">Typical ETA</div><div class="metric">{typical_eta:.1f} min</div></div></div>
<div class="grid"><section><h2>Average ETA by traffic</h2>{svg_bar_chart(averages)}</section><section><h2>What drives the estimate?</h2><p>Positive values add time; the base time is the starting estimate.</p><ul>{coefficient_rows}</ul><p><strong>Operational note:</strong> show customers a range, not a false promise of an exact minute.</p></section></div>
<section><h2>Prediction sample</h2><table><thead><tr><th>Distance</th><th>Items</th><th>Traffic</th><th>Weather</th><th>Actual</th><th>Predicted</th></tr></thead><tbody>{''.join(sample_rows)}</tbody></table></section>
</main></body></html>"""
    REPORT_FILE.write_text(page, encoding="utf-8")


def save_data(records: list[dict[str, float | str]]) -> None:
    with DATA_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Food Delivery ETA dashboard.")
    parser.add_argument("--distance", type=float, help="Distance in kilometres for one ETA prediction.")
    parser.add_argument("--items", type=int, default=2, help="Number of items in the order.")
    parser.add_argument("--traffic", choices=TRAFFIC_EFFECT, default="medium")
    parser.add_argument("--weather", choices=WEATHER_EFFECT, default="clear")
    parser.add_argument("--open", action="store_true", help="Open the generated report in a browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(exist_ok=True)
    records = make_demo_data()
    save_data(records)
    coefficients, mae, predictions = train(records)
    create_report(records, coefficients, mae, predictions)
    print(f"Dashboard created: {REPORT_FILE}")
    print(f"Model MAE: {mae:.1f} minutes")
    if args.distance is not None:
        estimate = predict(coefficients, args.distance, args.items, args.traffic, args.weather)
        print(f"Estimated delivery time: {estimate:.0f}-{estimate + mae:.0f} minutes")
    if args.open:
        webbrowser.open(REPORT_FILE.as_uri())


if __name__ == "__main__":
    main()
