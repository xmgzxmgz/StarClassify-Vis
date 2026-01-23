import csv
import random
from pathlib import Path


def clamp(value: float, low: float, high: float) -> float:
    """将数值限制在给定范围内。"""
    return max(low, min(high, value))


def rnd(value: float) -> float:
    """将数值格式化为较短小数。"""
    return round(value, 5)


def gen_row(cls: str, noise: float) -> dict:
    """生成单条样本数据。"""
    if cls == "STAR":
        u = random.gauss(18.2, 0.6 + noise)
        g = random.gauss(17.6, 0.5 + noise)
        r = random.gauss(16.9, 0.45 + noise)
        i = random.gauss(16.6, 0.45 + noise)
        z = random.gauss(16.3, 0.5 + noise)
        redshift = abs(random.gauss(0.02, 0.02 + noise))
        temperature = random.gauss(5800, 1200 + noise * 800)
        logg = random.gauss(4.3, 0.35 + noise * 0.4)
        metallicity = random.gauss(-0.05, 0.25 + noise * 0.2)
    elif cls == "GALAXY":
        u = random.gauss(20.8, 0.7 + noise)
        g = random.gauss(20.1, 0.65 + noise)
        r = random.gauss(19.5, 0.6 + noise)
        i = random.gauss(19.1, 0.6 + noise)
        z = random.gauss(18.7, 0.65 + noise)
        redshift = abs(random.gauss(0.22, 0.12 + noise * 0.2))
        temperature = random.gauss(5200, 900 + noise * 600)
        logg = random.gauss(3.4, 0.45 + noise * 0.5)
        metallicity = random.gauss(-0.2, 0.35 + noise * 0.25)
    else:
        u = random.gauss(19.2, 0.9 + noise)
        g = random.gauss(19.1, 0.85 + noise)
        r = random.gauss(19.0, 0.8 + noise)
        i = random.gauss(18.8, 0.8 + noise)
        z = random.gauss(18.6, 0.8 + noise)
        redshift = abs(random.gauss(1.4, 0.6 + noise * 0.6))
        temperature = random.gauss(7500, 1800 + noise * 1200)
        logg = random.gauss(3.8, 0.6 + noise * 0.6)
        metallicity = random.gauss(-0.35, 0.4 + noise * 0.3)

    return {
        "u": rnd(clamp(u, 13.5, 25.0)),
        "g": rnd(clamp(g, 13.0, 24.5)),
        "r": rnd(clamp(r, 12.8, 24.2)),
        "i": rnd(clamp(i, 12.5, 24.0)),
        "z": rnd(clamp(z, 12.3, 23.8)),
        "redshift": rnd(clamp(redshift, 0.0, 3.5)),
        "temperature": rnd(clamp(temperature, 2500, 20000)),
        "logg": rnd(clamp(logg, 0.0, 6.0)),
        "metallicity": rnd(clamp(metallicity, -2.5, 0.8)),
        "class": cls,
    }


def build_dataset(size: int, probs: dict[str, float], noise: float) -> list[dict]:
    """按类别分布与噪声等级生成样本列表。"""
    labels = list(probs.keys())
    weights = [probs[k] for k in labels]
    rows: list[dict] = []
    for _ in range(size):
        cls = random.choices(labels, weights=weights, k=1)[0]
        rows.append(gen_row(cls, noise))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """将数据写入 CSV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "u",
        "g",
        "r",
        "i",
        "z",
        "redshift",
        "temperature",
        "logg",
        "metallicity",
        "class",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """生成多个模拟数据集。"""
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "datasets"
    random.seed(42)

    write_csv(
        out_dir / "sdss_like_small.csv",
        build_dataset(
            size=300,
            probs={"STAR": 0.6, "GALAXY": 0.3, "QSO": 0.1},
            noise=0.05,
        ),
    )

    write_csv(
        out_dir / "sdss_like_balanced.csv",
        build_dataset(
            size=1200,
            probs={"STAR": 0.34, "GALAXY": 0.33, "QSO": 0.33},
            noise=0.08,
        ),
    )

    write_csv(
        out_dir / "sdss_like_noisy.csv",
        build_dataset(
            size=600,
            probs={"STAR": 0.5, "GALAXY": 0.35, "QSO": 0.15},
            noise=0.18,
        ),
    )


if __name__ == "__main__":
    main()
