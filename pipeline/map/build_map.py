"""Turn an Overpass export of Vancouver into the compact line/polygon JSON the web map draws.

    curl --data-urlencode data@query.overpassql https://overpass-api.de/api/interpreter -o vancouver_osm.json
    python build_map.py   # -> web/public/map/vancouver.json

Map data © OpenStreetMap contributors, ODbL.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "web" / "public" / "map" / "vancouver.json"
MAJOR = {"motorway", "trunk", "primary"}


def rdp(pts, eps):
    if len(pts) < 3:
        return pts
    (x1, y1), (x2, y2) = pts[0], pts[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = (dx * dx + dy * dy) ** 0.5 or 1e-12
    dists = [abs(dy * x - dx * y + x2 * y1 - y2 * x1) / norm for x, y in pts[1:-1]]
    i = max(range(len(dists)), key=dists.__getitem__)
    if dists[i] > eps:
        return rdp(pts[: i + 2], eps)[:-1] + rdp(pts[i + 1:], eps)
    return [pts[0], pts[-1]]


def area(pts):
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))) / 2


def main():
    data = json.loads((HERE / "vancouver_osm.json").read_text())
    out = {"major": [], "minor": [], "coast": [], "water": [], "parks": []}
    for el in data["elements"]:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        pts = [(round(g["lon"], 5), round(g["lat"], 5)) for g in el["geometry"]]
        tags = el.get("tags", {})
        if "highway" in tags:
            out["major" if tags["highway"] in MAJOR else "minor"].append(rdp(pts, 0.00004))
        elif tags.get("natural") == "coastline":
            out["coast"].append(rdp(pts, 0.00003))
        elif tags.get("natural") == "water" and area(pts) > 3e-7:
            out["water"].append(rdp(pts, 0.00003))
        elif tags.get("leisure") == "park" and area(pts) > 4e-6:
            out["parks"].append(rdp(pts, 0.00004))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print({k: len(v) for k, v in out.items()}, f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
