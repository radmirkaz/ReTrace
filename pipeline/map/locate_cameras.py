"""Snap every camera to its real Vancouver intersection using the OSM street names.

    python pipeline/map/locate_cameras.py   # -> pipeline/map/camera_locations.json

An intersection is the node shared by the two named streets (closest pair of points as a
fallback); a bridge camera sits at the middle of the bridge span. Roundabout cameras are
placed along Cambie St as a corridor.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

CROSS = {
    "hwy1-boundary": ("Trans-Canada Highway", "Boundary Road"),
    "main-broadway": ("Main Street", "East Broadway"),
    "granville-broadway": ("Granville Street", "West Broadway"),
    "cambie-41st": ("Cambie Street", "West 41st Avenue"),
    "georgia-denman": ("West Georgia Street", "Denman Street"),
    "hastings-commercial": ("East Hastings Street", "Commercial Drive"),
    "marine-main": ("Southeast Marine Drive", "Main Street"),
    "broadway-arbutus": ("West Broadway", "Arbutus Street"),
    "broadway-oak": ("West Broadway", "Oak Street"),
    "broadway-commercial": ("East Broadway", "Commercial Drive"),
    "kingsway-fraser": ("Kingsway", "Fraser Street"),
    "kingsway-knight": ("Kingsway", "Knight Street"),
    "kingsway-victoria": ("Kingsway", "Victoria Drive"),
    "kingsway-nanaimo": ("Kingsway", "Nanaimo Street"),
    "kingsway-joyce": ("Kingsway", "Joyce Street"),
    "cambie-king-edward-n": ("Cambie Street", "West Broadway"),
    "cambie-king-edward-e": ("Cambie Street", "West 16th Avenue"),
    "cambie-king-edward-s": ("Cambie Street", "King Edward Avenue"),
    "cambie-king-edward-w": ("Cambie Street", "West 33rd Avenue"),
}
# Display names that differ from the export spec (the four RoundaboutHD feeds form a Cambie St corridor).
NAMES = {
    "cambie-king-edward-n": "Cambie St & Broadway",
    "cambie-king-edward-e": "Cambie St & 16th Ave",
    "cambie-king-edward-s": "Cambie St & King Edward Ave",
    "cambie-king-edward-w": "Cambie St & 33rd Ave",
}
BRIDGES = {"knight-bridge": "Knight", "oak-bridge": "Oak Street", "lions-gate": "Lions Gate"}


def main():
    ways = [e for e in json.loads((HERE / "vancouver_osm.json").read_text())["elements"] if e.get("type") == "way" and "highway" in e.get("tags", {})]

    def points(name, exact=True):
        out = []
        for w in ways:
            n = w["tags"].get("name", "")
            if (n == name if exact else name.lower() in n.lower()):
                out += [(g["lat"], g["lon"]) for g in w["geometry"]]
        return out

    locs = {}
    for cid, (a, b) in CROSS.items():
        pa, pb = points(a) or points(a, False), points(b) or points(b, False)
        if not pa or not pb:
            print("NOT FOUND", cid, a if not pa else b)
            continue
        shared = set(pa) & set(pb)
        if shared:
            lat, lon = min(shared)
            how = "shared node"
        else:
            p, q = min(((p, q) for p in pa for q in pb), key=lambda pq: (pq[0][0] - pq[1][0]) ** 2 + (pq[0][1] - pq[1][1]) ** 2)
            lat, lon = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
            how = "closest points"
        locs[cid] = [round(lat, 6), round(lon, 6)]
        print(f"{cid:22} {lat:.5f} {lon:.5f}  ({how})")

    for cid, key in BRIDGES.items():
        pts = [(g["lat"], g["lon"]) for w in ways if w["tags"].get("bridge") == "yes" and key.lower() in w["tags"].get("name", "").lower() for g in w["geometry"]]
        if not pts:
            print("NOT FOUND", cid)
            continue
        pts.sort()
        lat, lon = pts[len(pts) // 2]
        locs[cid] = [round(lat, 6), round(lon, 6)]
        print(f"{cid:22} {lat:.5f} {lon:.5f}  (bridge mid-span)")

    for cid, name in NAMES.items():
        if cid in locs:
            locs[cid].append(name)
    (HERE / "camera_locations.json").write_text(json.dumps(locs, indent=1))
    print(len(locs), "camera locations")


if __name__ == "__main__":
    main()
