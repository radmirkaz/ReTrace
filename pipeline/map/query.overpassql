[out:json][timeout:90];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"](49.195,-123.235,49.318,-123.015);
  way["natural"="coastline"](49.15,-123.30,49.36,-122.95);
  way["natural"="water"](49.195,-123.235,49.318,-123.015);
  way["leisure"="park"](49.195,-123.235,49.318,-123.015);
);
out geom;
