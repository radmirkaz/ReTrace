import { useEffect, useMemo, useState } from 'react'
import type { Camera } from '../data'

// City of Vancouver bounding box (lon/lat).
const B = { w: -123.235, e: -123.015, s: 49.195, n: 49.318 }
const K = Math.cos((49.26 * Math.PI) / 180)
const W = 1000
const H = Math.round((W * (B.n - B.s)) / ((B.e - B.w) * K))
const px = (lon: number, lat: number): [number, number] => [((lon - B.w) / (B.e - B.w)) * W, ((B.n - lat) / (B.n - B.s)) * H]

type Line = [number, number][]
interface MapData { water: Line[]; parks?: Line[]; major: Line[]; minor: Line[]; coast?: Line[] }

const path = (lines: Line[], close = false) =>
  lines.map((l) => l.map(([lon, lat], i) => `${i ? 'L' : 'M'}${px(lon, lat).map((v) => v.toFixed(1)).join(' ')}`).join('') + (close ? 'Z' : '')).join('')

const COND: Record<Camera['condition'], string> = { day: '#5eead4', dusk: '#fbbf24', night: '#a78bfa', rain: '#60a5fa', snow: '#e2e8f0' }

interface Props {
  cameras: Camera[]
  selected?: string
  highlight?: string[]
  onSelect: (id: string) => void
}

/** Stylised Vancouver: OpenStreetMap water/roads from /map/vancouver.json, or a city-grid fallback. */
export default function VancouverMap({ cameras, selected, highlight = [], onSelect }: Props) {
  const [map, setMap] = useState<MapData | null>(null)
  const [hover, setHover] = useState<string>()
  useEffect(() => { fetch('/map/vancouver.json').then((r) => (r.ok ? r.json() : null)).then(setMap).catch(() => setMap(null)) }, [])

  const layers = useMemo(() => (map ? { water: path(map.water, true), parks: path(map.parks ?? [], true), major: path(map.major), minor: path(map.minor), coast: path(map.coast ?? []) } : null), [map])
  const trail = highlight.map((id) => cameras.find((c) => c.id === id)).filter(Boolean) as Camera[]

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" style={{ width: '100%', height: '100%', display: 'block' }} role="img" aria-label="Map of Vancouver with camera locations">
      <defs>
        <pattern id="vm-grid" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0H0V20" fill="none" stroke="rgba(94,234,212,.05)" /></pattern>
        <radialGradient id="vm-glow"><stop offset="0" stopColor="rgba(94,234,212,.35)" /><stop offset="1" stopColor="rgba(94,234,212,0)" /></radialGradient>
      </defs>
      <rect width={W} height={H} fill="#070b12" />
      <rect width={W} height={H} fill="url(#vm-grid)" />
      {layers ? (
        <>
          <path d={layers.parks} fill="rgba(94,234,212,.035)" stroke="rgba(94,234,212,.08)" strokeWidth="0.6" />
          <path d={layers.water} fill="#0a1828" stroke="#1d4a6b" strokeWidth="0.8" />
          <path d={layers.coast} fill="none" stroke="rgba(56,189,248,.18)" strokeWidth="6" strokeLinejoin="round" />
          <path d={layers.coast} fill="none" stroke="#38bdf8" strokeWidth="1.2" strokeLinejoin="round" opacity=".75" />
          <path d={layers.minor} fill="none" stroke="#1a2635" strokeWidth="0.7" strokeLinecap="round" />
          <path d={layers.major} fill="none" stroke="#2c4a64" strokeWidth="1.5" strokeLinecap="round" />
          <path d={layers.major} fill="none" stroke="rgba(94,234,212,.12)" strokeWidth="4" strokeLinecap="round" />
          <text x={W - 8} y={H - 8} textAnchor="end" fontFamily="JetBrains Mono, monospace" fontSize="8" fill="#3d4a5c">Demo footage from public datasets; camera placements are illustrative · © OpenStreetMap contributors</text>
        </>
      ) : (
        <FallbackGrid />
      )}

      {trail.length > 1 && (
        <polyline points={trail.map((c) => px(c.lon, c.lat).join(',')).join(' ')} fill="none" stroke="#fbbf24" strokeWidth="2.5" strokeDasharray="6 6">
          <animate attributeName="stroke-dashoffset" from="24" to="0" dur="1s" repeatCount="indefinite" />
        </polyline>
      )}

      {cameras.map((c) => {
        const [x, y] = px(c.lon, c.lat)
        const on = c.id === selected
        const hit = highlight.includes(c.id)
        const col = hit ? '#fbbf24' : COND[c.condition]
        return (
          <g key={c.id} transform={`translate(${x} ${y})`} style={{ cursor: 'pointer' }} onMouseEnter={() => setHover(c.id)} onMouseLeave={() => setHover(undefined)} onFocus={() => setHover(c.id)} onBlur={() => setHover(undefined)} onClick={() => onSelect(c.id)} role="button" tabIndex={0} aria-label={c.name} onKeyDown={(e) => { if (e.key === 'Enter') onSelect(c.id) }}>
            <circle r={on ? 40 : 26} fill="url(#vm-glow)" opacity={on || hit ? 1 : 0.5} />
            <circle r="9" fill="none" stroke={col} strokeWidth="1.5" opacity=".6">
              <animate attributeName="r" from="6" to="22" dur="2.4s" repeatCount="indefinite" />
              <animate attributeName="opacity" from=".7" to="0" dur="2.4s" repeatCount="indefinite" />
            </circle>
            <circle r={on ? 7 : 5} fill={col} stroke="#05070b" strokeWidth="2" />
            {(on || hit || hover === c.id) && (
              <text x="12" y="4" fontFamily="JetBrains Mono, monospace" fontSize={13} fill="#e6edf5" style={{ paintOrder: 'stroke', stroke: '#070b12', strokeWidth: 4, pointerEvents: 'none' }}>{c.name}</text>
            )}
          </g>
        )
      })}
    </svg>
  )
}

/** Shown until real OSM geometry is bundled: Vancouver's street grid with approximate shoreline. */
function FallbackGrid() {
  const water: Line[] = [
    [[-123.235, 49.318], [-123.015, 49.318], [-123.015, 49.293], [-123.06, 49.29], [-123.1, 49.288], [-123.115, 49.29], [-123.125, 49.3], [-123.14, 49.316], [-123.16, 49.302], [-123.145, 49.287], [-123.15, 49.275], [-123.2, 49.272], [-123.235, 49.275]],
    [[-123.145, 49.276], [-123.12, 49.273], [-123.1, 49.272], [-123.1, 49.268], [-123.13, 49.268], [-123.146, 49.271]],
    [[-123.235, 49.195], [-123.015, 49.195], [-123.015, 49.2], [-123.1, 49.205], [-123.2, 49.207], [-123.235, 49.21]],
  ]
  const lines: string[] = []
  for (let lat = 49.205; lat < 49.29; lat += 0.0045) lines.push(`M${px(-123.225, lat).join(' ')}L${px(-123.023, lat).join(' ')}`)
  for (let lon = -123.225; lon < -123.02; lon += 0.0068) lines.push(`M${px(lon, 49.2).join(' ')}L${px(lon, 49.288).join(' ')}`)
  return (
    <>
      <path d={lines.join('')} stroke="#141d29" strokeWidth="0.8" />
      <path d={path(water, true)} fill="#0b1a2a" stroke="#14304a" />
    </>
  )
}
