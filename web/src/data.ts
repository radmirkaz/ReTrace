// Data contract shared with the backend (backend/app.py) and the offline cache
// written by pipeline/build_cache.py into web/public/cache/.

/** One detection: [trackId, x1, y1, x2, y2, confidence, classId] in source-video pixels. */
export type Det = [number, number, number, number, number, number, number]

export interface Tracks {
  w: number
  h: number
  fps: number
  n: number
  names: Record<string, string>
  frames: Det[][]
}

export interface Camera {
  id: string
  name: string
  lat: number
  lon: number
  /** Where the footage really comes from — shown in the UI so demo feeds are never passed off as Vancouver. */
  source: string
  condition: 'day' | 'dusk' | 'night' | 'rain' | 'snow'
  view: 'overpass' | 'pole' | 'roundabout' | 'intersection'
  res: string
  clip: string
  tracks: string
  poster: string
}

export interface Prediction { name: string; p: number }

export interface Sighting { cam: string; track: number; t: number; crop: string; sim: number; verified?: boolean }

export interface Vehicle {
  gid: string
  crop: string
  top5: Prediction[]
  color?: string
  speed?: number
  quality?: number
  sightings: Sighting[]
}

export interface SearchHit { gid: string; cam: string; track: number; crop: string; score: number }

const API = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '')
const CACHE = '/cache'

export type Mode = 'live' | 'cached'

// Feeds, tracks and vehicles always load from the bundled index (fast, works offline);
// the GPU backend is only used for CLIP search and Whisper transcription.
let mode: Mode = API ? 'live' : 'cached'
export const getMode = () => mode

async function getJson<T>(_livePath: string, cachePath: string): Promise<T> {
  const r = await fetch(CACHE + cachePath, { cache: 'no-cache' })
  if (!r.ok) throw new Error(`missing ${cachePath}`)
  return (await r.json()) as T
}

export const asset = (p: string) => `${CACHE}/${p}`

export const loadCameras = () => getJson<Camera[]>('/cameras', '/cameras.json')
export const loadTracks = (cam: Camera) => getJson<Tracks>(`/cameras/${cam.id}/tracks`, '/' + cam.tracks)
export const loadVehicles = () => getJson<Record<string, Vehicle>>('/vehicles', '/vehicles.json')
export const loadLanding = () => getJson<LandingData>('/landing', '/landing.json')

export async function searchText(q: string): Promise<SearchHit[]> {
  if (mode === 'live' && API) {
    try {
      const r = await fetch(`${API}/search?q=${encodeURIComponent(q)}`, { signal: AbortSignal.timeout(6000) })
      if (r.ok) return r.json()
    } catch {
      mode = 'cached' // backend unreachable: keep the demo running on label matching
    }
  }
  // Offline fallback: match the query words against cached class names and colours.
  const vehicles = await loadVehicles()
  const words = q.toLowerCase().split(/\W+/).filter(Boolean)
  return Object.values(vehicles)
    .map((v) => {
      const hay = `${v.color ?? ''} ${v.top5.map((t) => t.name).join(' ')}`.toLowerCase()
      const score = words.filter((w) => hay.includes(w)).length / Math.max(1, words.length)
      const s = v.sightings[0]
      return { gid: v.gid, cam: s?.cam ?? '', track: s?.track ?? 0, crop: v.crop, score }
    })
    .filter((h) => h.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 12)
}

export async function transcribe(audio: Blob): Promise<string> {
  if (!API) throw new Error('Voice search needs the live backend')
  const body = new FormData()
  body.append('audio', audio, 'query.webm')
  const r = await fetch(`${API}/transcribe`, { method: 'POST', body })
  if (!r.ok) throw new Error('transcription failed')
  return ((await r.json()) as { text: string }).text
}

/** Scenes and the traced car used by the landing page. */
export interface LandingScene { cam: string; title: string; sub: string; clip: string; tracks: string; poster: string; res: string }
export interface LandingData {
  scenes: LandingScene[]
  trace: { cam: string; clip: string; tracks: string; poster: string; track: number; t: number; vehicle: string }
  witness?: { transcript: string; query: string; hits: SearchHit[] }
  vehicles?: Record<string, Vehicle>
}
