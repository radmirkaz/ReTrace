import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import TrackedVideo from '../components/TrackedVideo'
import { asset, vehicleTitle, type LandingData, type Tracks, type Vehicle } from '../data'
import { usePinProgress, ease, span } from '../hooks'
import { Header } from './Witness'

interface Rect { x: number; y: number; w: number; h: number }

export default function TraceOne({ trace, vehicles }: { trace: LandingData['trace']; vehicles: Record<string, Vehicle> }) {
  const [ref, p] = usePinProgress<HTMLElement>()
  const [tracks, setTracks] = useState<Tracks | null>(null)
  useEffect(() => { fetch(asset(trace.tracks)).then((r) => r.json()).then(setTracks) }, [trace.tracks])
  const v = vehicles[trace.vehicle]

  const stage = useRef<HTMLDivElement>(null)
  const videoBox = useRef<HTMLDivElement>(null)
  const slot = useRef<HTMLDivElement>(null)
  const [geo, setGeo] = useState<{ video: Rect; slot: Rect } | null>(null)
  useLayoutEffect(() => {
    const measure = () => {
      if (!stage.current || !videoBox.current || !slot.current) return
      const s = stage.current.getBoundingClientRect()
      const rel = (e: Element): Rect => { const r = e.getBoundingClientRect(); return { x: r.left - s.left, y: r.top - s.top, w: r.width, h: r.height } }
      setGeo({ video: rel(videoBox.current), slot: rel(slot.current) })
    }
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [tracks])

  const targetFrame = tracks ? Math.min(tracks.n - 1, Math.round(trace.t * tracks.fps)) : 0
  const scrub = tracks ? (targetFrame / Math.max(1, tracks.n - 1)) * span(p, 0, 0.12) : 0
  const iso = span(p, 0.12, 0.22)
  const pull = ease(span(p, 0.26, 0.46))
  const info = span(p, 0.44, 0.48)
  const mapO = span(p, 0.7, 0.76)
  const phase = p < 0.12 ? 0 : p < 0.26 ? 1 : p < 0.46 ? 2 : p < 0.68 ? 3 : 4
  const others = (v?.journey && v.journey.length > 1 ? v.journey.filter((s) => !s.self) : (v?.sightings ?? []).slice(1)).slice(0, 4)
  const verified = others.some((s) => s.correct !== undefined && s.correct !== null)
  const right = others.filter((s) => s.correct).length
  const captions = ['Every vehicle, detected.', 'Pick one.', 'Pull it out of the video.', 'Know what it is.', verified && right === others.length ? 'Found again on every camera.' : 'Find it on other cameras.']

  // Box of the target car at the held frame, in stage pixels.
  const det = tracks?.frames[targetFrame]?.find((d) => d[0] === trace.track)
  let from: Rect | null = null
  if (geo && tracks && det) {
    const [, x1, y1, x2, y2] = det
    from = { x: geo.video.x + (x1 / tracks.w) * geo.video.w, y: geo.video.y + (y1 / tracks.h) * geo.video.h, w: ((x2 - x1) / tracks.w) * geo.video.w, h: ((y2 - y1) / tracks.h) * geo.video.h }
  }
  const to = geo?.slot
  const lerp = (a: number, b: number) => a + (b - a) * pull
  const crop = from && to ? { left: lerp(from.x, to.x), top: lerp(from.y, to.y), width: lerp(from.w, to.w), height: lerp(from.h, to.h) } : null

  return (
    <section id="trace" ref={ref} style={{ position: 'relative', height: '520vh' }}>
      <div style={{ position: 'sticky', top: 0, height: '100vh', display: 'flex', alignItems: 'center' }}>
        <div className="wrap">
          <Header label="03 — Trace one car" color="var(--amber)" title={captions[phase]} phase={phase} />
          <div ref={stage} style={{ position: 'relative', display: 'flex', gap: '4%', alignItems: 'flex-start' }}>
            <div ref={videoBox} style={{ position: 'relative', width: '60%', flex: 'none', borderRadius: 10, overflow: 'hidden', border: '1px solid var(--line)' }}>
              <div style={{ opacity: 1 - mapO }}>
                <TrackedVideo src={asset(trace.clip)} poster={asset(trace.poster)} tracks={tracks} progress={scrub} highlight={pull < 0.95 ? trace.track : -1} othersOpacity={1 - iso}>
                  <div style={{ position: 'absolute', inset: 0, background: '#05070b', opacity: 0.55 * iso, pointerEvents: 'none' }} />
                </TrackedVideo>
              </div>
              <div className="grid-bg" style={{ position: 'absolute', inset: 0, background: '#070b12', opacity: mapO, padding: 18, display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div className="mono" style={{ fontSize: 11, color: '#93a4b8' }}>{verified ? `RE-ID MODEL'S MATCHES · ${right}/${others.length} CONFIRMED BY GROUND TRUTH` : 'NEAREST APPEARANCE MATCHES ON OTHER CAMERAS · COSINE SIMILARITY'}</div>
                <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.max(1, others.length)}, minmax(0, 1fr))`, gap: 12, flex: 1 }}>
                  {others.map((s, i) => {
                    const k = span(p, 0.76 + i * 0.05, 0.8 + i * 0.05)
                    return (
                      <div key={s.cam + s.track} style={{ opacity: k, transform: `translateY(${16 * (1 - k)}px)`, borderRadius: 8, border: '1px solid var(--line-2)', background: 'var(--panel-2)', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                        <img src={asset(s.crop)} alt="" style={{ width: '100%', aspectRatio: '4 / 3', objectFit: 'cover' }} />
                        <div className="mono" style={{ padding: 10, fontSize: 11, lineHeight: 1.6 }}>
                          {s.cam.toUpperCase()} · #{s.track}<br />
                          <span style={{ color: 'var(--amber)' }}>sim {s.sim.toFixed(2)}</span>{s.correct === true && <><br /><span style={{ color: 'var(--teal)' }}>✓ same car</span></>}{s.correct === false && <><br /><span style={{ color: 'var(--rose)' }}>✗ wrong match</span></>}
                        </div>
                      </div>
                    )
                  })}
                </div>
                <div className="mono" style={{ fontSize: 10, color: 'var(--faint)' }}>{verified ? 'Matches are proposed by the re-ID model (appearance + body/colour filter + mutual best match); CityFlowV2 labels grade each one.' : 'These feeds come from unrelated videos, so matches are look-alikes, not the same car.'}</div>
              </div>
            </div>

            <div style={{ width: '36%', flex: 'none' }}>
              <div ref={slot} style={{ width: '100%', aspectRatio: '4 / 3' }} />
              {v && (
                <div style={{ marginTop: 14, padding: 18, borderRadius: 10, border: '1px solid var(--line)', background: 'var(--panel)', opacity: info, transform: `translateY(${16 * (1 - info)}px)` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                    <div style={{ fontWeight: 700, fontSize: 18 }}>{vehicleTitle(v)} · #{trace.track}</div>
                    <span className="chip">{trace.cam.toUpperCase()}</span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 14 }}>
                    {v.top5.map((r, i) => {
                      const o = span(p, 0.48 + i * 0.03, 0.52 + i * 0.03)
                      const g = span(p, 0.5 + i * 0.03, 0.58 + i * 0.03)
                      return (
                        <div key={r.name} style={{ opacity: o }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}><span>{r.name}</span><span className="mono" style={{ color: i ? 'var(--faint)' : 'var(--amber)' }}>{r.p.toFixed(2)}</span></div>
                          <div style={{ height: 4, marginTop: 4, borderRadius: 2, background: '#141c28' }}><div style={{ height: '100%', borderRadius: 2, width: `${r.p * 100 * g}%`, background: i ? 'var(--faint)' : 'var(--amber)' }} /></div>
                        </div>
                      )
                    })}
                  </div>
                  <div className="mono" style={{ marginTop: 14, fontSize: 10, color: 'var(--dim)' }}>2048-D APPEARANCE FINGERPRINT</div>
                  <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 34, marginTop: 6 }}>
                    {Array.from({ length: 40 }, (_, i) => <span key={i} style={{ flex: 1, height: `${18 + ((i * 37 + 11) % 77)}%`, background: 'var(--amber)', opacity: span(p, 0.6 + i * 0.002, 0.63 + i * 0.002) }} />)}
                  </div>
                </div>
              )}
            </div>

            {crop && v && (
              <div style={{ position: 'absolute', ...crop, opacity: p > 0.25 ? 1 : 0, borderRadius: 4, overflow: 'hidden', border: '2px solid var(--amber)', boxShadow: `0 0 ${40 * pull}px rgba(251,191,36,.35)`, pointerEvents: 'none' }}>
                <img src={asset(v.crop)} alt="Selected vehicle" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
