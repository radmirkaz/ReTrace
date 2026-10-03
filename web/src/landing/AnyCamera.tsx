import { useEffect, useState } from 'react'
import TrackedVideo from '../components/TrackedVideo'
import { asset, type LandingScene, type Tracks } from '../data'
import { usePinProgress, span } from '../hooks'

export default function AnyCamera({ scenes }: { scenes: LandingScene[] }) {
  const [ref, p] = usePinProgress<HTMLElement>()
  const [tracks, setTracks] = useState<Record<string, Tracks>>({})
  useEffect(() => {
    scenes.forEach((s) => fetch(asset(s.tracks)).then((r) => r.json()).then((t: Tracks) => setTracks((m) => ({ ...m, [s.cam]: t }))))
  }, [scenes])

  const segs = scenes.length
  const seg = 1 / segs
  const active = Math.min(segs - 1, Math.floor(p / seg))
  const cur = scenes[Math.min(active, scenes.length - 1)]
  const items = scenes.map((s) => [s.title, s.sub])
  const curTracks = cur ? tracks[cur.cam] : undefined
  const q = span(p, active * seg, (active + 1) * seg)
  const curFrame = curTracks ? Math.round(span(q, 0.05, 0.95) * (curTracks.n - 1)) : 0
  const count = curTracks ? curTracks.frames[curFrame]?.length ?? 0 : 0

  return (
    <section ref={ref} style={{ position: 'relative', height: `${segs * 110}vh` }}>
      <div style={{ position: 'sticky', top: 0, height: '100vh', display: 'flex', alignItems: 'center' }}>
        <div className="wrap" style={{ display: 'flex', flexWrap: 'wrap', gap: 40, alignItems: 'center' }}>
          <div style={{ flex: '1 1 300px', maxWidth: 380 }}>
            <div className="label" style={{ color: 'var(--teal)' }}>01 — Any camera, any condition</div>
            <h2 style={{ fontSize: 44, lineHeight: 1.04, letterSpacing: -1.5, margin: '14px 0 26px' }}>Plug into the cameras a city already has.</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {items.map(([title, sub], i) => (
                <div key={title} style={{ display: 'flex', gap: 14, padding: '12px 14px', borderRadius: 8, border: `1px solid ${i === active ? 'var(--teal)' : 'transparent'}`, background: i === active ? 'rgba(94,234,212,.06)' : 'transparent', opacity: i === active ? 1 : 0.45, transition: 'all .3s' }}>
                  <span className="mono" style={{ fontSize: 12, color: i === active ? 'var(--teal)' : 'var(--faint)', paddingTop: 3 }}>0{i + 1}</span>
                  <div><div style={{ fontWeight: 700 }}>{title}</div><div style={{ fontSize: 13, color: '#93a4b8', marginTop: 3 }}>{sub}</div></div>
                </div>
              ))}
            </div>
            <div style={{ height: 2, marginTop: 22, background: '#141c28' }}><div style={{ height: '100%', width: `${p * 100}%`, background: 'var(--teal)' }} /></div>
          </div>

          <div style={{ flex: '2 1 560px', minWidth: 0 }}>
            <div style={{ position: 'relative', aspectRatio: '16 / 9', background: 'var(--panel)', border: '1px solid var(--line)', borderRadius: 10, overflow: 'hidden' }}>
              {scenes.map((s, k) => {
                const o = Math.min(k === 0 ? 1 : span(p, k * seg - 0.02, k * seg + 0.005), 1 - span(p, (k + 1) * seg - 0.02, (k + 1) * seg + 0.005))
                if (o <= 0 && k !== active) return null
                const local = span(p, k * seg, (k + 1) * seg)
                return (
                  <div key={s.cam} style={{ position: 'absolute', inset: 0, display: 'flex', justifyContent: 'center', opacity: o }}>
                    <TrackedVideo
                      src={asset(s.clip)}
                      poster={asset(s.poster)}
                      tracks={tracks[s.cam]}
                      progress={span(local, 0.05, 0.95)}
                      reveal={(order) => span(local, 0.12 + order * 0.04, 0.2 + order * 0.04)}
                      className="scene"
                    >
                      <div style={{ position: 'absolute', left: 0, right: 0, height: '18%', top: `${-18 + 118 * span(local, 0.02, 0.3)}%`, opacity: local > 0.02 && local < 0.3 ? 1 : 0, background: 'linear-gradient(transparent, rgba(94,234,212,.28) 85%, #5eead4 86%, transparent 88%)' }} />
                    </TrackedVideo>
                  </div>
                )
              })}
              <span className="chip" style={{ position: 'absolute', left: 12, top: 12, display: 'flex', gap: 8, alignItems: 'center', color: 'var(--text)' }}>
                <span className="blink" style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--rose)' }} />
                {cur?.title.toUpperCase()}
              </span>
              <span className="chip" style={{ position: 'absolute', right: 12, top: 12 }}>{cur?.res}</span>
              <span className="chip" style={{ position: 'absolute', left: 12, bottom: 12, color: 'var(--text)', fontSize: 12 }}>VEHICLES <b style={{ color: 'var(--teal)' }}>{count}</b></span>
              <span className="chip" style={{ position: 'absolute', right: 12, bottom: 12 }}>YOLOv12 + BYTETRACK · NO FINE-TUNING</span>
            </div>
            <div className="mono" style={{ marginTop: 10, fontSize: 11, color: 'var(--faint)' }}>Scroll scrubs the real footage; every box is the detector’s own output.</div>
          </div>
        </div>
      </div>
    </section>
  )
}
