import { asset, type LandingData, type Vehicle } from '../data'
import { usePinProgress, useTicker, span } from '../hooks'

const KEYWORDS: Record<string, string> = {
  white: 'colour · white', black: 'colour · black', red: 'colour · red', blue: 'colour · blue', silver: 'colour · silver',
  sedan: 'body · sedan', suv: 'body · SUV', pickup: 'body · pickup', truck: 'body · truck', van: 'body · van', hatchback: 'body · hatchback',
  bmw: 'make · BMW?', honda: 'make · Honda?', toyota: 'make · Toyota?', ford: 'make · Ford?',
  east: 'heading · east', west: 'heading · west', north: 'heading · north', south: 'heading · south',
  kingsway: 'street · Kingsway', hastings: 'street · Hastings', broadway: 'street · Broadway',
}

export default function Witness({ data, vehicles }: { data: NonNullable<LandingData['witness']>; vehicles: Record<string, Vehicle> }) {
  const [ref, p] = usePinProgress<HTMLElement>()
  const t = useTicker(90)
  const words = data.transcript.split(' ')
  const speaking = p > 0.08 && p < 0.4
  const shown = Math.round(span(p, 0.08, 0.4) * words.length)
  const keys = words.map((w) => KEYWORDS[w.toLowerCase().replace(/[^a-z]/g, '')]).map((k, i) => (k ? { k, i } : null)).filter(Boolean) as { k: string; i: number }[]
  const ranked = p >= 0.66
  const typed = data.query.slice(0, Math.round(span(p, 0.56, 0.63) * data.query.length))
  const hits = data.hits.slice(0, 8)
  const shuffled = [3, 6, 1, 7, 0, 5, 2, 4].filter((i) => i < hits.length)
  const fill = span(p, 0.66, 0.74)
  const lead = hits[0] && vehicles[hits[0].gid]
  const leadK = span(p, 0.84, 0.89)
  const phase = p < 0.08 ? 0 : p < 0.42 ? 1 : p < 0.58 ? 2 : p < 0.84 ? 3 : 4
  const captions = ['A witness doesn’t remember the plate.', 'But they remember the car.', 'Whisper turns their words into a search.', 'CLIP ranks every vehicle crop.', 'One lead — then trace it.']
  const maxScore = hits[0]?.score || 1

  return (
    <section ref={ref} style={{ position: 'relative', height: '480vh' }}>
      <div style={{ position: 'sticky', top: 0, height: '100vh', display: 'flex', alignItems: 'center' }}>
        <div className="wrap">
          <Header label="02 — Hit-and-run · the witness speaks" color="var(--violet)" title={captions[phase]} phase={phase} />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px 4%', alignItems: 'flex-start' }}>
            <div style={{ flex: '0 0 38%', minWidth: 300, padding: 22, borderRadius: 14, border: '1px solid var(--line)', background: 'var(--panel)' }}>
              <div className="mono" style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--dim)' }}><span>WITNESS STATEMENT · VOICE</span><span>21:52</span></div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16, margin: '18px 0' }}>
                <div style={{ position: 'relative', width: 56, height: 56, flex: 'none' }}>
                  <div style={{ position: 'absolute', inset: -8, borderRadius: '50%', border: '2px solid var(--violet)', opacity: speaking ? 0.3 + 0.5 * Math.abs(Math.sin(t * 0.3)) : 0, transform: `scale(${speaking ? 1 + 0.15 * Math.abs(Math.sin(t * 0.3)) : 1})` }} />
                  <div aria-hidden style={{ width: 56, height: 56, borderRadius: '50%', background: speaking ? 'var(--violet)' : 'var(--faint)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#05070b" strokeWidth="2" strokeLinecap="round"><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></svg>
                  </div>
                </div>
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 2, height: 44 }}>
                  {Array.from({ length: 36 }, (_, i) => {
                    const a = speaking ? Math.abs(Math.sin(t * 0.55 + i * 0.7)) * Math.abs(Math.sin(i * 1.3 + t * 0.21)) : 0
                    return <span key={i} style={{ flex: 1, height: `${10 + 90 * a}%`, borderRadius: 2, background: 'var(--violet)', opacity: speaking ? 1 : 0.35 }} />
                  })}
                </div>
              </div>
              <p style={{ fontSize: 20, lineHeight: 1.5, margin: '0 0 16px', minHeight: 120 }}>
                “{words.map((w, i) => {
                  const ki = keys.findIndex((k) => k.i === i)
                  const lit = ki >= 0 && p > 0.42 + ki * 0.025
                  return <span key={i} style={{ opacity: i < shown ? 1 : 0.08, color: lit ? 'var(--teal)' : undefined, fontWeight: lit ? 700 : 400, transition: 'color .3s' }}>{w} </span>
                })}”
              </p>
              <div className="mono" style={{ fontSize: 10, color: 'var(--dim)', marginBottom: 8 }}>WHISPER → EXTRACTED</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {keys.map(({ k }, i) => {
                  const o = span(p, 0.42 + i * 0.025, 0.45 + i * 0.025)
                  return <span key={k} className="mono" style={{ fontSize: 12, padding: '5px 9px', borderRadius: 4, border: '1px solid var(--teal)', color: 'var(--teal)', background: 'rgba(94,234,212,.06)', opacity: o, transform: `translateY(${10 * (1 - o)}px)` }}>{k}</span>
                })}
              </div>
            </div>

            <div style={{ flex: '1 1 480px', minWidth: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '14px 16px', borderRadius: 10, border: '1px solid #23344a', background: '#0a111b', marginBottom: 14, opacity: span(p, 0.54, 0.57) }}>
                <span className="mono" style={{ fontSize: 11, color: 'var(--dim)' }}>CLIP</span>
                <span className="mono" style={{ fontSize: 16, flex: 1 }}>{typed}<span className="blink" style={{ display: 'inline-block', width: 8, height: 16, marginLeft: 2, verticalAlign: -2, background: 'var(--teal)' }} /></span>
                <span className="mono" style={{ fontSize: 11, color: '#93a4b8' }}>{ranked ? `TOP ${hits.length} OF ${Object.keys(vehicles).length} CROPS` : typed.length === data.query.length ? 'RANKING…' : ''}</span>
              </div>
              <div style={{ position: 'relative', aspectRatio: '2 / 1' }}>
                {hits.map((h, i) => {
                  const slot = ranked ? i : shuffled.indexOf(i)
                  const top = ranked && i === 0
                  return (
                    <div key={h.gid} style={{ position: 'absolute', left: `${(slot % 4) * 25.5}%`, top: `${Math.floor(slot / 4) * 53}%`, width: '23.5%', height: '47%', transition: 'left .7s cubic-bezier(.2,.8,.2,1), top .7s cubic-bezier(.2,.8,.2,1)', borderRadius: 8, overflow: 'hidden', border: `2px solid ${top && fill > 0.5 ? 'var(--amber)' : 'var(--line)'}`, background: 'var(--panel)', opacity: span(p, 0.58, 0.63) * (ranked && i > 3 ? 0.55 : 1) }}>
                      <img src={asset(h.crop)} alt="" style={{ width: '100%', height: '70%', objectFit: 'cover' }} />
                      <div style={{ padding: '6px 8px' }}>
                        <div className="mono" style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10 }}>
                          <span style={{ color: '#93a4b8' }}>{h.cam.toUpperCase()} · #{h.track}</span>
                          <span style={{ color: top ? 'var(--amber)' : 'var(--teal)' }}>{ranked ? h.score.toFixed(3) : '—'}</span>
                        </div>
                        <div style={{ height: 3, marginTop: 5, borderRadius: 2, background: '#141c28' }}>
                          <div style={{ height: '100%', borderRadius: 2, background: top ? 'var(--amber)' : 'var(--teal)', width: `${(h.score / maxScore) * 100 * fill}%`, transition: 'width .8s' }} />
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
              {lead && (
                <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 14, marginTop: 16, padding: '16px 18px', borderRadius: 10, border: '1px solid var(--amber)', background: 'rgba(251,191,36,.06)', opacity: leadK, transform: `translateY(${14 * (1 - leadK)}px)` }}>
                  <span className="mono" style={{ fontSize: 11, color: 'var(--amber)' }}>LEAD</span>
                  <span style={{ fontWeight: 700 }}>#{hits[0].track} · {lead.top5[0]?.name} · {hits[0].cam}</span>
                  <a href="#trace" className="mono" style={{ marginLeft: 'auto', fontSize: 13, color: 'var(--amber)' }}>Trace it ↓</a>
                </div>
              )}
              <div className="mono" style={{ marginTop: 8, fontSize: 10, color: 'var(--faint)' }}>Scores are raw CLIP cosine similarity for “{data.query}”. Statement scripted for the demo.</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export function Header({ label, color, title, phase }: { label: string; color: string; title: string; phase: number }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'flex-end', gap: 16, marginBottom: 22 }}>
      <div>
        <div className="label" style={{ color }}>{label}</div>
        <h2 style={{ fontSize: 42, lineHeight: 1.05, letterSpacing: -1.5, margin: '10px 0 0' }}>{title}</h2>
      </div>
      <div style={{ display: 'flex', gap: 6 }}>
        {[0, 1, 2, 3, 4].map((i) => <span key={i} style={{ width: 28, height: 4, borderRadius: 2, background: i <= phase ? color : 'var(--line)' }} />)}
      </div>
    </div>
  )
}
