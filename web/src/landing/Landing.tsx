import { useEffect, useState } from 'react'
import { asset, loadLanding, loadVehicles, type LandingData, type Vehicle } from '../data'
import AnyCamera from './AnyCamera'
import Problem from './Problem'
import TraceOne from './TraceOne'
import Witness from './Witness'

export default function Landing() {
  const [data, setData] = useState<LandingData | null>(null)
  const [vehicles, setVehicles] = useState<Record<string, Vehicle>>({})
  const [error, setError] = useState('')
  useEffect(() => {
    loadLanding().then(async (l) => { setData(l); setVehicles(l.vehicles ?? (await loadVehicles())) }).catch((e) => setError(String(e)))
  }, [])

  return (
    <main>
      <Hero poster={data ? asset(data.trace.poster) : undefined} />
      <Problem />
      {error && <p className="wrap mono" style={{ color: 'var(--rose)' }}>Could not load footage: {error}</p>}
      {data && (
        <>
          <AnyCamera scenes={data.scenes} />
          {data.witness && <Witness data={data.witness} vehicles={vehicles} />}
          <TraceOne trace={data.trace} vehicles={vehicles} />
        </>
      )}
      <Outro />
    </main>
  )
}

function Hero({ poster }: { poster?: string }) {
  return (
    <section style={{ position: 'relative', minHeight: 'calc(100vh - 72px)', overflow: 'hidden', display: 'flex', alignItems: 'center' }}>
      {poster && <img src={poster} alt="" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', opacity: 0.3, filter: 'saturate(.6)' }} />}
      <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(180deg, rgba(5,7,11,.35), #05070b 92%)' }} />
      <div className="grid-bg" style={{ position: 'absolute', inset: 0, backgroundSize: '64px 64px' }} />
      <div className="wrap" style={{ position: 'relative' }}>
        <div className="label" style={{ color: 'var(--teal)' }}>Vehicle re-identification across cameras</div>
        <h1 style={{ fontSize: 'clamp(48px, 8vw, 92px)', lineHeight: 0.98, letterSpacing: -3, margin: '20px 0 22px' }}>
          See every car.<br /><span style={{ color: 'var(--teal)' }}>Trace any one.</span>
        </h1>
        <p style={{ fontSize: 20, lineHeight: 1.5, color: 'var(--muted)', maxWidth: 580, margin: '0 0 32px' }}>
          ReTrace turns the cameras a city already has into a searchable memory of every vehicle, found by appearance,
          by description, or by a witness’s voice. No licence plate needed.
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
          <a className="btn btn-primary" href="#/live">Open the live map</a>
          <a className="btn" href="#problem">Scroll the story ↓</a>
        </div>
      </div>
    </section>
  )
}

function Outro() {
  return (
    <section className="wrap" style={{ textAlign: 'center', padding: '140px 28px 120px' }}>
      <h2 style={{ fontSize: 'clamp(40px, 6vw, 64px)', lineHeight: 1.02, letterSpacing: -2, margin: '0 0 18px' }}>
        From raw video<br />to a car’s <span style={{ color: 'var(--teal)' }}>whole story.</span>
      </h2>
      <p style={{ color: 'var(--muted)', margin: '0 0 30px', fontSize: 18 }}>No new cameras. No per-site training. Open the city map and pick a camera.</p>
      <a className="btn btn-primary" href="#/live">Open the live map →</a>
    </section>
  )
}
