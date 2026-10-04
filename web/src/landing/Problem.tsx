import { useRevealProgress, useTicker, ease } from '../hooks'

const PLATES = [
  { plate: 'LKX 492', blur: 0, label: 'Readable, on a good day', color: '#12304f' },
  { plate: 'LKX 492', blur: 6, label: 'Covered in mud or snow', color: '#12304f' },
  { plate: 'GRT 118', blur: 0, label: 'Swapped from another car', color: '#b45309' },
  { plate: '??? ???', blur: 0, label: 'The witness never saw it', color: '#9ca3af' },
]

const card: React.CSSProperties = { padding: 24, borderRadius: 12, border: '1px solid var(--line)', background: 'var(--panel)' }

export default function Problem() {
  const [ref, p] = useRevealProgress<HTMLElement>()
  const t = useTicker(100)
  const e = ease(p / 0.8)
  const dots = Math.round(169 * ease((p - 0.3) / 0.7))
  const plate = PLATES[Math.floor(t / 18) % PLATES.length]

  return (
    <section id="problem" ref={ref} className="wrap" style={{ padding: "120px 28px 100px" }}>
      <div className="label" style={{ color: 'var(--rose)' }}>00 / The problem</div>
      <h2 style={{ fontSize: 'clamp(38px, 5vw, 60px)', lineHeight: 1.02, letterSpacing: -2, margin: '14px 0 18px', maxWidth: 940 }}>
        In Canada, a car is stolen <span style={{ color: 'var(--rose)' }}>every five minutes.</span> Hit-and-run drivers simply drive away.
      </h2>
      <p style={{ fontSize: 19, lineHeight: 1.55, color: 'var(--muted)', maxWidth: 760, margin: '0 0 48px' }}>
        Cameras are scarce, rarely recorded, and built around licence plates: the one thing a thief changes first and a
        hit-and-run witness never catches. What people <em>do</em> remember is the car.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 1, background: 'var(--line)', border: '1px solid var(--line)', borderRadius: 12, overflow: 'hidden', marginBottom: 20 }}>
        {[
          [Math.round(105000 * e).toLocaleString('en-US'), 'vehicles stolen in Canada in 2022'],
          [`C$${(1.5 * e).toFixed(1)}B`, 'in theft claims, called a “national crisis” by insurers'],
          [String(Math.round(221 * e)), 'traffic cameras for 710,000 people in Vancouver'],
        ].map(([v, l]) => (
          <div key={l} style={{ background: '#070b12', padding: 28 }}>
            <div style={{ fontSize: 52, fontWeight: 700, letterSpacing: -2 }}>{v}</div>
            <div style={{ color: '#93a4b8', marginTop: 6 }}>{l}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 20, alignItems: 'stretch' }}>
        <div style={{ ...card, flex: '1.45 1 520px', minWidth: 0 }}>
          <div className="label" style={{ color: 'var(--dim)', fontSize: 11 }}>Cameras per 10,000 people</div>
          <div className="mono" style={{ fontSize: 10, color: 'var(--faint)', marginTop: 4 }}>Dubai: 10,000+ cameras, 4.8M residents, tourists excluded (2026)</div>
          <Row name="Moscow" value="169" tag="mass surveillance · privacy risk" tagColor="var(--rose)" />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
            {Array.from({ length: 169 }, (_, i) => (
              <span key={i} style={{ width: 7, height: 7, borderRadius: 1, background: 'var(--rose)', opacity: i < dots ? 0.7 : 0.12 }} />
            ))}
          </div>
          <Row name="Dubai" value="21+" tag="road-safety coverage · the balance" tagColor="var(--teal)" color="var(--teal)" />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
            {Array.from({ length: 21 }, (_, i) => (
              <span key={i} style={{ width: 7, height: 7, borderRadius: 1, background: 'var(--teal)', opacity: i < Math.round(dots * 21 / 169) ? 1 : 0.12 }} />
            ))}
          </div>
          <Row name="Vancouver" value="3" color="var(--amber)" tag="too few to help" tagColor="var(--amber)" />
          <div style={{ display: 'flex', gap: 3 }}>
            {[0, 1, 2].map((i) => <span key={i} style={{ width: 7, height: 7, borderRadius: 1, background: 'var(--amber)' }} />)}
          </div>
          <div className="mono" style={{ fontSize: 10, color: 'var(--faint)', marginTop: 6 }}>221 traffic cameras, only 140 of them red-light / speed enforcement</div>
          <div className="label" style={{ color: 'var(--dim)', fontSize: 11, margin: '22px 0 10px' }}>Area to cover, km²</div>
          {([['Metro Vancouver', 2883, 'var(--amber)'], ['Moscow', 2562, 'var(--rose)'], ['Dubai (urban)', 1491, 'var(--teal)']] as const).map(([n, km, c]) => (
            <div key={n} style={{ display: 'grid', gridTemplateColumns: '120px 1fr 52px', gap: 10, alignItems: 'center', fontSize: 13, marginBottom: 6 }}>
              <span>{n}</span>
              <div style={{ height: 6, background: '#141c28', borderRadius: 3 }}><div style={{ height: '100%', width: `${(km / 2883) * 100 * Math.min(1, e * 1.2)}%`, background: c, borderRadius: 3, transition: 'width .4s' }} /></div>
              <span className="mono" style={{ color: 'var(--dim)', textAlign: 'right' }}>{km.toLocaleString('en-US')}</span>
            </div>
          ))}
          <p style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.5, margin: '16px 0 0' }}>
            Metro Vancouver is the largest of the three, with a fraction of the cameras.
          </p>
          <p style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.5, margin: '10px 0 0' }}>
            More cameras is not the goal. ReTrace gets more out of the cameras a city has, and it indexes vehicles, not faces or plates.
          </p>
        </div>

        <div style={{ flex: '1 1 380px', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 20 }}>
        <div style={card}>
          <div className="label" style={{ color: 'var(--dim)', fontSize: 11 }}>Plates fail</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 22, marginTop: 18 }}>
            <div style={{ width: 210, flex: 'none', padding: '10px 12px 12px', borderRadius: 8, background: '#f4f4f5', color: '#12304f', textAlign: 'center', boxShadow: 'inset 0 0 0 3px #12304f' }}>
              <div style={{ fontSize: 10, letterSpacing: 2, fontWeight: 700 }}>BRITISH COLUMBIA</div>
              <div className="mono" style={{ fontSize: 38, fontWeight: 600, letterSpacing: 2, filter: `blur(${plate.blur}px)`, color: plate.color, transition: 'filter .4s' }}>{plate.plate}</div>
            </div>
            <div style={{ flex: '1 1 140px', fontSize: 17, lineHeight: 1.35 }}>{plate.label}</div>
          </div>
        </div>

        <div style={{ ...card, flex: 1 }}>
          <div className="label" style={{ color: 'var(--dim)', fontSize: 11 }}>Feeds aren’t kept</div>
          <div style={{ position: 'relative', margin: '16px 0 12px', aspectRatio: '16 / 9', borderRadius: 6, overflow: 'hidden', background: '#111b28' }}>
            <img src="/cache/story/clips/hwy1-boundary.jpg" alt="Traffic camera still" style={{ width: '100%', height: '100%', objectFit: 'cover', filter: 'grayscale(1)', opacity: 0.6 }} />
            <span className="chip" style={{ position: 'absolute', left: 8, top: 8 }}>STILL · DELAYED</span>
            <span className="chip blink" style={{ position: 'absolute', right: 8, top: 8, border: '1px solid var(--rose)', color: 'var(--rose)' }}>NOT RECORDED</span>
          </div>
          <div style={{ height: 8, borderRadius: 4, background: 'repeating-linear-gradient(135deg, var(--line) 0 6px, var(--panel) 6px 12px)' }} />
          <div className="mono" style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--faint)', marginTop: 6 }}>
            <span>21:00</span><span>NO ARCHIVE</span><span>NOW</span>
          </div>
        </div>
        </div>
      </div>

      <p style={{ fontSize: 28, lineHeight: 1.3, margin: '56px 0 0', maxWidth: 840 }}>
        ReTrace doesn’t need the plate. <span style={{ color: 'var(--teal)' }}>It remembers the car.</span>
      </p>
    </section>
  )
}

function Row({ name, value, color = '#93a4b8', tag, tagColor }: { name: string; value: string; color?: string; tag?: string; tagColor?: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, margin: '16px 0 8px', fontSize: 14 }}>
      <span style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        {name}
        {tag && <span className="mono" style={{ fontSize: 10, padding: '2px 6px', borderRadius: 3, border: `1px solid ${tagColor}`, color: tagColor }}>{tag}</span>}
      </span>
      <span className="mono" style={{ color }}>{value}</span>
    </div>
  )
}
