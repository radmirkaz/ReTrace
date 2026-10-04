import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'

interface Variant { tau?: number; precision?: number; recall?: number; f1?: number }
interface ReidScore { rank1: number; rank5: number; mAP: number; queries: number }
interface Metrics {
  reid_cityflow_s01?: Record<string, ReidScore>
  journey_matching_live_feeds?: { variants: Record<string, Variant>; gt_pairs: number }
}
interface Curves { precision: number[]; recall: number[]; map50: number[]; map5095: number[]; epochs: number }

const SLIDES = ['Overview', 'Detection', 'Re-ID data', 'Re-ID model', 'Trustworthy labels', 'Cross-camera', 'Smart search', 'Speed', 'Evaluation', "What's next"]

const COVERAGE = [['USA', 20.9], ['Japan', 13.5], ['Germany', 13.5], ['Italy', 6.4], ['France', 5.9], ['Korea', 5.1], ['China', 4.4], ['UK', 3.8], ['Other', 26.5]] as const

/** Full-screen technical presentation ("Under the hood"): snap-scrolling slides, arrow keys to move. */
export default function Nerd({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [curves, setCurves] = useState<Curves | null>(null)
  const [cur, setCur] = useState(0)
  const deck = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    fetch('/cache/metrics.json').then((r) => (r.ok ? r.json() : null)).then(setMetrics).catch(() => {})
    fetch('/cache/detector_training.json').then((r) => (r.ok ? r.json() : null)).then(setCurves).catch(() => {})
  }, [open])

  // Track the visible slide and support keyboard navigation.
  useEffect(() => {
    const el = deck.current
    if (!open || !el) return
    const io = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) setCur(Number((e.target as HTMLElement).dataset.i)) }), { root: el, threshold: 0.55 })
    el.querySelectorAll('[data-i]').forEach((s) => io.observe(s))
    const go = (i: number) => el.querySelector(`[data-i="${Math.max(0, Math.min(SLIDES.length - 1, i))}"]`)?.scrollIntoView({ behavior: 'smooth' })
    const key = (e: KeyboardEvent) => {
      if (['ArrowDown', 'PageDown', ' ', 'ArrowRight'].includes(e.key)) { e.preventDefault(); go(cur + 1) }
      if (['ArrowUp', 'PageUp', 'ArrowLeft'].includes(e.key)) { e.preventDefault(); go(cur - 1) }
    }
    window.addEventListener('keydown', key)
    return () => { io.disconnect(); window.removeEventListener('keydown', key) }
  }, [open, cur])

  const jump = (i: number) => deck.current?.querySelector(`[data-i="${i}"]`)?.scrollIntoView({ behavior: 'smooth' })
  const reid = metrics?.reid_cityflow_s01
  const match = metrics?.journey_matching_live_feeds

  return (
    <AnimatePresence>
      {open && (
        <motion.div role="dialog" aria-label="Under the hood" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}
          style={{ position: 'fixed', inset: 0, zIndex: 90, background: '#05070b' }}>
          <div className="mono" style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, padding: '14px 24px', background: 'linear-gradient(#05070b, rgba(5,7,11,0))' }}>
            <span className="label" style={{ color: 'var(--violet)' }}>{'</>'} Under the hood · {String(cur + 1).padStart(2, '0')} / {String(SLIDES.length).padStart(2, '0')} · {SLIDES[cur]}</span>
            <button className="btn" onClick={onClose} aria-label="Close" style={{ fontSize: 12 }}>ESC ✕</button>
          </div>
          <nav aria-label="Slides" style={{ position: 'absolute', right: 18, top: '50%', transform: 'translateY(-50%)', zIndex: 2, display: 'flex', flexDirection: 'column', gap: 10 }}>
            {SLIDES.map((s, i) => (
              <button key={s} onClick={() => jump(i)} aria-label={s} title={s} style={{ width: 10, height: 10, padding: 0, borderRadius: '50%', cursor: 'pointer', border: '1px solid var(--violet)', background: i === cur ? 'var(--violet)' : 'transparent' }} />
            ))}
          </nav>

          <div ref={deck} style={{ height: '100%', overflowY: 'auto', scrollSnapType: 'y mandatory' }}>
            <Slide i={0} root={deck} kicker="ReTrace · technical overview" title="Four models, one memory of every car.">
              <Flow lanes={[
                { colour: 'var(--teal)', steps: ['Camera feed', 'YOLOv12n detect', 'ByteTrack', 'Best crops', 'EffNetV2-M class + 2048-D', 'Vehicle index'] },
                { colour: 'var(--violet)', steps: ['Voice', 'Whisper', 'Colour / body / make', 'CLIP text', 'Rank + gate crops'] },
                { colour: 'var(--amber)', steps: ['Selected car', 'v24 fingerprint', 'Body & colour gate', 'Mutual best match', 'Journey across cameras'] },
              ]} />
              <Stats items={[['9,630', 'make / model / generation classes'], ['500k', 'training images'], ['18', 'camera feeds on the map'], ['~60 fps', 'detector on an RTX 3090, 4K']]} />
            </Slide>

            <Slide i={1} root={deck} kicker="1 / Detection & tracking" title="See every vehicle, in every frame.">
              <Cols>
                <div>
                  <Cards items={[
                    ['VehicleCount', 'Street and parking-lot images with car and truck boxes, many angles and lighting.'],
                    ['Udacity Self-Driving', 'Dashcam frames from highways and city streets in varied weather.'],
                    ['Stanford Cars', 'High-resolution vehicles from multiple viewpoints.'],
                  ]} />
                  <p style={P}>~50,000 images, chosen for relevant conditions and perspectives.</p>
                  <ul style={UL}>
                    <li>YOLOv12n, 5 MB: ~60 fps on an RTX 3090 at 4K</li>
                    <li>Confidence 0.05, IoU 0.1 for high recall</li>
                    <li>ByteTrack keeps one ID per vehicle through occlusion</li>
                  </ul>
                </div>
                <div>
                  <Stats items={[['88.8%', 'precision'], ['86.1%', 'recall'], ['92.4%', 'AP50'], ['65.8%', 'mAP50-95']]} compact />
                  {curves && <Chart curves={curves} />}
                </div>
              </Cols>
            </Slide>

            <Slide i={2} root={deck} kicker="2 / Re-identification data" title="500,000 images. 9,630 models and generations.">
              <Stats items={[['900', 'brands'], ['1930s to 2025', 'model years'], ['All origins', 'markets worldwide'], ['14,000+', 'unique vehicles seen']]} />
              <Cols>
                <Cards items={[
                  ['1 · Pretrain (top 2, MCS 2022)', 'CompCars, Google Images and several Kaggle sets · ~5,600 classes'],
                  ['2 · Fine-tune', 'GigaFlexhicle (top 1, MCS 2022) and Google Images for newer models · ~9,600 classes'],
                ]} />
                <div>
                  <div className="label" style={{ color: 'var(--dim)', fontSize: 11, marginBottom: 10 }}>Share of fine-tune images by brand origin</div>
                  {COVERAGE.map(([k, v]) => (
                    <div key={k} style={{ display: 'grid', gridTemplateColumns: '80px 1fr 50px', gap: 10, alignItems: 'center', fontSize: 13, marginBottom: 6 }}>
                      <span>{k}</span>
                      <div style={{ height: 6, background: '#141c28', borderRadius: 3 }}><motion.div initial={{ width: 0 }} whileInView={{ width: `${(v / 27) * 100}%` }} transition={{ duration: 0.8 }} style={{ height: '100%', background: 'var(--violet)', borderRadius: 3 }} /></div>
                      <span className="mono" style={{ color: 'var(--dim)' }}>{v}%</span>
                    </div>
                  ))}
                </div>
              </Cols>
            </Slide>

            <Slide i={3} root={deck} kicker="3 / Re-identification model" title="One network: a name and a fingerprint.">
              <Cols>
                <div>
                  <ul style={UL}>
                    <li>EfficientNetV2-M, 300×300 input, 2048-D embedding</li>
                    <li>Cross-entropy + contrastive + circle loss</li>
                    <li>FP16, 8 epochs, batch 16, LR 6e-5</li>
                    <li>Make/model from the classifier; matching from the v24 embedding</li>
                  </ul>
                  <Stats items={[['91.6%', 'Acc@1, Stanford Cars'], ['92.6%', 'MAP@5'], ['93.7%', 'MAP@5 with class + embedding score']]} compact />
                </div>
                <div>
                  <div className="label" style={{ color: 'var(--dim)', fontSize: 11, marginBottom: 10 }}>Cross-camera re-ID · CityFlowV2 S01 · 36 cars · 5 cameras</div>
                  {reid ? <Table head={['Embedding', 'Rank-1', 'Rank-5', 'mAP']} rows={Object.entries(reid).map(([k, r]) => [k, pct(r.rank1), pct(r.rank5), pct(r.mAP)])} /> : <p style={P}>Benchmark loading…</p>}
                  <p style={P}>Zero-shot: neither checkpoint has seen CityFlow.</p>
                </div>
              </Cols>
            </Slide>

            <Slide i={4} root={deck} kicker="4 / Built at StormHacks" title="Labels you can trust, or none at all.">
              <Cards cols={3} items={[
                ['Best-crop selection', 'Every detection is scored for frame-edge cut-off, overlap, shape, size, confidence and sharpness. The best 3 to 4 crops are averaged.'],
                ['Sink-class control', 'Acura MDX took 24% of labels on weak crops. It is removed, and no class may take more than 3% of all labels.'],
                ['Body-type cross-check', 'A model whose body type contradicts what CLIP sees (an SUV label on a pickup) is replaced from the top 20. 157 labels corrected.'],
                ['Reliability gate', 'A make/model needs crop agreement, a clear margin and enough detail. Rain, snow and night need strong agreement.'],
                ['Honest fallback', 'When unsure, ReTrace says "white pickup truck", not a confident wrong model.'],
                ['Native resolution', 'Tracking and crops run on source footage up to 1920 px; the web only gets a 960 px copy.'],
              ]} />
            </Slide>

            <Slide i={5} root={deck} kicker="5 / Following one car" title="The model proposes. Ground truth grades.">
              <Cols>
                <ul style={UL}>
                  <li>Best match on each camera along the corridor</li>
                  <li>Body group and colour family must agree (brown and black count as one)</li>
                  <li>Mutual best match: the other car must pick this one back</li>
                  <li>Threshold chosen by F1 on labelled cross-camera pairs</li>
                </ul>
                <div>
                  <div className="label" style={{ color: 'var(--dim)', fontSize: 11, marginBottom: 10 }}>Journey matching on live feeds · {match?.gt_pairs ?? '…'} labelled pairs</div>
                  {match ? <Table head={['Method', 'Precision', 'Recall', 'F1']} rows={Object.entries(match.variants).map(([k, v]) => [k, pct(v.precision), pct(v.recall), pct(v.f1)])} /> : <p style={P}>Loading…</p>}
                </div>
              </Cols>
            </Slide>

            <Slide i={6} root={deck} kicker="6 / Smart search" title="Say what you saw. Get the car.">
              <Cards cols={3} items={[
                ['Whisper', 'Transcribes a spoken statement, robust to accents and noise.'],
                ['Query parsing', 'Colour, body type and make are read from the words ("blue pickup, maybe a Toyota").'],
                ['CLIP ViT-B/32', 'Ranks every crop against the description in a shared text-image space, zero-shot.'],
                ['Attribute gate', 'Results must match the colour family and body group; a named make lifts matching cars.'],
                ['Qwen3-4B metadata', 'Every class tagged with region, market and North American availability to steer predictions.'],
                ['Same car, more cameras', 'Any result opens its journey across the city.'],
              ]} />
            </Slide>

            <Slide i={7} root={deck} kicker="7 / Speed estimation" title="Pixels to km/h with a homography.">
              <ol style={{ ...UL, listStyle: 'decimal' }}>
                <li>Four points on the road plane plus a reference distance</li>
                <li>Homography from image to ground coordinates</li>
                <li>Pixel displacement becomes metres per frame</li>
                <li>Frame rate turns it into km/h</li>
                <li>A smoothing filter removes detection jitter</li>
              </ol>
            </Slide>

            <Slide i={8} root={deck} kicker="8 / Evaluation footage" title="Held out, every condition.">
              <Table head={['Dataset', 'Where', 'What', 'Used for']} rows={[
                ['CityFlowV2', 'Iowa, USA', '46 cameras, 880 labelled vehicles', 'Kingsway corridor, re-ID benchmark'],
                ['RoundaboutHD', 'UK', '4 × 4K cameras, 512 vehicles', 'Cambie corridor, 4K feeds'],
                ['AAU RainSnow', 'Denmark', '22 sequences, rain, snow, night', 'Weather feeds'],
                ['MIO-TCD', 'Canada + USA', '137k traffic-camera frames', 'Camera quality and angles'],
              ]} />
              <p style={P}>Map locations are illustrative; feeds come from these public datasets.</p>
            </Slide>

            <Slide i={9} root={deck} kicker="9 / What's next" title="From demo to city deployment.">
              <Cards cols={2} items={[
                ['Pedestrians and cyclists', 'Extend detection to vulnerable road users for hit-and-run cases.'],
                ['Newer vehicles', 'Refresh the class set with 2022+ trucks and EVs, the known gap.'],
                ['Edge processing', 'Detection and crops at the camera; only fingerprints travel to the index.'],
                ['Municipal pilots', 'Plug into existing camera networks and incident reports, one corridor at a time.'],
              ]} />
            </Slide>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

const P: React.CSSProperties = { color: 'var(--muted)', lineHeight: 1.55, fontSize: 15 }
const UL: React.CSSProperties = { color: 'var(--muted)', lineHeight: 1.9, fontSize: 17, paddingLeft: 20, margin: '0 0 18px' }
const pct = (v?: number) => (v === undefined ? '…' : `${(v * 100).toFixed(1)}%`)

function Slide({ i, root, kicker, title, children }: { i: number; root: React.RefObject<HTMLDivElement | null>; kicker: string; title: string; children: React.ReactNode }) {
  return (
    <section data-i={i} style={{ minHeight: '100vh', scrollSnapAlign: 'start', display: 'flex', alignItems: 'center', padding: '80px 0 48px' }}>
      <motion.div className="wrap" initial={{ opacity: 0, y: 34 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ root, amount: 0.3 }} transition={{ duration: 0.55, ease: 'easeOut' }} style={{ maxWidth: 1180 }}>
        <div className="label" style={{ color: 'var(--violet)' }}>{kicker}</div>
        <h2 style={{ fontSize: 'clamp(32px, 4.4vw, 52px)', lineHeight: 1.05, letterSpacing: -1.5, margin: '12px 0 30px' }}>{title}</h2>
        {children}
      </motion.div>
    </section>
  )
}

function Cols({ children }: { children: React.ReactNode }) {
  return <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 40, alignItems: 'start' }}>{children}</div>
}

function Stats({ items, compact }: { items: [string, string][]; compact?: boolean }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(auto-fit, minmax(${compact ? 130 : 200}px, 1fr))`, gap: 1, background: 'var(--line)', border: '1px solid var(--line)', borderRadius: 12, overflow: 'hidden', margin: '24px 0' }}>
      {items.map(([v, l]) => (
        <div key={l} style={{ background: '#070b12', padding: compact ? 16 : 24 }}>
          <div style={{ fontSize: compact ? 28 : 40, fontWeight: 700, letterSpacing: -1 }}>{v}</div>
          <div style={{ color: '#93a4b8', fontSize: 13, marginTop: 4 }}>{l}</div>
        </div>
      ))}
    </div>
  )
}

function Cards({ items, cols = 1 }: { items: [string, string][]; cols?: number }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: cols > 1 ? `repeat(auto-fit, minmax(${cols === 3 ? 260 : 320}px, 1fr))` : '1fr', gap: 14 }}>
      {items.map(([t, d], k) => (
        <motion.div key={t} initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} transition={{ delay: k * 0.06, duration: 0.4 }}
          style={{ padding: 20, borderRadius: 12, border: '1px solid var(--line-2)', background: 'var(--panel)' }}>
          <div style={{ fontWeight: 700, fontSize: 17, marginBottom: 8, color: 'var(--text)' }}>{t}</div>
          <div style={{ color: 'var(--muted)', lineHeight: 1.55, fontSize: 14 }}>{d}</div>
        </motion.div>
      ))}
    </div>
  )
}

function Flow({ lanes }: { lanes: { colour: string; steps: string[] }[] }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {lanes.map((lane, li) => (
        <div key={li} style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8 }}>
          {lane.steps.map((s, i) => (
            <motion.span key={s} initial={{ opacity: 0, x: -10 }} whileInView={{ opacity: 1, x: 0 }} transition={{ delay: li * 0.15 + i * 0.07 }} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="mono" style={{ fontSize: 13, padding: '8px 12px', borderRadius: 6, border: `1px solid ${lane.colour}`, color: lane.colour }}>{s}</span>
              {i < lane.steps.length - 1 && <span style={{ color: 'var(--faint)' }}>→</span>}
            </motion.span>
          ))}
        </div>
      ))}
    </div>
  )
}

function Chart({ curves }: { curves: Curves }) {
  const W = 520, H = 220, pad = 30
  const series: [string, number[], string][] = [['AP50', curves.map50, '#5eead4'], ['Precision', curves.precision, '#fbbf24'], ['Recall', curves.recall, '#a78bfa'], ['mAP50-95', curves.map5095, '#93a4b8']]
  const lo = 0.5, hi = 0.95
  const pt = (v: number, i: number, n: number) => `${pad + (i / (n - 1)) * (W - pad - 10)},${H - pad - ((v - lo) / (hi - lo)) * (H - pad - 10)}`
  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }} role="img" aria-label="Detector validation metrics over 50 epochs">
        {[0.6, 0.7, 0.8, 0.9].map((g) => (
          <g key={g}>
            <line x1={pad} x2={W - 10} y1={H - pad - ((g - lo) / (hi - lo)) * (H - pad - 10)} y2={H - pad - ((g - lo) / (hi - lo)) * (H - pad - 10)} stroke="#141c28" />
            <text x={4} y={H - pad - ((g - lo) / (hi - lo)) * (H - pad - 10) + 4} fill="#5b6b80" fontSize="10" fontFamily="JetBrains Mono, monospace">{g.toFixed(1)}</text>
          </g>
        ))}
        {series.map(([name, s, c]) => (
          <motion.polyline key={name} points={s.map((v, i) => pt(v, i, s.length)).join(' ')} fill="none" stroke={c} strokeWidth="2" initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} transition={{ duration: 1.2 }} />
        ))}
        <text x={W - 10} y={H - 8} textAnchor="end" fill="#5b6b80" fontSize="10" fontFamily="JetBrains Mono, monospace">epoch {curves.epochs}</text>
      </svg>
      <figcaption className="mono" style={{ display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 11, color: 'var(--dim)', marginTop: 6 }}>
        {series.map(([name, , c]) => <span key={name}><span style={{ color: c }}>■</span> {name}</span>)}
      </figcaption>
    </figure>
  )
}

function Table({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <div style={{ overflowX: 'auto', border: '1px solid var(--line)', borderRadius: 10 }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
        <thead><tr>{head.map((h) => <th key={h} className="mono" style={{ textAlign: 'left', padding: '10px 14px', fontSize: 11, color: 'var(--dim)', borderBottom: '1px solid var(--line)', fontWeight: 400 }}>{h}</th>)}</tr></thead>
        <tbody>{rows.map((r, k) => <tr key={k}>{r.map((c, i) => <td key={i} style={{ padding: '10px 14px', borderBottom: '1px solid #111824', color: i ? 'var(--muted)' : 'var(--text)' }}>{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  )
}
