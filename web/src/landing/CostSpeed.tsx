import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'

interface Bench { gpu: string; detect_track_fps_1080p: number; reid_crops_per_s: number; match_ms_vs_1M_vehicles: number }

// Used RTX 3090 market price, Oct 2026 (USD), mid-range of eBay/market trackers.
const GPU_PRICE = 1000
const SEARCH_MS = 36 // measured end to end: text query -> ranked crops, over the network

/** Cost & speed: measured throughput on one RTX 3090 and what a city deployment would need. */
export default function CostSpeed() {
  const [b, setB] = useState<Bench | null>(null)
  const [cams, setCams] = useState(221)
  const [fps, setFps] = useState(10)
  useEffect(() => { fetch('/cache/benchmark.json').then((r) => (r.ok ? r.json() : null)).then(setB).catch(() => {}) }, [])

  const gpuFps = b?.detect_track_fps_1080p ?? 68
  const perGpu = Math.max(1, Math.floor(gpuFps / fps))
  const gpus = Math.ceil(cams / perGpu)
  const cost = gpus * GPU_PRICE
  const hourMinutes = Math.round((3600 * fps) / gpuFps / 60)

  const speed: [string, string][] = [
    [`${Math.round(gpuFps)} fps`, 'detection + tracking at 1080p on one GPU'],
    [`${Math.round(b?.reid_crops_per_s ?? 427)}/s`, 'vehicles identified (make/model + fingerprint)'],
    [`${(b?.match_ms_vs_1M_vehicles ?? 9.9).toFixed(0)} ms`, 'to match one car against 1 million vehicles'],
    [`${SEARCH_MS} ms`, 'from a typed description to ranked results'],
  ]

  return (
    <section className="wrap" style={{ padding: '120px 28px 40px' }}>
      <div className="label" style={{ color: 'var(--teal)' }}>04 / Cost and speed</div>
      <h2 style={{ fontSize: 'clamp(36px, 5vw, 56px)', lineHeight: 1.04, letterSpacing: -2, margin: '14px 0 16px', maxWidth: 900 }}>
        Hours of footage in minutes. <span style={{ color: 'var(--teal)' }}>A city on a handful of GPUs.</span>
      </h2>
      <p style={{ color: 'var(--muted)', fontSize: 18, lineHeight: 1.55, maxWidth: 760, margin: '0 0 40px' }}>
        Measured on a single {b?.gpu ?? 'RTX 3090'}, one stream at a time with no batching. No new cameras: ReTrace runs on the feeds a city already has.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 1, background: 'var(--line)', border: '1px solid var(--line)', borderRadius: 12, overflow: 'hidden' }}>
        {speed.map(([v, l], i) => (
          <motion.div key={l} initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.08 }} style={{ background: '#070b12', padding: 26 }}>
            <div style={{ fontSize: 44, fontWeight: 700, letterSpacing: -1.5 }}>{v}</div>
            <div style={{ color: '#93a4b8', marginTop: 6, lineHeight: 1.4 }}>{l}</div>
          </motion.div>
        ))}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, marginTop: 24 }}>
        <div style={{ flex: '1 1 360px', padding: 26, borderRadius: 12, border: '1px solid var(--line)', background: 'var(--panel)' }}>
          <div className="label" style={{ color: 'var(--dim)', fontSize: 11, marginBottom: 18 }}>Deployment calculator</div>
          <Slider label="Cameras" value={cams} min={10} max={1000} step={1} onChange={setCams} note={cams === 221 ? 'Vancouver today' : undefined} />
          <Slider label="Frames per second analysed" value={fps} min={2} max={30} step={1} onChange={setFps} />
          <p className="mono" style={{ fontSize: 11, color: 'var(--faint)', lineHeight: 1.6, margin: '14px 0 0' }}>
            {Math.round(gpuFps)} fps per GPU ÷ {fps} fps per camera = {perGpu} cameras per GPU. Used RTX 3090 ≈ US${GPU_PRICE.toLocaleString('en-US')}.
          </p>
        </div>
        <div style={{ flex: '1 1 360px', display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 12 }}>
          {[
            [`${gpus}`, `GPUs for ${cams} cameras, live`],
            [`US$${cost.toLocaleString('en-US')}`, 'one-time GPU hardware'],
            [`US$${Math.round(cost / cams)}`, 'per camera, one time'],
            [`${hourMinutes} min`, 'to search one hour of footage from one camera'],
          ].map(([v, l]) => (
            <div key={l} style={{ padding: 20, borderRadius: 12, border: '1px solid var(--teal)', background: 'rgba(94,234,212,.05)' }}>
              <div style={{ fontSize: 32, fontWeight: 700, letterSpacing: -1, color: 'var(--teal)' }}>{v}</div>
              <div style={{ color: 'var(--muted)', fontSize: 14, marginTop: 6, lineHeight: 1.4 }}>{l}</div>
            </div>
          ))}
        </div>
      </div>
      <p className="mono" style={{ fontSize: 11, color: 'var(--faint)', marginTop: 14 }}>
        Throughput measured in this repo (pipeline/benchmark.py). Batching streams and smaller edge GPUs would lower cost further.
      </p>
    </section>
  )
}

function Slider({ label, value, min, max, step, onChange, note }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; note?: string }) {
  const id = label.replace(/\W+/g, '-').toLowerCase()
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <label htmlFor={id} style={{ fontSize: 15 }}>{label}</label>
        <span className="mono" style={{ fontSize: 18, color: 'var(--teal)' }}>{value}{note && <span style={{ fontSize: 11, color: 'var(--dim)', marginLeft: 8 }}>{note}</span>}</span>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} style={{ width: '100%', accentColor: '#5eead4' }} />
    </div>
  )
}
