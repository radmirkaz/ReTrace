import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useState } from 'react'

interface Metrics { [k: string]: unknown }

const MODELS = [
  ['Detector', 'YOLOv12n, fine-tuned', '~45k traffic images (Roboflow VehicleCount, vehicles-q0x2v, Stanford Cars, Udacity)', 'P 88% · R 86% · AP50 92%'],
  ['Tracker', 'ByteTrack', 'No training; associates detections frame to frame', 'Stable IDs through occlusion'],
  ['Re-ID / classifier', 'EfficientNetV2-M', '407k images · 9,630 make/model/generation classes · CE + contrastive + circle loss', 'Acc@1 91.6% · MAP@5 92.6% · 2048-D embedding'],
  ['Text → image', 'CLIP ViT-B/32 (LAION-2B)', 'Zero-shot, no fine-tuning', 'Ranks every crop against a description'],
  ['Speech', 'Whisper base', 'Zero-shot', 'Witness statement → search query'],
  ['Priors', 'Qwen3-4B (offline)', 'Tagged every class with region, market, sold in US/Canada', 'Re-weights rare/foreign models'],
]

const DATASETS = [
  ['CityFlowV2', 'USA · Iowa', '46 cameras, 880 IDs', 'Cross-camera tracking'],
  ['RoundaboutHD', 'UK', '4 × 4K cameras, 512 IDs', 'Cross-camera re-ID'],
  ['AAU RainSnow', 'Denmark', '22 sequences, rain/snow/night', 'Weather robustness'],
  ['MIO-TCD', 'Canada + USA', '137k traffic-camera frames', 'Camera quality & angle'],
]

const COVERAGE = [['USA', 20.9], ['Japan', 13.5], ['Germany', 13.5], ['Italy', 6.4], ['France', 5.9], ['Korea', 5.1], ['China', 4.4], ['UK', 3.8], ['Other', 26.5]] as const

const PIPE = ['Camera feed', 'YOLOv12n detect', 'ByteTrack', 'Best crop per track', 'EffNetV2-M → class + 2048-D', 'Index']
const QUERY = ['Voice', 'Whisper', 'Text', 'CLIP text encoder', 'Cosine rank vs. crops']

export default function Nerd({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  useEffect(() => { if (open) fetch('/cache/metrics.json').then((r) => (r.ok ? r.json() : null)).then(setMetrics).catch(() => setMetrics(null)) }, [open])

  return (
    <AnimatePresence>
      {open && (
        <motion.aside
          role="dialog"
          aria-label="Under the hood"
          initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }} transition={{ type: 'spring', damping: 30, stiffness: 260 }}
          style={{ position: 'fixed', top: 0, right: 0, bottom: 0, width: 'min(760px, 100vw)', zIndex: 80, background: '#070a10', borderLeft: '1px solid var(--violet)', overflowY: 'auto', padding: '28px 28px 60px', boxShadow: '-30px 0 80px rgba(0,0,0,.6)' }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <div className="label" style={{ color: 'var(--violet)' }}>{'</>'} Under the hood</div>
            <button className="btn mono" onClick={onClose} aria-label="Close" style={{ fontSize: 12 }}>ESC ✕</button>
          </div>
          <h2 style={{ fontSize: 34, letterSpacing: -1, margin: '8px 0 26px' }}>Models, data and numbers.</h2>

          <H>Pipeline</H>
          <Flow steps={PIPE} color="var(--teal)" />
          <Flow steps={QUERY} color="var(--violet)" />

          <H>Models</H>
          <Table head={['Stage', 'Model', 'Trained on', 'Result']} rows={MODELS} />

          <H>Held-out evaluation footage</H>
          <Table head={['Dataset', 'Country', 'Size', 'Used for']} rows={DATASETS} />

          <H>Class coverage: share of 407k training images by brand origin</H>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {COVERAGE.map(([k, v]) => (
              <div key={k} style={{ display: 'grid', gridTemplateColumns: '90px 1fr 50px', gap: 10, alignItems: 'center', fontSize: 13 }}>
                <span>{k}</span>
                <div style={{ height: 6, background: '#141c28', borderRadius: 3 }}><div style={{ height: '100%', width: `${(v / 27) * 100}%`, background: 'var(--violet)', borderRadius: 3 }} /></div>
                <span className="mono" style={{ color: 'var(--dim)' }}>{v}%</span>
              </div>
            ))}
          </div>
          <p style={{ color: 'var(--dim)', fontSize: 13, lineHeight: 1.5 }}>72% of images are models sold in the US or Canada. Known gap: 2022+ US trucks/EVs and Chinese brands are thin.</p>

          <H>Robustness & hardware</H>
          {metrics ? (
            <pre className="mono" style={{ fontSize: 12, whiteSpace: 'pre-wrap', color: 'var(--muted)', background: 'var(--panel)', padding: 16, borderRadius: 8 }}>{JSON.stringify(metrics, null, 2)}</pre>
          ) : (
            <p style={{ color: 'var(--dim)', fontSize: 14 }}>Per-condition AP50 / re-ID mAP and RTX 3090 throughput are being measured by <code>pipeline/eval_robustness.py</code>; results appear here automatically.</p>
          )}
        </motion.aside>
      )}
    </AnimatePresence>
  )
}

function H({ children }: { children: React.ReactNode }) {
  return <div className="label" style={{ color: 'var(--dim)', fontSize: 11, margin: '30px 0 12px' }}>{children}</div>
}

function Flow({ steps, color }: { steps: string[]; color: string }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6, marginBottom: 10 }}>
      {steps.map((s, i) => (
        <span key={s} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span className="mono" style={{ fontSize: 12, padding: '6px 10px', borderRadius: 4, border: `1px solid ${color}`, color }}>{s}</span>
          {i < steps.length - 1 && <span style={{ color: 'var(--faint)' }}>→</span>}
        </span>
      ))}
    </div>
  )
}

function Table({ head, rows }: { head: string[]; rows: readonly (readonly string[])[] }) {
  return (
    <div style={{ overflowX: 'auto', border: '1px solid var(--line)', borderRadius: 8 }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead><tr>{head.map((h) => <th key={h} className="mono" style={{ textAlign: 'left', padding: '10px 12px', fontSize: 11, color: 'var(--dim)', borderBottom: '1px solid var(--line)', fontWeight: 400 }}>{h}</th>)}</tr></thead>
        <tbody>{rows.map((r) => <tr key={r[0]}>{r.map((c, i) => <td key={i} style={{ padding: '10px 12px', borderBottom: '1px solid #111824', verticalAlign: 'top', color: i ? 'var(--muted)' : 'var(--text)' }}>{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  )
}
