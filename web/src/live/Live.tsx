import { useEffect, useMemo, useRef, useState } from 'react'
import Lightbox from '../components/Lightbox'
import TrackedVideo from '../components/TrackedVideo'
import { asset, getMode, loadCameras, loadTracks, loadVehicles, searchText, transcribe, vehicleTitle, type Camera, type Det, type SearchHit, type Tracks, type Vehicle } from '../data'
import FollowCam from './FollowCam'
import VancouverMap from './VancouverMap'

const FULL = 'calc(100vh - 73px)'

/** Only re-render the dashboard when the set of vehicles in frame changes, not on every video frame. */
const sameIds = (a: Det[], b: Det[]) => a.length === b.length && a.every((d, i) => d[0] === b[i][0])

export default function Live() {
  const [cameras, setCameras] = useState<Camera[]>([])
  const [vehicles, setVehicles] = useState<Record<string, Vehicle>>({})
  const [camId, setCamId] = useState<string>()
  const [startAt, setStartAt] = useState<number>()
  const [tracks, setTracks] = useState<Tracks | null>(null)
  const [dets, setDets] = useState<Det[]>([])
  const [gid, setGid] = useState<string>()
  const [hits, setHits] = useState<SearchHit[] | null>(null)
  const [zoom, setZoom] = useState<{ src: string; caption: string } | null>(null)

  useEffect(() => { Promise.all([loadCameras(), loadVehicles()]).then(([c, v]) => { setCameras(c); setVehicles(v) }) }, [])
  const cam = cameras.find((c) => c.id === camId)
  useEffect(() => { setTracks(null); if (cam) loadTracks(cam).then(setTracks) }, [cam])

  const vehicle = gid ? vehicles[gid] : undefined
  const selectedTrack = gid && cam && gid.startsWith(cam.id + ':') ? Number(gid.split(':')[1]) : undefined
  const journey = vehicle?.journey && vehicle.journey.length > 1 ? vehicle.journey : []
  const trail = journey.map((s) => s.cam)
  const focus = trail.length ? trail : cam ? [cam.id] : []
  const camName = (id: string) => cameras.find((c) => c.id === id)?.name ?? id

  /** Open a camera; with a vehicle, start the feed just before that vehicle is in view. */
  const open = (cameraId: string, g?: string) => {
    const t = g ? vehicles[g]?.sightings[0]?.t : undefined
    setStartAt(t !== undefined ? Math.max(0, t - 1) : undefined)
    setCamId(cameraId)
    setGid(g)
  }
  const close = () => { setCamId(undefined); setGid(undefined) }

  return (
    <main style={{ display: 'flex', flexWrap: 'wrap', minHeight: FULL }}>
      <div style={{ position: 'relative', flex: cam ? '0 1 40%' : '1 1 560px', minWidth: cam ? 320 : 0, height: FULL, minHeight: 520, borderRight: '1px solid var(--line)' }}>
        <VancouverMap cameras={cameras} selected={camId} highlight={trail} focus={focus} onSelect={(id) => open(id)} />
        <SearchBar onResults={setHits} compact={!!cam} />
        {hits && <Results hits={hits} vehicles={vehicles} camName={camName} onPick={(h) => { open(h.cam, h.gid); setHits(null) }} onClose={() => setHits(null)} />}
        <div className="mono" style={{ position: 'absolute', left: 16, bottom: 14, fontSize: 11, color: 'var(--dim)' }}>
          <span className="blink" style={{ display: 'inline-block', width: 7, height: 7, borderRadius: '50%', background: 'var(--rose)', marginRight: 6 }} />
          {cameras.length} CAMERAS ONLINE · {Object.keys(vehicles).length} VEHICLES INDEXED
        </div>
      </div>

      {!cam ? (
        <aside style={{ flex: '0 1 420px', minWidth: 300, height: FULL, overflowY: 'auto', background: 'var(--panel)', padding: 28 }}>
          <div className="label" style={{ color: 'var(--teal)' }}>City of Vancouver</div>
          <h2 style={{ fontSize: 30, letterSpacing: -1, margin: '10px 0' }}>Pick a camera.</h2>
          <p style={{ color: 'var(--muted)', lineHeight: 1.55 }}>Every feed runs detection, tracking and re-identification. Click a vehicle to follow it, see what it is and where else it was seen, or search by description or voice.</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 18 }}>
            {cameras.map((c) => (
              <button key={c.id} className="btn" onClick={() => open(c.id)} style={{ justifyContent: 'space-between' }}>
                <span>{c.name}</span><span className="mono" style={{ fontSize: 11, color: 'var(--dim)' }}>{c.condition.toUpperCase()}</span>
              </button>
            ))}
          </div>
        </aside>
      ) : (
        <section style={{ flex: '1 1 640px', minWidth: 0, height: FULL, overflowY: 'auto', background: 'var(--panel)', padding: '18px 22px 40px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
            <div>
              <div className="label" style={{ color: 'var(--teal)', fontSize: 11 }}>{cam.view} · {cam.condition} · {cam.res}</div>
              <h2 style={{ fontSize: 28, letterSpacing: -0.5, margin: '6px 0 2px' }}>{cam.name}</h2>
              <div className="mono" style={{ fontSize: 10, color: 'var(--faint)' }}>Feed source: {cam.source}</div>
            </div>
            <button className="btn" aria-label="Close camera" onClick={close}>✕ Map</button>
          </div>

          {vehicle && <TrackingBar v={vehicle} current={cam.id} camName={camName} onStep={open} onStop={() => setGid(undefined)} />}

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginTop: 14, alignItems: 'flex-start' }}>
            <div style={{ flex: '1.7 1 420px', minWidth: 0, borderRadius: 8, overflow: 'hidden', border: '1px solid var(--line)' }}>
              <TrackedVideo
                key={cam.id + (startAt ?? '')}
                src={asset(cam.clip)}
                poster={asset(cam.poster)}
                tracks={tracks}
                startAt={startAt}
                fullscreen
                highlight={selectedTrack}
                othersOpacity={selectedTrack !== undefined ? 0.45 : 1}
                labelFor={(tid) => { const v = vehicles[`${cam.id}:${tid}`]; return v ? vehicleTitle(v) : undefined }}
                onFrame={(_, d) => setDets((prev) => (sameIds(prev, d) ? prev : d))}
                onPick={(tid) => setGid(`${cam.id}:${tid}`)}
              >
                <span className="chip" style={{ position: 'absolute', left: 8, top: 8, display: 'flex', gap: 6, alignItems: 'center', color: 'var(--text)' }}>
                  <span className="blink" style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--rose)' }} />LIVE · {dets.length} VEHICLES
                </span>
              </TrackedVideo>
            </div>
            <div style={{ flex: '1 1 260px', minWidth: 240 }}>
              {tracks && selectedTrack !== undefined ? (
                <FollowCam key={gid} src={asset(cam.clip)} tracks={tracks} track={selectedTrack} />
              ) : (
                <div style={{ aspectRatio: '4 / 3', borderRadius: 8, border: '1px dashed var(--line-2)', display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: 20, color: 'var(--dim)', fontSize: 14 }}>
                  Click a vehicle in the feed<br />to follow it here.
                </div>
              )}
            </div>
          </div>

          {vehicle && <VehicleDetail v={vehicle} camName={camName} onJump={open} onZoom={(src, caption) => setZoom({ src, caption })} onBack={() => setGid(undefined)} />}
          <Strip cam={cam} dets={dets} tracks={tracks} vehicles={vehicles} selected={gid} onPick={setGid} />
        </section>
      )}
      {zoom && <Lightbox src={zoom.src} caption={zoom.caption} onClose={() => setZoom(null)} />}
    </main>
  )
}

function VehicleDetail({ v, camName, onJump, onZoom, onBack }: { v: Vehicle; camName: (id: string) => string; onJump: (cam: string, gid: string) => void; onZoom: (src: string, caption: string) => void; onBack: () => void }) {
  const [guesses, setGuesses] = useState(false)
  const uncertain = v.reliable === false
  const title = vehicleTitle(v)
  return (
    <div style={{ marginTop: 18, padding: 18, borderRadius: 10, border: '1px solid var(--line)', background: 'var(--panel-2)' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 18 }}>
        <button onClick={() => onZoom(asset(v.crop), title)} className="zoomable" aria-label="View crop full screen" style={{ padding: 0, border: '2px solid var(--amber)', borderRadius: 6, overflow: 'hidden', background: 'none', flex: '0 0 220px' }}>
          <img src={asset(v.crop)} alt={title} style={{ width: 220, aspectRatio: '4 / 3', objectFit: 'cover' }} />
        </button>
        <div style={{ flex: '1 1 260px', minWidth: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
            <div>
              <div style={{ fontWeight: 700, fontSize: 22 }}>{title}</div>
              <div className="mono" style={{ fontSize: 11, color: 'var(--dim)', marginTop: 4 }}>#{v.gid.split(':')[1]} · {v.color}{v.body ? ` · ${v.body}` : ''}</div>
            </div>
            <button className="btn mono" onClick={onBack} aria-label="Deselect vehicle" style={{ fontSize: 12, minHeight: 36, padding: '6px 12px', alignSelf: 'flex-start' }}>✕</button>
          </div>
          {uncertain && (
            <div style={{ marginTop: 12 }}>
              <span className="chip" style={{ border: '1px solid var(--line-2)', color: 'var(--muted)' }}>MAKE / MODEL UNCERTAIN</span>
              <p style={{ fontSize: 13, color: 'var(--dim)', lineHeight: 1.5, margin: '8px 0' }}>
                Crops disagree or lack detail{v.evidence ? ` (agreement ${Math.round(v.evidence.consistency * 100)}%, ${v.evidence.min_side}px)` : ''}. Matching still works on appearance.
              </p>
              <button className="btn mono" onClick={() => setGuesses((g) => !g)} style={{ fontSize: 11, minHeight: 32, padding: '4px 10px' }}>{guesses ? 'Hide' : 'Show'} raw guesses</button>
            </div>
          )}
          {(!uncertain || guesses) && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginTop: 12, opacity: uncertain ? 0.55 : 1 }}>
              {v.top5.map((r, i) => (
                <div key={r.name}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}><span>{r.name}</span><span className="mono" style={{ color: i ? 'var(--faint)' : 'var(--amber)' }}>{r.p.toFixed(2)}</span></div>
                  <div style={{ height: 4, marginTop: 3, background: '#141c28', borderRadius: 2 }}><div style={{ height: '100%', width: `${r.p * 100}%`, background: i ? 'var(--faint)' : 'var(--amber)', borderRadius: 2 }} /></div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <LookAlikes items={v.similar ?? v.sightings.slice(1).filter((x) => !x.verified)} camName={camName} onJump={onJump} />
    </div>
  )
}

function TrackingBar({ v, current, camName, onStep, onStop }: { v: Vehicle; current: string; camName: (id: string) => string; onStep: (cam: string, gid: string) => void; onStop: () => void }) {
  const steps = v.journey && v.journey.length > 1 ? v.journey : null
  const i = steps ? steps.findIndex((s) => s.cam === current) : -1
  const go = (k: number) => { const s = steps?.[k]; if (s) onStep(s.cam, `${s.cam}:${s.track}`) }
  const graded = steps ? steps.filter((s) => !s.self && s.correct !== undefined && s.correct !== null).length : 0
  const right = steps ? steps.filter((s) => s.correct === true).length : 0
  return (
    <div style={{ marginTop: 14, padding: '12px 14px', borderRadius: 10, border: '1px solid var(--amber)', background: 'rgba(251,191,36,.06)' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 10 }}>
        <span className="label" style={{ color: 'var(--amber)', fontSize: 11 }}>Tracking</span>
        <img src={asset(v.crop)} alt="" style={{ width: 44, height: 33, objectFit: 'cover', borderRadius: 4, border: '1px solid var(--amber)' }} />
        <strong style={{ fontSize: 16 }}>{vehicleTitle(v)}</strong>
        <span className="mono" style={{ fontSize: 11, color: 'var(--dim)' }}>
          {steps ? `model found it on ${steps.length} cameras · step ${i + 1} of ${steps.length}${graded ? ` · ${right}/${graded} confirmed by ground truth` : ''}` : 'no match on other cameras'}
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          {steps && <button className="btn" disabled={i <= 0} onClick={() => go(i - 1)} style={{ minHeight: 34, padding: '4px 12px', opacity: i <= 0 ? 0.4 : 1 }}>‹ Prev</button>}
          {steps && <button className="btn" disabled={i >= steps.length - 1} onClick={() => go(i + 1)} style={{ minHeight: 34, padding: '4px 12px', opacity: i >= steps.length - 1 ? 0.4 : 1 }}>Next ›</button>}
          <button className="btn" onClick={onStop} aria-label="Stop tracking" style={{ minHeight: 34, padding: '4px 10px' }}>✕</button>
        </div>
      </div>
      {steps && (
        <ol style={{ listStyle: 'none', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6, margin: '12px 0 0', padding: 0 }}>
          {steps.map((s, k) => {
            const here = k === i
            return (
              <li key={s.cam + s.track} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <button onClick={() => go(k)} aria-current={here ? 'step' : undefined} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px 6px 6px', borderRadius: 8, cursor: 'pointer', background: here ? 'var(--amber)' : 'var(--panel)', color: here ? 'var(--amber-ink)' : 'var(--text)', border: `1px solid ${here ? 'var(--amber)' : 'var(--line-2)'}` }}>
                  <span className="mono" style={{ width: 22, height: 22, borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, fontWeight: 700, background: here ? 'var(--amber-ink)' : 'transparent', color: here ? 'var(--amber)' : 'var(--amber)', border: here ? 'none' : '1px solid var(--amber)' }}>{k + 1}</span>
                  <img src={asset(s.crop)} alt="" style={{ width: 40, height: 30, objectFit: 'cover', borderRadius: 3 }} />
                  <span style={{ fontSize: 13, fontWeight: here ? 700 : 500, whiteSpace: 'nowrap' }}>{camName(s.cam)}</span>
                  {here && <span className="mono" style={{ fontSize: 10 }}>YOU ARE HERE</span>}
                  {!s.self && s.correct === true && <span title="Confirmed by ground truth" style={{ color: here ? 'var(--amber-ink)' : 'var(--teal)' }}>✓</span>}
                  {!s.self && s.correct === false && <span title="Wrong match per ground truth" style={{ color: here ? 'var(--amber-ink)' : 'var(--rose)' }}>✗</span>}
                </button>
                {k < steps.length - 1 && <span aria-hidden style={{ color: 'var(--amber)' }}>→</span>}
              </li>
            )
          })}
        </ol>
      )}
    </div>
  )
}

function LookAlikes({ items, camName, onJump }: { items: Vehicle['sightings']; camName: (id: string) => string; onJump: (cam: string, gid: string) => void }) {
  const [open, setOpen] = useState(false)
  if (!items.length) return null
  return (
    <div style={{ marginTop: 18 }}>
      <button className="btn mono" onClick={() => setOpen((o) => !o)} aria-expanded={open} style={{ fontSize: 11, minHeight: 34, padding: '4px 12px' }}>
        {open ? '▾' : '▸'} Similar-looking vehicles on other cameras · not confirmed the same car ({items.length})
      </button>
      {open && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 10, marginTop: 10 }}>
          {items.map((s) => (
            <button key={s.cam + s.track} onClick={() => onJump(s.cam, `${s.cam}:${s.track}`)} style={{ padding: 0, textAlign: 'left', cursor: 'pointer', borderRadius: 8, overflow: 'hidden', border: '1px solid var(--line-2)', background: 'var(--panel)', opacity: 0.85 }}>
              <img src={asset(s.crop)} alt="" style={{ width: '100%', aspectRatio: '4 / 3', objectFit: 'cover' }} />
              <div style={{ padding: '6px 9px' }}>
                <div style={{ fontSize: 12 }}>{camName(s.cam)}</div>
                <div className="mono" style={{ fontSize: 10, color: 'var(--dim)', marginTop: 3 }}>look-alike · sim {s.sim.toFixed(2)}</div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function Strip({ cam, dets, tracks, vehicles, selected, onPick }: { cam: Camera; dets: Det[]; tracks: Tracks | null; vehicles: Record<string, Vehicle>; selected?: string; onPick: (g: string) => void }) {
  const all = useMemo(() => {
    const ids = new Set<number>()
    tracks?.frames.forEach((f) => f.forEach((d) => ids.add(d[0])))
    return [...ids].filter((tid) => vehicles[`${cam.id}:${tid}`])
  }, [tracks, vehicles, cam.id])
  const visible = new Set(dets.map((d) => d[0]))
  return (
    <div style={{ marginTop: 20 }}>
      <div className="label" style={{ color: 'var(--dim)', fontSize: 11, marginBottom: 10 }}>Identified on this camera · {all.length} · teal = in frame now</div>
      <div style={{ display: 'flex', gap: 8, overflowX: 'auto', paddingBottom: 6 }}>
        {all.map((tid) => {
          const v = vehicles[`${cam.id}:${tid}`]
          const on = selected === v.gid
          return (
            <button key={tid} onClick={() => onPick(v.gid)} style={{ flex: '0 0 132px', padding: 0, borderRadius: 6, overflow: 'hidden', cursor: 'pointer', textAlign: 'left', background: 'var(--panel-2)', border: `${on ? 2 : 1}px solid ${on ? 'var(--amber)' : visible.has(tid) ? 'var(--teal)' : 'var(--line)'}` }}>
              <img src={asset(v.crop)} alt="" style={{ width: '100%', aspectRatio: '4 / 3', objectFit: 'cover' }} />
              <div className="mono" style={{ padding: '5px 7px', fontSize: 10, lineHeight: 1.4 }}>
                #{tid}<br /><span style={{ color: v.reliable === false ? 'var(--dim)' : 'var(--muted)' }}>{vehicleTitle(v)}</span>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function SearchBar({ onResults, compact }: { onResults: (h: SearchHit[]) => void; compact: boolean }) {
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState<'' | 'search' | 'listen'>('')
  const [note, setNote] = useState('')
  const rec = useRef<MediaRecorder | null>(null)

  const run = async (text: string) => {
    if (!text.trim()) return
    setBusy('search')
    try { onResults(await searchText(text)) } finally { setBusy('') }
  }

  const voice = async () => {
    if (busy === 'listen') { rec.current?.stop(); return }
    setNote('')
    if (getMode() !== 'live') {
      const SR = (window as unknown as { webkitSpeechRecognition?: new () => { lang: string; onresult: (e: { results: { 0: { 0: { transcript: string } } } }) => void; onend: () => void; start: () => void } }).webkitSpeechRecognition
      if (!SR) { setNote('Voice search is unavailable in this browser.'); return }
      const r = new SR()
      r.lang = 'en-US'
      r.onresult = (e) => { const t = e.results[0][0].transcript; setQ(t); run(t) }
      r.onend = () => setBusy('')
      setBusy('listen')
      r.start()
      return
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const chunks: Blob[] = []
    const mr = new MediaRecorder(stream)
    mr.ondataavailable = (e) => chunks.push(e.data)
    mr.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop())
      setBusy('search')
      try { const t = await transcribe(new Blob(chunks, { type: 'audio/webm' })); setQ(t); await run(t) } catch (e) { setNote(String(e)) } finally { setBusy('') }
    }
    rec.current = mr
    mr.start()
    setBusy('listen')
  }

  return (
    <form onSubmit={(e) => { e.preventDefault(); run(q) }} style={{ position: 'absolute', top: 16, left: 16, right: 16, maxWidth: 620, display: 'flex', gap: 8, alignItems: 'center', padding: 8, borderRadius: 10, background: 'rgba(10,15,22,.92)', border: '1px solid var(--line-2)', backdropFilter: 'blur(8px)' }}>
      {!compact && <label htmlFor="q" className="mono" style={{ fontSize: 11, color: 'var(--dim)', paddingLeft: 6 }}>FIND</label>}
      <input id="q" aria-label="Describe a vehicle" value={q} onChange={(e) => setQ(e.target.value)} placeholder={compact ? 'Describe a vehicle…' : 'white pickup, dark SUV with roof rails…'} style={{ flex: 1, minWidth: 0, background: 'transparent', border: 'none', outline: 'none', color: 'var(--text)', fontSize: 15, fontFamily: 'var(--sans)', minHeight: 40 }} />
      <button type="button" className="btn" onClick={voice} aria-label={busy === 'listen' ? 'Stop recording' : 'Search by voice'} style={{ minHeight: 40, padding: '8px 12px', borderColor: busy === 'listen' ? 'var(--violet)' : undefined, color: busy === 'listen' ? 'var(--violet)' : undefined }}>
        {busy === 'listen' ? '■' : '🎙'}
      </button>
      <button type="submit" className="btn btn-primary" style={{ minHeight: 40, padding: '8px 14px' }} disabled={busy === 'search'}>{busy === 'search' ? '…' : 'Search'}</button>
      {note && <span className="mono" style={{ position: 'absolute', top: '100%', left: 8, marginTop: 6, fontSize: 11, color: 'var(--rose)' }}>{note}</span>}
    </form>
  )
}

function Results({ hits, vehicles, camName, onPick, onClose }: { hits: SearchHit[]; vehicles: Record<string, Vehicle>; camName: (id: string) => string; onPick: (h: SearchHit) => void; onClose: () => void }) {
  return (
    <div style={{ position: 'absolute', top: 80, left: 16, right: 16, maxWidth: 620, maxHeight: 'calc(100% - 140px)', overflowY: 'auto', padding: 12, borderRadius: 10, background: 'rgba(10,15,22,.96)', border: '1px solid var(--line-2)', zIndex: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <span className="label" style={{ fontSize: 11, color: 'var(--dim)' }}>{hits.length} matches · click to open</span>
        <button className="btn" onClick={onClose} aria-label="Close results" style={{ minHeight: 32, padding: '4px 10px' }}>✕</button>
      </div>
      {!hits.length && <p style={{ color: 'var(--muted)' }}>No matches.</p>}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: 8 }}>
        {hits.map((h, i) => {
          const v = vehicles[h.gid]
          return (
            <button key={h.gid} onClick={() => onPick(h)} style={{ padding: 0, borderRadius: 6, overflow: 'hidden', border: `1px solid ${i ? 'var(--line)' : 'var(--amber)'}`, background: 'var(--panel-2)', cursor: 'pointer', textAlign: 'left' }}>
              <img src={asset(h.crop)} alt="" style={{ width: '100%', aspectRatio: '4 / 3', objectFit: 'cover' }} />
              <div className="mono" style={{ padding: '5px 7px', fontSize: 10, lineHeight: 1.5 }}>
                {camName(h.cam)}<br />
                <span style={{ color: 'var(--muted)' }}>{v ? vehicleTitle(v) : ''}</span> · <span style={{ color: i ? 'var(--teal)' : 'var(--amber)' }}>{h.score.toFixed(3)}</span>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
