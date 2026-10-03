import { useEffect, useMemo, useRef, useState } from 'react'
import TrackedVideo from '../components/TrackedVideo'
import { asset, getMode, loadCameras, loadTracks, loadVehicles, searchText, transcribe, type Camera, type Det, type SearchHit, type Tracks, type Vehicle } from '../data'
import VancouverMap from './VancouverMap'

export default function Live() {
  const [cameras, setCameras] = useState<Camera[]>([])
  const [vehicles, setVehicles] = useState<Record<string, Vehicle>>({})
  const [camId, setCamId] = useState<string>()
  const [tracks, setTracks] = useState<Tracks | null>(null)
  const [dets, setDets] = useState<Det[]>([])
  const [gid, setGid] = useState<string>()
  const [hits, setHits] = useState<SearchHit[] | null>(null)

  useEffect(() => { Promise.all([loadCameras(), loadVehicles()]).then(([c, v]) => { setCameras(c); setVehicles(v) }) }, [])
  const cam = cameras.find((c) => c.id === camId)
  useEffect(() => { setTracks(null); if (cam) loadTracks(cam).then(setTracks) }, [cam])

  const vehicle = gid ? vehicles[gid] : undefined
  const trail = vehicle ? [...new Set(vehicle.sightings.map((s) => s.cam))] : []
  const open = (cameraId: string, g?: string) => { setCamId(cameraId); setGid(g) }

  return (
    <main style={{ display: 'flex', flexWrap: 'wrap', height: 'calc(100vh - 73px)', minHeight: 560 }}>
      <div style={{ position: 'relative', flex: '1 1 560px', minWidth: 0, height: '100%', borderRight: '1px solid var(--line)' }}>
        <VancouverMap cameras={cameras} selected={camId} highlight={trail} onSelect={(id) => open(id)} />
        <SearchBar onResults={setHits} />
        {hits && <Results hits={hits} vehicles={vehicles} cameras={cameras} onPick={(h) => { open(h.cam, h.gid); setHits(null) }} onClose={() => setHits(null)} />}
        <div className="mono" style={{ position: 'absolute', left: 16, bottom: 14, fontSize: 11, color: 'var(--dim)', lineHeight: 1.7 }}>
          <span className="blink" style={{ display: 'inline-block', width: 7, height: 7, borderRadius: '50%', background: 'var(--rose)', marginRight: 6 }} />
          {cameras.length} CAMERAS ONLINE · {Object.keys(vehicles).length} VEHICLES INDEXED
        </div>
      </div>

      <aside style={{ flex: '0 1 460px', minWidth: 320, height: '100%', overflowY: 'auto', background: 'var(--panel)' }}>
        {!cam ? (
          <div style={{ padding: 28 }}>
            <div className="label" style={{ color: 'var(--teal)' }}>City of Vancouver</div>
            <h2 style={{ fontSize: 30, letterSpacing: -1, margin: '10px 0' }}>Pick a camera.</h2>
            <p style={{ color: 'var(--muted)', lineHeight: 1.55 }}>Every feed runs detection, tracking and re-identification. Click a vehicle to see what it is and which other cameras saw something similar — or search by description or voice.</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 18 }}>
              {cameras.map((c) => (
                <button key={c.id} className="btn" onClick={() => open(c.id)} style={{ justifyContent: 'space-between' }}>
                  <span>{c.name}</span><span className="mono" style={{ fontSize: 11, color: 'var(--dim)' }}>{c.condition.toUpperCase()}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div style={{ padding: 18 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
              <div>
                <div className="label" style={{ color: 'var(--teal)', fontSize: 11 }}>{cam.view} · {cam.condition} · {cam.res}</div>
                <h2 style={{ fontSize: 24, margin: '6px 0 2px' }}>{cam.name}</h2>
                <div className="mono" style={{ fontSize: 11, color: 'var(--faint)' }}>Feed source: {cam.source}</div>
              </div>
              <button className="btn" aria-label="Close camera" onClick={() => { setCamId(undefined); setGid(undefined) }}>✕</button>
            </div>
            <div style={{ marginTop: 14, borderRadius: 8, overflow: 'hidden', border: '1px solid var(--line)' }}>
              <TrackedVideo key={cam.id} src={asset(cam.clip)} poster={asset(cam.poster)} tracks={tracks} highlight={gid?.startsWith(cam.id + ':') ? Number(gid.split(':')[1]) : undefined} onFrame={(_, d) => setDets(d)} onPick={(tid) => setGid(`${cam.id}:${tid}`)}>
                <span className="chip" style={{ position: 'absolute', left: 8, top: 8, display: 'flex', gap: 6, alignItems: 'center', color: 'var(--text)' }}>
                  <span className="blink" style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--rose)' }} />LIVE · {dets.length} VEHICLES
                </span>
              </TrackedVideo>
            </div>
            {vehicle ? (
              <VehicleCard v={vehicle} cameras={cameras} onJump={(c, g) => open(c, g)} onBack={() => setGid(undefined)} />
            ) : (
              <InFrame cam={cam} dets={dets} tracks={tracks} vehicles={vehicles} onPick={setGid} />
            )}
          </div>
        )}
      </aside>
    </main>
  )
}

function InFrame({ cam, dets, tracks, vehicles, onPick }: { cam: Camera; dets: Det[]; tracks: Tracks | null; vehicles: Record<string, Vehicle>; onPick: (g: string) => void }) {
  const all = useMemo(() => {
    const ids = new Set<number>()
    tracks?.frames.forEach((f) => f.forEach((d) => ids.add(d[0])))
    return [...ids]
  }, [tracks])
  const visible = new Set(dets.map((d) => d[0]))
  return (
    <div style={{ marginTop: 16 }}>
      <div className="label" style={{ color: 'var(--dim)', fontSize: 11, marginBottom: 10 }}>Tracked on this camera · click to inspect</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 8 }}>
        {all.map((tid) => {
          const v = vehicles[`${cam.id}:${tid}`]
          return (
            <button key={tid} onClick={() => v && onPick(v.gid)} disabled={!v} style={{ padding: 0, border: `1px solid ${visible.has(tid) ? 'var(--teal)' : 'var(--line)'}`, borderRadius: 6, background: 'var(--panel-2)', overflow: 'hidden', cursor: v ? 'pointer' : 'default', textAlign: 'left' }}>
              {v ? <img src={asset(v.crop)} alt="" style={{ width: '100%', aspectRatio: '4 / 3', objectFit: 'cover' }} /> : <div style={{ aspectRatio: '4 / 3', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--faint)', fontSize: 11 }}>too far</div>}
              <div className="mono" style={{ padding: '5px 7px', fontSize: 10, lineHeight: 1.4 }}>
                #{tid}<br /><span style={{ color: 'var(--muted)' }}>{v ? v.top5[0]?.name : 'detected only'}</span>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function VehicleCard({ v, cameras, onJump, onBack }: { v: Vehicle; cameras: Camera[]; onJump: (cam: string, gid: string) => void; onBack: () => void }) {
  const name = (id: string) => cameras.find((c) => c.id === id)?.name ?? id
  return (
    <div style={{ marginTop: 16 }}>
      <button className="btn mono" onClick={onBack} style={{ fontSize: 12, minHeight: 36, padding: '6px 12px' }}>← all vehicles</button>
      <div style={{ display: 'flex', gap: 14, marginTop: 12 }}>
        <img src={asset(v.crop)} alt="Selected vehicle" style={{ width: 150, aspectRatio: '4 / 3', objectFit: 'cover', borderRadius: 6, border: '2px solid var(--amber)' }} />
        <div style={{ minWidth: 0 }}>
          <div style={{ fontWeight: 700, fontSize: 18 }}>{v.top5[0]?.name}</div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--dim)', marginTop: 4 }}>#{v.gid.split(':')[1]} · {v.color}</div>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginTop: 14 }}>
        {v.top5.map((r, i) => (
          <div key={r.name}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}><span>{r.name}</span><span className="mono" style={{ color: i ? 'var(--faint)' : 'var(--amber)' }}>{r.p.toFixed(2)}</span></div>
            <div style={{ height: 4, marginTop: 3, background: '#141c28', borderRadius: 2 }}><div style={{ height: '100%', width: `${r.p * 100}%`, background: i ? 'var(--faint)' : 'var(--amber)', borderRadius: 2 }} /></div>
          </div>
        ))}
      </div>
      <div className="label" style={{ color: 'var(--dim)', fontSize: 11, margin: '18px 0 10px' }}>{v.sightings.some((s) => s.verified) ? 'Same car on other cameras' : 'Most similar on other cameras'}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {v.sightings.slice(1).map((s) => (
          <button key={s.cam + s.track} className="btn" onClick={() => onJump(s.cam, `${s.cam}:${s.track}`)} style={{ padding: 6, gap: 10, justifyContent: 'flex-start' }}>
            <img src={asset(s.crop)} alt="" style={{ width: 64, aspectRatio: '4 / 3', objectFit: 'cover', borderRadius: 4 }} />
            <span style={{ flex: 1, textAlign: 'left', fontSize: 13 }}>{name(s.cam)} · #{s.track}</span>
            <span className="mono" style={{ fontSize: 12, color: s.verified ? 'var(--teal)' : 'var(--amber)' }}>{s.verified ? '✓ ' : ''}{s.sim.toFixed(2)}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

function SearchBar({ onResults }: { onResults: (h: SearchHit[]) => void }) {
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
      // Offline demo: browser speech recognition stands in for Whisper.
      const SR = (window as unknown as { webkitSpeechRecognition?: new () => { lang: string; onresult: (e: { results: { 0: { 0: { transcript: string } } } }) => void; onend: () => void; start: () => void } }).webkitSpeechRecognition
      if (!SR) { setNote('Voice needs the live backend (Whisper).'); return }
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
      <label htmlFor="q" className="mono" style={{ fontSize: 11, color: 'var(--dim)', paddingLeft: 6 }}>FIND</label>
      <input id="q" value={q} onChange={(e) => setQ(e.target.value)} placeholder="white sedan, dark SUV with roof rails…" style={{ flex: 1, minWidth: 0, background: 'transparent', border: 'none', outline: 'none', color: 'var(--text)', fontSize: 15, fontFamily: 'var(--sans)', minHeight: 40 }} />
      <button type="button" className="btn" onClick={voice} aria-label={busy === 'listen' ? 'Stop recording' : 'Search by voice'} style={{ minHeight: 40, padding: '8px 12px', borderColor: busy === 'listen' ? 'var(--violet)' : undefined, color: busy === 'listen' ? 'var(--violet)' : undefined }}>
        {busy === 'listen' ? '■ Stop' : '🎙'}
      </button>
      <button type="submit" className="btn btn-primary" style={{ minHeight: 40, padding: '8px 14px' }} disabled={busy === 'search'}>{busy === 'search' ? '…' : 'Search'}</button>
      {note && <span className="mono" style={{ position: 'absolute', top: '100%', left: 8, marginTop: 6, fontSize: 11, color: 'var(--rose)' }}>{note}</span>}
    </form>
  )
}

function Results({ hits, vehicles, cameras, onPick, onClose }: { hits: SearchHit[]; vehicles: Record<string, Vehicle>; cameras: Camera[]; onPick: (h: SearchHit) => void; onClose: () => void }) {
  return (
    <div style={{ position: 'absolute', top: 80, left: 16, width: 'min(620px, calc(100% - 32px))', maxHeight: 'calc(100% - 140px)', overflowY: 'auto', padding: 12, borderRadius: 10, background: 'rgba(10,15,22,.96)', border: '1px solid var(--line-2)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <span className="label" style={{ fontSize: 11, color: 'var(--dim)' }}>{hits.length} matches</span>
        <button className="btn" onClick={onClose} aria-label="Close results" style={{ minHeight: 32, padding: '4px 10px' }}>✕</button>
      </div>
      {!hits.length && <p style={{ color: 'var(--muted)' }}>No matches.</p>}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: 8 }}>
        {hits.map((h, i) => (
          <button key={h.gid} onClick={() => onPick(h)} style={{ padding: 0, borderRadius: 6, overflow: 'hidden', border: `1px solid ${i ? 'var(--line)' : 'var(--amber)'}`, background: 'var(--panel-2)', cursor: 'pointer', textAlign: 'left' }}>
            <img src={asset(h.crop)} alt="" style={{ width: '100%', aspectRatio: '4 / 3', objectFit: 'cover' }} />
            <div className="mono" style={{ padding: '5px 7px', fontSize: 10, lineHeight: 1.5 }}>
              {cameras.find((c) => c.id === h.cam)?.name ?? h.cam}<br />
              <span style={{ color: 'var(--muted)' }}>{vehicles[h.gid]?.top5[0]?.name}</span> · <span style={{ color: i ? 'var(--teal)' : 'var(--amber)' }}>{h.score.toFixed(3)}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
