import { useEffect, useRef, useState } from 'react'
import type { Det, Tracks } from '../data'

interface Props {
  src: string
  poster?: string
  tracks?: Tracks | null
  /** 0..1 — when set, the video follows scroll instead of playing. */
  progress?: number
  /** Track to draw in amber; every other box is dimmed by `othersOpacity`. */
  highlight?: number
  othersOpacity?: number
  /** Per-box reveal 0..1 (index = order of first appearance); omit to show all boxes. */
  reveal?: (order: number) => number
  onFrame?: (frame: number, dets: Det[]) => void
  onPick?: (track: number) => void
  /** Seconds to jump to once the clip loads (e.g. when opening a sighting). */
  startAt?: number
  /** Show a fullscreen button; boxes stay aligned because the overlay goes fullscreen with the video. */
  fullscreen?: boolean
  /** Custom box label per track (defaults to detector class + confidence). */
  labelFor?: (track: number) => string | undefined
  className?: string
  children?: React.ReactNode
}

/** Plays (or scroll-scrubs) a clip and overlays the detector's per-frame tracks. */
export default function TrackedVideo({ src, poster, tracks, progress, highlight, othersOpacity = 1, reveal, onFrame, onPick, startAt, fullscreen, labelFor, className, children }: Props) {
  const outer = useRef<HTMLDivElement>(null)
  const video = useRef<HTMLVideoElement>(null)
  const [frame, setFrame] = useState(0)
  const wanted = useRef(0)
  const scrub = progress !== undefined

  // Playback mode: follow the decoded frame exactly so boxes never drift from the cars.
  useEffect(() => {
    const v = video.current
    if (!v || scrub || !tracks) return
    let handle = 0
    let alive = true
    const tick = (_: number, meta?: VideoFrameCallbackMetadata) => {
      if (!alive) return
      const t = meta ? meta.mediaTime : v.currentTime
      setFrame(Math.min(tracks.n - 1, Math.round(t * tracks.fps)))  // round: mediaTime k/fps can land just below k
      handle = 'requestVideoFrameCallback' in v ? v.requestVideoFrameCallback(tick) : requestAnimationFrame(() => tick(0))
    }
    handle = 'requestVideoFrameCallback' in v ? v.requestVideoFrameCallback(tick) : requestAnimationFrame(() => tick(0))
    return () => {
      alive = false
      if ('cancelVideoFrameCallback' in v) v.cancelVideoFrameCallback(handle)
      else cancelAnimationFrame(handle)
    }
  }, [scrub, tracks, src])

  // Scrub mode: scroll position drives the clip.
  useEffect(() => {
    const v = video.current
    if (!v || !scrub || !tracks) return
    const f = Math.min(tracks.n - 1, Math.max(0, Math.round(progress! * (tracks.n - 1))))
    setFrame(f)
    wanted.current = f / tracks.fps
    if (v.readyState >= 1) v.currentTime = wanted.current
  }, [progress, scrub, tracks])

  const dets = tracks?.frames[frame] ?? []
  useEffect(() => { onFrame?.(frame, dets) }, [frame]) // eslint-disable-line react-hooks/exhaustive-deps

  // Order tracks by first appearance so reveal() can stagger them.
  const order = useRef(new Map<number, number>())
  useEffect(() => {
    order.current = new Map()
    tracks?.frames.forEach((f) => f.forEach((d) => { if (!order.current.has(d[0])) order.current.set(d[0], order.current.size) }))
  }, [tracks])

  const ar = tracks ? `${tracks.w} / ${tracks.h}` : '16 / 9'
  const inner = (
    <div className={fullscreen ? 'tv-inner' : className} style={{ position: 'relative', aspectRatio: ar, overflow: 'hidden', background: '#0a0f16', ['--ar' as string]: tracks ? tracks.w / tracks.h : 16 / 9 }}>
      <video
        ref={video}
        src={src}
        poster={poster}
        muted
        playsInline
        loop={!scrub}
        autoPlay={!scrub}
        preload="auto"
        onLoadedData={(e) => {
          if (scrub) e.currentTarget.currentTime = wanted.current + 0.001
          else if (startAt) e.currentTarget.currentTime = startAt
        }}
        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
      />
      {tracks && dets.map((d) => {
        const [tid, x1, y1, x2, y2, conf, cls] = d
        const isTarget = highlight === tid
        const r = reveal ? reveal(order.current.get(tid) ?? 0) : 1
        const o = isTarget ? 1 : r * othersOpacity
        if (o <= 0.01) return null
        return (
          <div
            key={tid}
            className={`bbox${isTarget ? ' target' : ''}`}
            onClick={onPick ? () => onPick(tid) : undefined}
            style={{
              left: `${(x1 / tracks.w) * 100}%`, top: `${(y1 / tracks.h) * 100}%`,
              width: `${((x2 - x1) / tracks.w) * 100}%`, height: `${((y2 - y1) / tracks.h) * 100}%`,
              opacity: o, transform: `scale(${1.25 - 0.25 * Math.min(1, r)})`, cursor: onPick ? 'pointer' : undefined,
            }}
          >
            <span>#{tid} {labelFor?.(tid) ?? `${tracks.names[String(cls)] ?? 'vehicle'} ${conf.toFixed(2)}`}</span>
          </div>
        )
      })}
      {children}
    </div>
  )
  if (!fullscreen) return inner
  const toggle = () => (document.fullscreenElement ? document.exitFullscreen() : outer.current?.requestFullscreen())
  return (
    <div ref={outer} className={`tv-outer ${className ?? ''}`} style={{ position: 'relative' }}>
      {inner}
      <button className="tv-fs" onClick={toggle} aria-label="Toggle fullscreen" title="Fullscreen">⛶</button>
    </div>
  )
}
