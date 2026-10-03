import { useEffect } from 'react'

/** Full-screen view of a vehicle crop. Click anywhere or press Esc to close. */
export default function Lightbox({ src, caption, onClose }: { src: string; caption?: string; onClose: () => void }) {
  useEffect(() => {
    const on = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  }, [onClose])
  return (
    <div className="lightbox" role="dialog" aria-label="Vehicle crop" onClick={onClose}>
      <img src={src} alt={caption ?? 'Vehicle crop'} />
      {caption && <div className="mono" style={{ fontSize: 13, color: 'var(--muted)' }}>{caption}</div>}
      <div className="mono" style={{ fontSize: 11, color: 'var(--faint)' }}>CLICK OR ESC TO CLOSE</div>
    </div>
  )
}
