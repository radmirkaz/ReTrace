import { pub } from '../data'

interface Props {
  page: 'landing' | 'live'
  nerd: boolean
  onNerd: () => void
}

export default function Nav({ page, nerd, onNerd }: Props) {
  return (
    <header
      style={{
        position: 'sticky', top: 0, zIndex: 50, display: 'flex', flexWrap: 'wrap', gap: 12,
        alignItems: 'center', justifyContent: 'space-between', padding: '14px 28px',
        background: 'rgba(5,7,11,.72)', backdropFilter: 'blur(12px)', borderBottom: '1px solid #141c28',
      }}
    >
      <a href="#/" style={{ display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none', color: 'var(--text)' }}>
        <img src={pub('mark.svg')} alt="" width={40} height={32} />
        <span style={{ fontSize: 22, fontWeight: 700, letterSpacing: -0.5 }}>
          <span style={{ color: 'var(--teal)' }}>Re</span>Trace
        </span>
      </a>
      <nav style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
        <a className="btn" href="#/" style={page === 'landing' ? { borderColor: 'var(--teal)' } : undefined}>Story</a>
        <a className="btn" href="#/live" style={page === 'live' ? { borderColor: 'var(--teal)' } : undefined}>
          <span className="blink" style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--rose)' }} />
          Live map
        </a>
        <button className="btn mono" onClick={onNerd} aria-pressed={nerd} style={{ fontSize: 13, borderColor: nerd ? 'var(--violet)' : undefined, color: nerd ? 'var(--violet)' : undefined }}>
          {'</>'} {nerd ? 'Hide' : 'Under the hood'}
        </button>
      </nav>
    </header>
  )
}
