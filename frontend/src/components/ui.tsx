import type { CSSProperties, ReactNode } from 'react'
import type { PetStatus } from '../lib/api'
import { STATUS_LABEL } from '../lib/format'

export function Icon({ name, size, fill, style, className = '' }: { name: string; size?: number; fill?: boolean; style?: CSSProperties; className?: string }) {
  return (
    <span aria-hidden="true" className={`icon${fill ? ' fill' : ''} ${className}`} style={{ fontSize: size, ...style }}>
      {name}
    </span>
  )
}

export function LogoMark({ size = 36 }: { size?: number }) {
  return (
    <svg className="logo-mark" width={size} height={size} viewBox="0 0 36 36" aria-hidden="true">
      <rect width="36" height="36" rx="11" fill="#D65A31" />
      <g fill="#FFF9F2">
        <ellipse cx="18" cy="22.5" rx="7" ry="5.5" />
        <ellipse cx="9.5" cy="14" rx="2.5" ry="3" />
        <ellipse cx="15" cy="9.5" rx="2.5" ry="3" />
        <ellipse cx="21" cy="9.5" rx="2.5" ry="3" />
        <ellipse cx="26.5" cy="14" rx="2.5" ry="3" />
      </g>
    </svg>
  )
}

export function StatusBadge({ status, resolved, small }: { status: PetStatus; resolved?: boolean; small?: boolean }) {
  const label = resolved ? (status === 'adopt' ? 'Adopted' : 'Reunited') : STATUS_LABEL[status]
  return (
    <span className={`badge ${resolved ? 'reunited' : status}`} style={small ? { padding: '4px 9px' } : undefined}>
      <span className="d" />
      {label}
    </span>
  )
}

export function Spinner({ variant }: { variant?: 'teal' | 'dark' }) {
  return <span className={`spinner${variant ? ` ${variant}` : ''}`} aria-label="Loading" />
}

export function FieldError({ msg }: { msg?: string }) {
  if (!msg) return null
  return (
    <span className="field-error" role="alert">
      <Icon name="error" />
      {msg}
    </span>
  )
}

export interface Option<T extends string = string> {
  value: T
  label: string
}

export function Seg<T extends string>({ options, value, onChange, className = '', style }: { options: Option<T>[]; value: T; onChange: (v: T) => void; className?: string; style?: CSSProperties }) {
  return (
    <div className={`seg ${className}`} style={style} role="radiogroup">
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={o.value === value} className={o.value === value ? 'on' : ''} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Pills<T extends string>({ options, value, onChange }: { options: Option<T>[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="pills" role="radiogroup">
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={o.value === value} className={`pill${o.value === value ? ' on' : ''}`} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Toggle({ on, onChange, label }: { on: boolean; onChange: (v: boolean) => void; label: string }) {
  return <button type="button" role="switch" aria-checked={on} aria-label={label} className={`toggle${on ? ' on' : ''}`} onClick={() => onChange(!on)} />
}

/** Circular score like the design's conic-gradient rings. */
export function ScoreRing({ score, size = 76, inner, fontSize = 17, color = '#176B63', track = '#E6EEEC', label }: { score: number; size?: number; inner?: number; fontSize?: number; color?: string; track?: string; label?: ReactNode }) {
  const i = inner ?? Math.round(size * 0.815)
  return (
    <div className="ring" style={{ width: size, height: size, background: `conic-gradient(${color} ${Math.round(score * 3.6)}deg, ${track} 0)` }}>
      <div style={{ width: i, height: i, fontSize }}>{label ?? `${score}%`}</div>
    </div>
  )
}

export function BreedBars({ breeds, compact }: { breeds: [string, number][]; compact?: boolean }) {
  return (
    <div className="col" style={{ gap: compact ? 9 : 10 }}>
      {breeds.slice(0, 3).map(([label, pct], i) => (
        <div key={label}>
          <div className="row between" style={{ fontSize: compact ? 13 : 14, marginBottom: compact ? 4 : 5 }}>
            <span style={{ fontWeight: compact ? 500 : i === 0 ? 800 : 500 }}>{label}</span>
            <span className="mono" style={{ fontWeight: 600 }}>{pct}%</span>
          </div>
          <div className="bar" style={compact ? { height: 5 } : undefined}>
            <div style={{ width: `${pct}%`, background: i === 0 ? '#176B63' : '#9BBFBA' }} />
          </div>
        </div>
      ))}
    </div>
  )
}

export function Empty({ icon, title, children, actions }: { icon: string; title: string; children?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="empty">
      <div className="bubble"><Icon name={icon} /></div>
      <div className="empty-title">{title}</div>
      {children && <p style={{ maxWidth: 420, fontSize: 15, lineHeight: 1.55, color: 'var(--muted)' }}>{children}</p>}
      {actions && <div className="wrap-row" style={{ gap: 10, justifyContent: 'center', marginTop: 6 }}>{actions}</div>}
    </div>
  )
}

export function SectionHead({ eyebrow, title, teal, action }: { eyebrow?: string; title: string; teal?: boolean; action?: ReactNode }) {
  return (
    <div className="section-head">
      <div>
        {eyebrow && <div className={`eyebrow${teal ? ' teal' : ''}`}>{eyebrow}</div>}
        <h2 className="h-section">{title}</h2>
      </div>
      {action}
    </div>
  )
}

export function Skeleton({ height, style }: { height: number | string; style?: CSSProperties }) {
  return <div className="skeleton" style={{ height, ...style }} />
}
