import type { Pet, PetStatus } from './api'

const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/** Sized image URL. Unsplash photos are resized by their CDN; uploads are served as-is. */
export function img(url: string | undefined, w: number, h?: number): string {
  if (!url) return ''
  if (!url.startsWith('https://images.unsplash.com/')) return url
  const sep = url.includes('?') ? '&' : '?'
  return `${url}${sep}w=${w}${h ? `&h=${h}` : ''}&q=80&auto=format&fit=crop`
}

export function ago(days: number): string {
  if (days <= 0) return 'Today'
  if (days === 1) return 'Yesterday'
  if (days < 7) return `${days} days ago`
  if (days < 14) return '1 week ago'
  return `${Math.round(days / 7)} weeks ago`
}

export const fmtKm = (d: number) => (d < 10 ? `${d.toFixed(1)} km` : `${Math.round(d)} km`)

export function initials(name: string): string {
  return name.split(' ').filter(Boolean).slice(0, 2).map((w) => w[0]).join('').toUpperCase()
}

export function fmtDate(iso: string): string {
  const d = new Date(iso.length === 10 ? `${iso}T12:00:00` : iso)
  return `${d.getDate()} ${MON[d.getMonth()]} ${d.getFullYear()}`
}

const pad = (n: number) => String(n).padStart(2, '0')

/** "Today, 08:10" / "Yesterday, 21:40" / "13 Oct, 22:15" */
export function dayTime(iso: string): string {
  const d = new Date(iso)
  const today = new Date()
  const y = new Date(today)
  y.setDate(today.getDate() - 1)
  const t = `${pad(d.getHours())}:${pad(d.getMinutes())}`
  if (d.toDateString() === today.toDateString()) return `Today, ${t}`
  if (d.toDateString() === y.toDateString()) return `Yesterday, ${t}`
  return `${d.getDate()} ${MON[d.getMonth()]}, ${t}`
}

/** "12 min ago" / "2 h ago" / "Yesterday" / "3 days ago" */
export function relTime(iso: string): string {
  const mins = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000))
  if (mins < 1) return 'Just now'
  if (mins < 60) return `${mins} min ago`
  const hrs = Math.round(mins / 60)
  if (hrs < 24) return `${hrs} h ago`
  const days = Math.round(hrs / 24)
  return days === 1 ? 'Yesterday' : `${days} days ago`
}

export const STATUS_LABEL: Record<PetStatus, string> = { lost: 'Lost', found: 'Found', adopt: 'For adoption' }
export const STATUS_DOT: Record<PetStatus, string> = { lost: '#D65A31', found: '#176B63', adopt: '#C99A2E' }

export function whenLabel(p: Pet): string {
  const a = ago(p.days_ago)
  return p.status === 'adopt' ? `Listed ${a.toLowerCase()}` : a
}

export const metaLabel = (p: Pet) => `${p.breed} · ${p.age_label}`

export const firstName = (name: string) => name.split(' ')[0] ?? name

export const GOVERNORATES = ['Tunis', 'Ariana', 'Ben Arous', 'Manouba', 'Nabeul', 'Sousse', 'Sfax']

export const SPEED = ['Same day', '1–7 days', '8–30 days', '31–90 days', '100+ days']
export const speedColor = (i: number) => (i <= 1 ? '#176B63' : i === 2 ? '#C99A2E' : '#C9522B')
