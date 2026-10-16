import { Link, useParams } from 'react-router-dom'
import { Icon, Spinner } from '../components/ui'
import { fmtDate, img } from '../lib/format'
import { usePet } from '../lib/queries'

/** A4 "missing pet" poster for printing or saving as PDF. */
export default function Poster() {
  const id = Number(useParams().id)
  const { data: pet } = usePet(id)
  if (!pet) return <div className="row" style={{ minHeight: '60vh', justifyContent: 'center' }}><Spinner variant="teal" /></div>
  const url = `${window.location.origin}/pets/${pet.id}`
  const lost = pet.status === 'lost'
  return (
    <div style={{ background: '#e9e2d8', minHeight: '100vh', padding: '24px 16px' }}>
      <style>{`@media print { .no-print { display: none !important; } body, .poster-bg { background: #fff !important; } .poster { box-shadow: none !important; margin: 0 !important; } @page { size: A4; margin: 0; } }`}</style>
      <div className="no-print row" style={{ maxWidth: 794, margin: '0 auto 16px', gap: 10, justifyContent: 'space-between' }}>
        <Link to={`/pets/${pet.id}`} className="btn btn-outline"><Icon name="arrow_back" />Back to listing</Link>
        <button className="btn btn-primary" onClick={() => window.print()}><Icon name="print" />Print or save as PDF</button>
      </div>
      <div className="poster col" style={{ width: '100%', maxWidth: 794, aspectRatio: '210/297', margin: '0 auto', background: '#fff', boxShadow: '0 30px 60px -30px rgba(0,0,0,.4)', padding: 40, gap: 22 }}>
        <div style={{ background: lost ? '#D65A31' : '#176B63', color: '#fff', textAlign: 'center', fontSize: 64, fontWeight: 800, letterSpacing: '.08em', padding: '10px 0', borderRadius: 16 }}>{lost ? 'MISSING' : 'FOUND'}</div>
        <div style={{ flex: 1, minHeight: 0, borderRadius: 20, overflow: 'hidden', background: '#f4eadf' }}>
          <img src={img(pet.photos[0], 1200, 900)} alt={pet.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        </div>
        <div>
          <div style={{ fontSize: 52, fontWeight: 800, letterSpacing: '-.03em', lineHeight: 1 }}>{pet.name}</div>
          <div style={{ fontSize: 22, color: '#4E5A63', marginTop: 6 }}>{pet.breed} · {pet.sex} · {pet.color}</div>
          <div style={{ fontSize: 18, marginTop: 12 }}><strong>{lost ? 'Last seen' : 'Found'}:</strong> {pet.city}, {pet.gov} · {fmtDate(pet.event_date)}</div>
          {pet.reward && <div style={{ fontSize: 20, fontWeight: 800, color: '#C9522B', marginTop: 8 }}>{pet.reward}</div>}
          <p style={{ fontSize: 16, lineHeight: 1.55, marginTop: 12, color: '#3B474F' }}>{pet.description}</p>
        </div>
        <div className="row between" style={{ borderTop: '2px solid #F0E4D6', paddingTop: 18, gap: 16 }}>
          <div>
            <div style={{ fontSize: 14, color: '#68747D' }}>Seen {pet.name}? Contact {pet.contact_name}</div>
            <div style={{ fontSize: 26, fontWeight: 800 }}>{pet.contact_phone ?? 'Message via PawConnect'}</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div className="mono" style={{ fontSize: 13, color: '#68747D' }}>{pet.ref}</div>
            <div style={{ fontSize: 14, fontWeight: 700, wordBreak: 'break-all' }}>{url}</div>
          </div>
        </div>
      </div>
    </div>
  )
}
