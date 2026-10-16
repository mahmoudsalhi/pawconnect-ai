import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { MapView } from '../components/MapView'
import { PetGrid } from '../components/PetCard'
import { PetModal, type ModalKind } from '../components/PetModals'
import { BreedBars, Empty, Icon, Skeleton, StatusBadge } from '../components/ui'
import { api, type Pet } from '../lib/api'
import { dayTime, fmtDate, fmtKm, img, initials, STATUS_DOT, STATUS_LABEL } from '../lib/format'
import { useFavorite, usePet, useRequireAuth, useSimilar } from '../lib/queries'
import { useToast } from '../lib/toast'

function confidence(top: number) {
  if (top >= 80) return { label: 'High confidence', color: '#115850', bg: '#D7ECE8' }
  if (top >= 50) return { label: 'Moderate confidence', color: '#6B500D', bg: '#F4E8CC' }
  return { label: 'Low confidence · likely mixed breed', color: '#9E3C19', bg: '#FBE1D5' }
}

function OwnerActions({ pet }: { pet: Pet }) {
  const qc = useQueryClient()
  const toast = useToast()
  const navigate = useNavigate()
  const resolve = useMutation({
    mutationFn: () => api.post(`/api/pets/${pet.id}/resolve`),
    onSuccess: () => {
      qc.invalidateQueries()
      toast(pet.status === 'adopt' ? `${pet.name} is marked as adopted` : `Wonderful! ${pet.name} is marked as reunited`)
    },
    onError: (e: Error) => toast(e.message, true),
  })
  const remove = useMutation({
    mutationFn: () => api.del(`/api/pets/${pet.id}`),
    onSuccess: () => {
      qc.invalidateQueries()
      toast('Report deleted')
      navigate('/dashboard/listings')
    },
    onError: (e: Error) => toast(e.message, true),
  })
  const [confirm, setConfirm] = useState(false)
  if (pet.resolved) return <div className="info"><Icon name="favorite" size={20} />This listing is closed and no longer appears in search.</div>
  return (
    <div className="card col" style={{ padding: 16, borderRadius: 18, gap: 12 }}>
      <div style={{ fontSize: 14, fontWeight: 800 }}>This is your listing</div>
      <div className="wrap-row" style={{ gap: 8 }}>
        <button className="btn btn-teal btn-md" disabled={resolve.isPending} onClick={() => resolve.mutate()}><Icon name="favorite" size={18} />{pet.status === 'adopt' ? 'Mark as adopted' : 'Mark as reunited'}</button>
        <Link to={`/match?pet=${pet.id}`} className="btn btn-soft btn-md"><Icon name="image_search" size={18} />See possible matches</Link>
        {pet.status !== 'adopt' && (confirm
          ? <button className="btn btn-md" style={{ background: '#C23B22', color: '#fff' }} disabled={remove.isPending} onClick={() => remove.mutate()}>Confirm delete</button>
          : <button className="btn btn-ghost btn-md" onClick={() => setConfirm(true)}>Delete report</button>)}
      </div>
    </div>
  )
}

export default function PetDetail() {
  const id = Number(useParams().id)
  const { data: pet, isLoading, isError } = usePet(id)
  const similar = useSimilar(id)
  const toggleFav = useFavorite()
  const requireAuth = useRequireAuth()
  const toast = useToast()
  const [gi, setGi] = useState(0)
  const [modal, setModal] = useState<ModalKind | null>(null)

  if (isLoading) {
    return (
      <main className="wrap" style={{ paddingTop: 28, paddingBottom: 88 }}>
        <div className="wrap-row" style={{ gap: 44 }}><Skeleton height={520} style={{ flex: '1.25 1 480px' }} /><Skeleton height={520} style={{ flex: '1 1 380px' }} /></div>
      </main>
    )
  }
  if (isError || !pet) {
    return (
      <main className="wrap" style={{ paddingTop: 60, paddingBottom: 88 }}>
        <Empty icon="pets" title="This listing isn’t available" actions={<Link to="/pets" className="btn btn-dark">Browse pets</Link>}>It may have been removed, or the link is incorrect.</Empty>
      </main>
    )
  }

  const photos = pet.photos.length ? pet.photos : ['']
  const g = gi % photos.length
  const isLost = pet.status === 'lost'
  const isFound = pet.status === 'found'
  const isAdopt = pet.status === 'adopt'
  const conf = confidence(pet.ai_breeds[0]?.[1] ?? 0)
  const locWord = isLost ? 'Last seen' : isFound ? 'Found at' : 'Shelter'
  const sinceLabel = isLost ? `Missing since ${fmtDate(pet.event_date)}` : isFound ? `Found on ${fmtDate(pet.event_date)}` : `Listed on ${fmtDate(pet.event_date)}`
  const open = (k: ModalKind) => {
    if (k === 'sighting' || requireAuth(k === 'apply' ? 'Sign in to apply for adoption.' : 'Sign in to message the owner or shelter.')) setModal(k)
  }
  const share = async () => {
    const url = window.location.href
    try {
      if (navigator.share) await navigator.share({ title: `${pet.name} · PawConnect`, url })
      else {
        await navigator.clipboard.writeText(url)
        toast('Link copied')
      }
    } catch {
      /* share sheet dismissed */
    }
  }
  const attrs: [string, string][] = [['Breed', pet.breed || 'Unknown'], ['Age', pet.age_label], ['Sex', pet.sex], ['Size', pet.size], ['Colour', pet.color], [isAdopt ? 'Vaccinated' : 'Microchip', pet.chip || 'Unknown']]

  return (
    <main className="wrap" style={{ paddingTop: 28, paddingBottom: 88 }}>
      <div className="wrap-row" style={{ alignItems: 'center', gap: 8, fontSize: 14, color: 'var(--muted)', marginBottom: 22 }}>
        <Link to="/pets" className="row" style={{ gap: 4, fontWeight: 600, color: 'var(--muted)' }}><Icon name="arrow_back" size={18} />Find a pet</Link>
        <span>/</span><span>{STATUS_LABEL[pet.status]}</span><span>/</span><span style={{ color: 'var(--ink)', fontWeight: 700 }}>{pet.name}</span>
      </div>

      <div className="wrap-row" style={{ gap: 'clamp(24px,3vw,44px)', alignItems: 'flex-start' }}>
        <div className="col" style={{ flex: '1.25 1 480px', minWidth: 0, gap: 12 }}>
          <div style={{ position: 'relative', borderRadius: 24, overflow: 'hidden', aspectRatio: '5/4', background: 'var(--sand)' }}>
            <img src={img(photos[g], 1400, 1120)} alt={pet.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            {photos.length > 1 && (
              <>
                <button className="round-btn" aria-label="Previous photo" onClick={() => setGi((g + photos.length - 1) % photos.length)} style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', width: 44, height: 44 }}><Icon name="chevron_left" size={22} /></button>
                <button className="round-btn" aria-label="Next photo" onClick={() => setGi((g + 1) % photos.length)} style={{ position: 'absolute', right: 14, top: '50%', transform: 'translateY(-50%)', width: 44, height: 44 }}><Icon name="chevron_right" size={22} /></button>
              </>
            )}
            <div className="mono" style={{ position: 'absolute', right: 14, bottom: 14, padding: '5px 10px', borderRadius: 8, background: 'rgba(32,42,50,.75)', color: '#fff', fontSize: 12 }}>{g + 1} / {photos.length}</div>
          </div>
          {photos.length > 1 && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,minmax(0,1fr))', gap: 10 }}>
              {photos.map((src, i) => (
                <button key={src + i} onClick={() => setGi(i)} aria-label={`Photo ${i + 1}`} style={{ padding: 0, border: `2.5px solid ${i === g ? '#D65A31' : 'transparent'}`, borderRadius: 14, overflow: 'hidden', aspectRatio: '5/4', background: 'var(--sand)', opacity: i === g ? 1 : 0.7 }}>
                  <img src={img(src, 300, 240)} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="col" style={{ flex: '1 1 380px', minWidth: 0, gap: 20 }}>
          <div className="wrap-row" style={{ gap: 8, alignItems: 'center' }}>
            <StatusBadge status={pet.status} resolved={pet.resolved} />
            <span style={{ fontSize: 14, color: 'var(--muted)' }}>{sinceLabel}</span>
            <span className="mono muted" style={{ marginLeft: 'auto', fontSize: 12 }}>{pet.ref}</span>
          </div>
          <div>
            <h1 style={{ margin: '0 0 6px', fontSize: 'clamp(40px,4.4vw,58px)', fontWeight: 800, letterSpacing: '-.04em', lineHeight: 1 }}>{pet.name}</h1>
            <div style={{ fontSize: 17, color: 'var(--body)' }}>{pet.breed} · {pet.sex} · {pet.age_label}</div>
          </div>
          {pet.reward && (
            <div className="row" style={{ gap: 10, padding: '12px 14px', borderRadius: 14, background: 'var(--ink)', color: 'var(--cream)', fontSize: 14, fontWeight: 600 }}>
              <Icon name="volunteer_activism" size={20} style={{ color: '#F4A988' }} /><span>{pet.reward} offered by the owner</span>
            </div>
          )}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,minmax(0,1fr))', gap: 1, background: 'var(--line)', border: '1px solid var(--line)', borderRadius: 16, overflow: 'hidden' }}>
            {attrs.map(([l, v]) => (
              <div key={l} style={{ background: '#fff', padding: '14px 16px', minWidth: 0 }}>
                <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 3 }}>{l}</div>
                <div style={{ fontSize: 15, fontWeight: 700, overflowWrap: 'anywhere' }}>{v}</div>
              </div>
            ))}
          </div>
          <div className="row" style={{ gap: 12, alignItems: 'flex-start', fontSize: 15 }}>
            <Icon name="location_on" size={22} style={{ color: 'var(--teal)' }} />
            <div>
              <div style={{ fontWeight: 700 }}>{locWord}: {pet.city}, {pet.gov}</div>
              <div style={{ fontSize: 14, color: 'var(--muted)' }}>{fmtDate(pet.event_date)}{pet.event_time ? ` · around ${pet.event_time}` : ''} · {pet.distance_km < 0.1 ? 'Near you' : `${fmtKm(pet.distance_km)} from La Marsa`}</div>
            </div>
          </div>

          {pet.is_mine ? <OwnerActions pet={pet} /> : !pet.resolved && (
            <div className="wrap-row" style={{ gap: 10 }}>
              {isAdopt && <button className="btn btn-primary btn-lg glow" style={{ flex: '1 1 200px' }} onClick={() => open('apply')}><Icon name="favorite" />Apply for adoption</button>}
              {isLost && (
                <>
                  <button className="btn btn-primary btn-lg glow" style={{ flex: '1 1 180px' }} onClick={() => open('contact')}><Icon name="chat" />Contact owner</button>
                  <button className="btn btn-teal btn-lg" style={{ flex: '1 1 180px' }} onClick={() => open('sighting')}><Icon name="visibility" />Report a sighting</button>
                </>
              )}
              {isFound && <button className="btn btn-primary btn-lg" style={{ flex: '1 1 180px' }} onClick={() => open('contact')}><Icon name="pets" />This is my pet</button>}
              {isAdopt && <button className="btn btn-outline btn-lg" style={{ flex: '1 1 160px', fontSize: 15 }} onClick={() => open('contact')}>Ask the shelter</button>}
            </div>
          )}
          <div className="wrap-row" style={{ gap: 8 }}>
            <button className={`btn btn-outline btn-md fav${pet.is_favorite ? ' on' : ''}`} style={{ fontSize: 14, borderRadius: 12 }} onClick={() => toggleFav(pet)}>
              <Icon name="favorite" size={19} />{pet.is_favorite ? 'Saved' : 'Save'}
            </button>
            <button className="btn btn-outline btn-md" style={{ fontSize: 14, borderRadius: 12 }} onClick={share}><Icon name="ios_share" size={19} />Share</button>
            {pet.status !== 'adopt' && <Link to={`/match?pet=${pet.id}`} className="btn btn-soft btn-md" style={{ fontSize: 14, borderRadius: 12 }}><Icon name="image_search" size={19} />Find similar by photo</Link>}
          </div>

          <div className="card row" style={{ gap: 14, padding: 16, borderRadius: 18 }}>
            <div style={{ width: 52, height: 52, borderRadius: 14, background: 'var(--teal)', color: '#fff', display: 'grid', placeItems: 'center', fontWeight: 800, fontSize: 16, flexShrink: 0 }}>{initials(pet.contact_name)}</div>
            <div className="grow">
              <div style={{ fontSize: 12, color: 'var(--muted)', fontWeight: 600 }}>{isLost ? 'Owner' : isFound ? (pet.contact_role === 'Rescue group' ? 'Rescue group' : 'Finder') : 'Shelter'}</div>
              <div style={{ fontSize: 16, fontWeight: 800 }}>{pet.contact_name}</div>
              <div className="row" style={{ gap: 4, fontSize: 13, color: 'var(--teal)', fontWeight: 600 }}><Icon name="verified" size={16} />{pet.contact_note}</div>
            </div>
          </div>

          {pet.ai_breeds.length > 0 && (
            <div className="card" style={{ padding: 18, borderRadius: 18 }}>
              <div className="wrap-row between" style={{ alignItems: 'center', gap: 8, marginBottom: 14 }}>
                <div className="row" style={{ gap: 8, fontSize: 15, fontWeight: 800 }}><Icon name="auto_awesome" size={20} style={{ color: 'var(--teal)' }} />AI breed estimate</div>
                <span className="demo-tag" style={{ padding: '3px 7px' }}>DEMO RESULT · M1 BREED CNN</span>
              </div>
              <BreedBars breeds={pet.ai_breeds} />
              <div className="wrap-row" style={{ gap: 8, alignItems: 'center', marginTop: 14 }}>
                <span style={{ padding: '4px 9px', borderRadius: 999, background: conf.bg, color: conf.color, fontSize: 12, fontWeight: 700 }}>{conf.label}</span>
                <span style={{ fontSize: 12.5, color: 'var(--muted)', lineHeight: 1.45 }}>Estimated from the photo only. Listed breed: {pet.breed || 'not given'}.</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="wrap-row" style={{ gap: 'clamp(24px,3vw,44px)', marginTop: 44, alignItems: 'flex-start' }}>
        <div className="col" style={{ flex: '1.25 1 480px', minWidth: 0, gap: 28 }}>
          <div>
            <h2 className="h-card" style={{ marginBottom: 12 }}>About {pet.name}</h2>
            <p style={{ fontSize: 16, lineHeight: 1.7, color: 'var(--ink-2)', textWrap: 'pretty' }}>{pet.description}</p>
            {pet.tags.length > 0 && <div className="wrap-row" style={{ gap: 8, marginTop: 16 }}>{pet.tags.map((t) => <span key={t} className="chip">{t}</span>)}</div>}
          </div>
          {isLost && (
            <div>
              <div className="wrap-row between" style={{ alignItems: 'center', marginBottom: 14, gap: 12 }}>
                <h2 className="h-card">Sightings</h2>
                {!pet.resolved && !pet.is_mine && <button className="btn-link" style={{ fontSize: 14 }} onClick={() => open('sighting')}><Icon name="add" />Add a sighting</button>}
              </div>
              {pet.sightings.length ? (
                <div className="col">
                  {pet.sightings.map((s) => (
                    <div key={s.id} className="row" style={{ gap: 14, alignItems: 'stretch' }}>
                      <div className="col" style={{ alignItems: 'center' }}><span className="timeline-dot" /><span className="timeline-line" /></div>
                      <div style={{ paddingBottom: 20, minWidth: 0 }}>
                        <div style={{ fontSize: 13, color: 'var(--muted)' }}><span className="mono">{dayTime(s.seen_at ?? s.created_at)}</span> · by {s.by_name}</div>
                        <div style={{ fontSize: 15, fontWeight: 700, margin: '2px 0' }}>{s.place}</div>
                        {s.note && <div style={{ fontSize: 14, color: 'var(--body)', lineHeight: 1.5 }}>{s.note}</div>}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ padding: 20, borderRadius: 16, border: '1px dashed var(--line-dash)', fontSize: 14, color: 'var(--muted)' }}>No sightings reported yet. If you’ve seen {pet.name}, add the place and time.</div>
              )}
            </div>
          )}
          <div className="note"><Icon name="shield" /><span>Never send money before meeting. Ask for photos, vet records or a microchip check, and meet in a public place.</span></div>
        </div>
        <div style={{ flex: '1 1 380px', minWidth: 0 }}>
          <h2 className="h-card" style={{ marginBottom: 12 }}>{locWord}</h2>
          <MapView height={340} center={[pet.lat, pet.lng]} zoom={14} interactive={false}
            areaCircle={{ lat: pet.lat, lng: pet.lng, meters: 650, color: '#D65A31' }}
            pins={[{ kind: 'photo', id: pet.id, lat: pet.lat, lng: pet.lng, photo: img(pet.photos[0], 200, 200), ring: STATUS_DOT[pet.status], big: true }]}>
            <div className="map-overlay map-label">{pet.city} · approx. area</div>
          </MapView>
          <div style={{ fontSize: 13, color: 'var(--muted)', marginTop: 10 }}>Exact addresses are hidden to protect the owner’s privacy.</div>
        </div>
      </div>

      {similar.data && similar.data.length > 0 && (
        <div style={{ marginTop: 64 }}>
          <div className="wrap-row between" style={{ alignItems: 'flex-end', marginBottom: 22, gap: 12 }}>
            <h2 style={{ fontSize: 'clamp(24px,2.4vw,32px)', fontWeight: 800, letterSpacing: '-.03em' }}>Similar pets nearby</h2>
            <Link to={`/pets?species=${pet.species}`} className="btn-link">Browse all<Icon name="arrow_forward" /></Link>
          </div>
          <PetGrid pets={similar.data} min={240} gap={22} />
        </div>
      )}

      {modal && <PetModal kind={modal} pet={pet} onClose={() => setModal(null)} />}
    </main>
  )
}
