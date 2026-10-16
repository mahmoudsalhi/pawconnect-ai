import { useEffect, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { MapView } from '../components/MapView'
import { FavButton, PetGrid } from '../components/PetCard'
import { Empty, Icon, Pills, Seg, Skeleton, StatusBadge } from '../components/ui'
import type { Pet } from '../lib/api'
import { fmtKm, GOVERNORATES, img, metaLabel, STATUS_DOT, whenLabel } from '../lib/format'
import { usePets } from '../lib/queries'

const DEFAULTS = { status: 'all', q: '', gov: 'all', species: 'any', age: 'any', size: 'any', breed: '', days: 'any', distance: '300', sort: 'recent', view: 'grid', page: '1', scope: 'tunis' }
type Key = keyof typeof DEFAULTS
const PAGE_SIZE = 8

function useDebounced<T>(value: T, ms = 300) {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), ms)
    return () => window.clearTimeout(t)
  }, [value, ms])
  return v
}

function ListRow({ p }: { p: Pet }) {
  return (
    <Link to={`/pets/${p.id}`} className="card wrap-row" style={{ gap: 18, alignItems: 'center', padding: 12, borderRadius: 18, color: 'var(--ink)' }}>
      <img src={img(p.photos[0], 300, 240)} alt={p.name} style={{ width: 136, height: 112, borderRadius: 12, objectFit: 'cover', flexShrink: 0 }} />
      <div className="col" style={{ flex: '1 1 220px', minWidth: 0, gap: 6 }}>
        <div className="wrap-row" style={{ alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 18, fontWeight: 800 }}>{p.name}</span>
          <StatusBadge status={p.status} small />
        </div>
        <div style={{ fontSize: 14, color: 'var(--body)' }}>{metaLabel(p)} · {p.sex} · {p.color}</div>
        <div className="row" style={{ gap: 4, fontSize: 13, color: 'var(--muted)' }}><Icon name="location_on" size={17} />{p.city}, {p.gov} · {whenLabel(p)}</div>
      </div>
      <div className="row" style={{ gap: 10, paddingRight: 6 }}>
        <span className="mono muted" style={{ fontSize: 13 }}>{fmtKm(p.distance_km)}</span>
        <FavButton pet={p} className="icon-btn" size={20} />
        <span className="btn btn-dark btn-md">View details</span>
      </div>
    </Link>
  )
}

export default function Browse() {
  const [sp, setSp] = useSearchParams()
  const navigate = useNavigate()
  const get = (k: Key) => sp.get(k) ?? DEFAULTS[k]
  const set = (patch: Partial<Record<Key, string>>, resetPage = true) => {
    const next = new URLSearchParams(sp)
    for (const [k, v] of Object.entries({ ...(resetPage ? { page: '1' } : {}), ...patch })) {
      if (v === undefined || v === DEFAULTS[k as Key]) next.delete(k)
      else next.set(k, v)
    }
    setSp(next, { replace: true })
  }

  const [q, setQ] = useState(get('q'))
  const [breed, setBreed] = useState(get('breed'))
  const [distance, setDistance] = useState(Number(get('distance')))
  const dq = useDebounced(q)
  const dBreed = useDebounced(breed)
  const dDist = useDebounced(distance, 200)
  useEffect(() => { if (dq !== get('q')) set({ q: dq }) }, [dq]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (dBreed !== get('breed')) set({ breed: dBreed }) }, [dBreed]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (String(dDist) !== get('distance')) set({ distance: String(dDist) }) }, [dDist]) // eslint-disable-line react-hooks/exhaustive-deps

  const view = get('view')
  const isMap = view === 'map'
  const { data, isFetching, isError } = usePets({
    status: get('status'), q: get('q'), gov: get('gov'), species: get('species'), age: get('age'), size: get('size'),
    breed: get('breed'), days: get('days') === 'any' ? undefined : get('days'), distance: get('distance'), sort: get('sort'),
    page: isMap ? 1 : get('page'), page_size: isMap ? 100 : PAGE_SIZE,
  })
  const [mapSel, setMapSel] = useState<number | null>(null)
  const sel = data?.items.find((p) => p.id === mapSel)
  const page = data?.page ?? 1
  const total = data?.total ?? 0
  const counts = data?.counts

  const clear = () => {
    setQ('')
    setBreed('')
    setDistance(300)
    setSp(new URLSearchParams(view !== 'grid' ? { view } : {}), { replace: true })
  }
  const goPage = (n: number) => {
    set({ page: String(n) }, false)
    window.scrollTo(0, 0)
  }
  const scope = get('scope') === 'all' ? { c: [35.85, 10.35] as [number, number], z: 7 } : { c: [36.84, 10.22] as [number, number], z: 11 }

  return (
    <main className="wrap" style={{ paddingTop: 40, paddingBottom: 88 }}>
      <div className="wrap-row between" style={{ alignItems: 'flex-end', gap: 20, marginBottom: 24 }}>
        <div>
          <h1 className="h-page" style={{ marginBottom: 8 }}>Find a pet</h1>
          <p className="lede">Lost, found and adoptable animals across Tunisia. Distances are from La Marsa.</p>
        </div>
        <Link to="/match" className="btn btn-soft"><Icon name="image_search" />Search by photo instead</Link>
      </div>

      <div className="row" role="tablist" style={{ gap: 4, borderBottom: '1px solid #EADCCD', marginBottom: 20, overflowX: 'auto', scrollbarWidth: 'none' }}>
        {([['all', 'All'], ['lost', 'Lost'], ['found', 'Found'], ['adopt', 'Available for adoption']] as const).map(([v, l]) => {
          const a = get('status') === v
          return (
            <button key={v} role="tab" aria-selected={a} onClick={() => set({ status: v })} className="row" style={{ gap: 8, whiteSpace: 'nowrap', padding: '12px 14px', border: 0, borderBottom: `2.5px solid ${a ? '#D65A31' : 'transparent'}`, marginBottom: -1, background: 'none', color: a ? 'var(--ink)' : 'var(--muted)', fontSize: 15, fontWeight: 700 }}>
              {l}
              <span className="mono" style={{ fontSize: 11, padding: '2px 7px', borderRadius: 999, background: a ? '#202A32' : '#F2E8DD', color: a ? '#fff' : 'var(--muted)' }}>{counts?.[v] ?? '–'}</span>
            </button>
          )
        })}
      </div>

      <div className="card wrap-row" style={{ gap: 10, alignItems: 'center', padding: 8, borderRadius: 16, marginBottom: 24 }}>
        <label className="row" style={{ flex: '3 1 260px', gap: 10, padding: '0 12px', height: 48, minWidth: 0 }}>
          <Icon name="search" size={21} style={{ color: 'var(--muted)' }} />
          <input value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search" placeholder="Search by name, breed, colour or neighbourhood" style={{ flex: 1, minWidth: 0, border: 0, outline: 'none', background: 'transparent', fontSize: 15 }} />
        </label>
        <label className="row" style={{ flex: '1 1 170px', gap: 8, padding: '0 12px', height: 48, borderRadius: 12, background: 'var(--cream)' }}>
          <Icon name="location_on" size={20} style={{ color: 'var(--teal)' }} />
          <select value={get('gov')} onChange={(e) => set({ gov: e.target.value })} aria-label="Governorate" style={{ flex: 1, border: 0, outline: 'none', background: 'transparent', fontSize: 14, fontWeight: 600, minWidth: 0 }}>
            <option value="all">All of Tunisia</option>
            {GOVERNORATES.map((g) => <option key={g}>{g}</option>)}
          </select>
        </label>
        <label className="row" style={{ flex: '1 1 170px', gap: 8, padding: '0 12px', height: 48, borderRadius: 12, background: 'var(--cream)' }}>
          <Icon name="sort" size={20} style={{ color: 'var(--muted)' }} />
          <select value={get('sort')} onChange={(e) => set({ sort: e.target.value })} aria-label="Sort" style={{ flex: 1, border: 0, outline: 'none', background: 'transparent', fontSize: 14, fontWeight: 600, minWidth: 0 }}>
            <option value="recent">Most recent</option><option value="distance">Nearest first</option><option value="name">Name A–Z</option>
          </select>
        </label>
        <div className="row" style={{ padding: 4, background: 'var(--cream)', borderRadius: 12, gap: 2 }}>
          {([['grid', 'grid_view', 'Grid'], ['list', 'view_list', 'List'], ['map', 'map', 'Map']] as const).map(([v, i, l]) => (
            <button key={v} onClick={() => set({ view: v }, false)} aria-pressed={view === v} className="row" style={{ gap: 6, height: 40, padding: '0 12px', border: 0, borderRadius: 9, background: view === v ? '#202A32' : 'transparent', color: view === v ? '#fff' : 'var(--muted)', fontSize: 13, fontWeight: 700 }}>
              <Icon name={i} size={19} />{l}
            </button>
          ))}
        </div>
      </div>

      <div className="wrap-row" style={{ gap: 28, alignItems: 'flex-start' }}>
        <aside className="card col" style={{ flex: '1 1 250px', maxWidth: '100%', gap: 22, padding: 22 }}>
          <div className="row between">
            <span style={{ fontSize: 16, fontWeight: 800 }}>Filters</span>
            <button className="btn-link" style={{ fontSize: 13 }} onClick={clear}>Reset all</button>
          </div>
          <div><div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>Species</div>
            <Pills value={get('species')} onChange={(v) => set({ species: v })} options={[{ value: 'any', label: 'Any' }, { value: 'Dog', label: 'Dogs' }, { value: 'Cat', label: 'Cats' }]} /></div>
          <div><div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>Age</div>
            <Pills value={get('age')} onChange={(v) => set({ age: v })} options={[{ value: 'any', label: 'Any' }, { value: 'young', label: 'Puppy / kitten' }, { value: 'adult', label: 'Adult' }, { value: 'senior', label: 'Senior' }]} /></div>
          <div><div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>Size</div>
            <Pills value={get('size')} onChange={(v) => set({ size: v })} options={[{ value: 'any', label: 'Any' }, { value: 'Small', label: 'Small' }, { value: 'Medium', label: 'Medium' }, { value: 'Large', label: 'Large' }]} /></div>
          <label className="field">
            <span className="field-label" style={{ marginBottom: 10 }}>Breed</span>
            <input className="input" style={{ height: 44, fontSize: 14, borderWidth: 1 }} value={breed} onChange={(e) => setBreed(e.target.value)} placeholder="e.g. Beagle, tabby, spaniel" />
          </label>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>Date posted</div>
            <div className="radio-list" role="radiogroup">
              {([['any', 'Any time'], ['1', 'Last 24 hours'], ['7', 'Last 7 days'], ['30', 'Last 30 days']] as const).map(([v, l]) => (
                <button key={v} role="radio" aria-checked={get('days') === v} className={`radio${get('days') === v ? ' on' : ''}`} onClick={() => set({ days: v })}><span className="dot" />{l}</button>
              ))}
            </div>
          </div>
          <div>
            <div className="row between" style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>
              <span>Distance</span><span className="mono" style={{ fontWeight: 500, color: 'var(--teal)' }}>{distance >= 300 ? 'Any distance' : `Within ${distance} km`}</span>
            </div>
            <input type="range" min={1} max={300} value={distance} onChange={(e) => setDistance(+e.target.value)} aria-label="Distance" style={{ width: '100%' }} />
            <div className="row between mono muted" style={{ fontSize: 11, marginTop: 4 }}><span>1 km</span><span>Any</span></div>
          </div>
        </aside>

        <section className="col" style={{ flex: '999 1 560px', minWidth: 0, gap: 18, opacity: isFetching && data ? 0.7 : 1, transition: 'opacity .2s' }}>
          <div style={{ fontSize: 14, color: 'var(--muted)' }}>
            {!data ? 'Loading…' : total ? (isMap ? `${total} on the map` : `Showing ${(page - 1) * PAGE_SIZE + 1}–${Math.min(page * PAGE_SIZE, total)} of ${total}`) : 'No results'}
          </div>
          {isError && <div className="alert"><Icon name="error" />Couldn’t load pets. Check that the API is running and try again.</div>}
          {!data && !isError && <div className="pet-grid" style={{ gridTemplateColumns: 'repeat(auto-fill,minmax(230px,1fr))', gap: 20 }}>{[0, 1, 2, 3, 4, 5].map((i) => <Skeleton key={i} height={350} />)}</div>}

          {data && total > 0 && view === 'grid' && <PetGrid pets={data.items} min={230} gap={20} />}
          {data && total > 0 && view === 'list' && <div className="col" style={{ gap: 12 }}>{data.items.map((p) => <ListRow key={p.id} p={p} />)}</div>}
          {data && total > 0 && isMap && (
            <div className="wrap-row" style={{ gap: 16 }}>
              <div style={{ flex: '2 1 420px', minWidth: 0 }}>
                <MapView height={620} center={scope.c} zoom={scope.z}
                  pins={data.items.map((p) => ({ kind: 'photo' as const, id: p.id, lat: p.lat, lng: p.lng, photo: img(p.photos[0], 200, 200), ring: p.id === mapSel ? '#202A32' : STATUS_DOT[p.status], z: p.id === mapSel ? 5 : 1, onClick: () => setMapSel(p.id) }))}>
                  <div className="map-overlay" style={{ top: 14, left: 54 }}>
                    <Seg className="dark-thumb" style={{ background: 'rgba(255,255,255,.95)', boxShadow: '0 6px 16px rgba(32,42,50,.15)', gap: 2 }} value={get('scope')} onChange={(v) => set({ scope: v }, false)}
                      options={[{ value: 'tunis', label: 'Greater Tunis' }, { value: 'all', label: 'All Tunisia' }]} />
                  </div>
                  {sel && (
                    <button onClick={() => navigate(`/pets/${sel.id}`)} className="map-overlay row" style={{ left: 14, right: 14, bottom: 26, maxWidth: 420, gap: 12, padding: 10, background: '#fff', borderRadius: 16, border: 0, boxShadow: '0 18px 40px -16px rgba(32,42,50,.45)', textAlign: 'left' }}>
                      <img src={img(sel.photos[0], 200, 200)} alt="" style={{ width: 76, height: 76, borderRadius: 12, objectFit: 'cover' }} />
                      <div className="grow">
                        <div className="row" style={{ gap: 8 }}><span style={{ fontSize: 16, fontWeight: 800 }}>{sel.name}</span><StatusBadge status={sel.status} small /></div>
                        <div style={{ fontSize: 13, color: 'var(--muted)', marginTop: 3 }}>{metaLabel(sel)}</div>
                        <div style={{ fontSize: 13, color: 'var(--muted)' }}>{sel.city} · {whenLabel(sel)}</div>
                      </div>
                      <Icon name="chevron_right" size={22} style={{ color: 'var(--muted)' }} />
                    </button>
                  )}
                </MapView>
              </div>
              <div className="col" style={{ flex: '1 1 260px', maxHeight: 620, overflow: 'auto', gap: 8 }}>
                {data.items.slice(0, 30).map((p) => (
                  <button key={p.id} onClick={() => setMapSel(p.id)} onDoubleClick={() => navigate(`/pets/${p.id}`)} className="card row" style={{ gap: 12, padding: 8, borderRadius: 14, textAlign: 'left', borderColor: p.id === mapSel ? '#202A32' : undefined }}>
                    <img src={img(p.photos[0], 200, 200)} alt="" style={{ width: 56, height: 56, borderRadius: 10, objectFit: 'cover' }} />
                    <div className="grow"><div style={{ fontSize: 14, fontWeight: 800 }}>{p.name}</div><div style={{ fontSize: 12, color: 'var(--muted)' }}>{p.status === 'adopt' ? 'For adoption' : p.status === 'lost' ? 'Lost' : 'Found'} · {p.city}</div></div>
                    <span className="mono muted" style={{ fontSize: 12 }}>{fmtKm(p.distance_km)}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {data && total === 0 && (
            <Empty icon="search_off" title="No pets match these filters"
              actions={<><button className="btn btn-dark" onClick={clear}>Clear filters</button><Link to="/match" className="btn btn-outline">Search by photo</Link></>}>
              Try a wider distance, another governorate or fewer filters. If you have a photo, AI Matching can search by appearance instead.
            </Empty>
          )}

          {data && !isMap && data.pages > 1 && (
            <nav className="row" aria-label="Pagination" style={{ justifyContent: 'center', gap: 6, marginTop: 12 }}>
              <button className="icon-btn" aria-label="Previous page" disabled={page <= 1} onClick={() => goPage(page - 1)}><Icon name="chevron_left" size={20} /></button>
              {Array.from({ length: data.pages }, (_, i) => i + 1).map((n) => (
                <button key={n} aria-current={n === page ? 'page' : undefined} onClick={() => goPage(n)} className="mono" style={{ width: 42, height: 42, borderRadius: 12, border: '1px solid var(--line-2)', background: n === page ? '#202A32' : '#fff', color: n === page ? '#fff' : 'var(--ink)', fontSize: 14, fontWeight: 600 }}>{n}</button>
              ))}
              <button className="icon-btn" aria-label="Next page" disabled={page >= data.pages} onClick={() => goPage(page + 1)}><Icon name="chevron_right" size={20} /></button>
            </nav>
          )}
        </section>
      </div>
    </main>
  )
}
