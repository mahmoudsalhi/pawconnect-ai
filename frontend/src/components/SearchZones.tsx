import { useMemo } from 'react'
import type { Pet, RecoveryOutlook } from '../lib/api'
import { img, STATUS_DOT } from '../lib/format'
import { useLocationPrediction } from '../lib/queries'
import { MapView, type ZoneBand } from './MapView'
import { Icon, Skeleton } from './ui'

// Three clearly different colours, hottest = search first. Bands are exclusive, so 50 + 80 make the 80% zone.
const BAND_STYLE: Record<number, { color: string; opacity: number; label: string }> = {
  50: { color: '#C81E1E', opacity: 0.7, label: 'Search first' },
  80: { color: '#F97316', opacity: 0.5, label: 'Then here' },
  95: { color: '#FACC15', opacity: 0.42, label: 'Wider area' },
}

function km2(cells: number, cellM: number) {
  const v = (cells * cellM * cellM) / 1e6
  if (v < 0.01) return `${Math.round(v * 1e6).toLocaleString()} m²`
  return `${v.toFixed(v < 0.1 ? 2 : v < 10 ? 1 : 0)} km²`
}

const pct = (v: number) => `${Math.round(v * 100)}%`

function duration(hours: number) {
  if (hours < 1.5) return 'about an hour'
  if (hours < 36) return `about ${Math.round(hours)} hours`
  const d = hours / 24
  return d < 14 ? `about ${Math.round(d)} days` : `about ${Math.round(d / 7)} weeks`
}

function Outlook({ r, name, missing }: { r: RecoveryOutlook; name: string; missing: string }) {
  if (r.beyond_horizon) {
    return (
      <div className="note" style={{ marginBottom: 16 }}>
        <Icon name="hourglass_bottom" />
        <span>{name} has been missing longer than the {r.horizon_days} days this estimate covers. Pets are still found after
          that: keep posters up, check shelters and vets regularly, and keep sharing the report.</span>
      </div>
    )
  }
  const tone = r.found_alive >= 0.6 ? '#176B63' : r.found_alive >= 0.35 ? '#B7791F' : '#C81E1E'
  const tiles: [string, string][] = [
    ['Found in the next 24 hours', pct(r.within_24h)],
    ['Found in the next 7 days', pct(r.within_7d)],
    ['Typical wait from now, if found', r.median_hours_if_found == null ? '—' : duration(r.median_hours_if_found)],
  ]
  return (
    <div className="card" style={{ padding: 18, borderRadius: 18, marginBottom: 16 }}>
      <div className="row" style={{ gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ fontSize: 40, fontWeight: 800, letterSpacing: '-.03em', color: tone, lineHeight: 1 }}>{pct(r.found_alive)}</div>
        <div style={{ flex: '1 1 200px', fontSize: 14, lineHeight: 1.45 }}>
          <strong>Chance {name} is still found alive</strong>
          <div style={{ color: 'var(--muted)' }}>
            Missing {missing} so far. It was {pct(r.found_alive_at_start)} when {name} went missing and goes down a little
            each day without news.
          </div>
        </div>
      </div>
      {r.by_lifestyle && (
        <div style={{ fontSize: 13, color: 'var(--muted)', marginTop: 10, lineHeight: 1.5 }}>
          This is an average, because the report doesn’t say whether {name} goes outside:{' '}
          <strong style={{ color: 'var(--ink)' }}>{pct(r.by_lifestyle.indoor_only)}</strong> if indoor-only,{' '}
          <strong style={{ color: 'var(--ink)' }}>{pct(r.by_lifestyle.outdoor_access)}</strong> if {name} usually goes out.
        </div>
      )}
      <div className="row" style={{ gap: 10, marginTop: 14, flexWrap: 'wrap' }}>
        {tiles.map(([label, value]) => (
          <div key={label} style={{ flex: '1 1 120px', padding: '10px 12px', borderRadius: 12, background: 'var(--peach)' }}>
            <div style={{ fontSize: 12, color: 'var(--muted)' }}>{label}</div>
            <div style={{ fontSize: 18, fontWeight: 800 }}>{value}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

function missingFor(hours: number) {
  if (hours < 24) return `${Math.round(hours)} h`
  const d = hours / 24
  return `${d < 10 ? d.toFixed(1) : Math.round(d)} days`
}

/** M7: where a lost pet is most likely to be now. Shown only for open lost reports. */
export function SearchZones({ pet }: { pet: Pet }) {
  const q = useLocationPrediction(pet.id, pet.status === 'lost' && !pet.resolved)
  const d = q.data

  const view = useMemo(() => {
    if (!d) return null
    const bands: ZoneBand[] = [...d.zones].sort((a, b) => b.share - a.share)
      .map((z) => ({ cells: z.cells, color: BAND_STYLE[z.share].color, opacity: BAND_STYLE[z.share].opacity }))
    // Frame the "search first" zone (zoom out for the rest); a sample of its cells is enough for the bounds.
    const inner = d.zones.filter((z) => z.share === 50).flatMap((z) => z.cells)
    const step = Math.max(1, Math.floor(inner.length / 400))
    const fitTo = inner.filter((_, i) => i % step === 0)
    let cum = 0
    const legend = [...d.zones].sort((a, b) => a.share - b.share).map((z) => {
      cum += z.cells.length
      return { share: z.share, area: km2(cum, d.cell_m), ...BAND_STYLE[z.share] }
    })
    return { bands, fitTo: fitTo.length > 1 ? fitTo : undefined, legend }
  }, [d])

  if (q.isError) return null
  const species = pet.species.toLowerCase() === 'cat' ? 'cats' : 'dogs'

  return (
    <div style={{ marginTop: 36 }}>
      <div className="row" style={{ gap: 10, marginBottom: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <h2 className="h-card">Where to search</h2>
        <span className="chip" style={{ fontSize: 12, padding: '4px 10px' }}>
          <Icon name="auto_awesome" size={14} style={{ verticalAlign: '-2px', marginRight: 4 }} />AI estimate · experimental
        </span>
      </div>
      {!d || !view ? (
        <Skeleton height={360} style={{ borderRadius: 18 }} />
      ) : (
        <>
          {d.recovery && <Outlook r={d.recovery} name={pet.name} missing={missingFor(d.hours_missing)} />}
          <MapView height={360} center={[pet.lat, pet.lng]} zoom={14} zones={view.bands} cellM={d.cell_m} fitTo={view.fitTo} maxZoom={16}
            pins={[{ kind: 'photo', id: `zones-${pet.id}`, lat: pet.lat, lng: pet.lng, photo: img(pet.photos[0], 200, 200), ring: STATUS_DOT[pet.status] }]}>
            <div className="map-overlay map-label"><Icon name="schedule" size={16} />Missing {missingFor(d.hours_missing)}</div>
          </MapView>
          <div className="wrap-row" style={{ gap: 8, marginTop: 12 }}>
            {view.legend.map((l) => (
              <span key={l.share} className="chip row" style={{ gap: 8, alignItems: 'center' }}>
                <span style={{ width: 12, height: 12, borderRadius: 3, background: l.color, opacity: Math.min(1, l.opacity + 0.35) }} />
                {l.label} · {l.share}% · {l.area}
              </span>
            ))}
          </div>
          <p style={{ fontSize: 14, color: 'var(--muted)', marginTop: 12, lineHeight: 1.55 }}>
            Each zone holds {pet.name} with that likelihood, based on how lost {species} move through the streets,
            parks and hiding places around where they were last seen, and how long they have been missing.
            {d.self_return_share > 0 && <> About {Math.round(d.self_return_share * 100)}% of similar {species} come home on their own,
            so keep a door or window open and leave out food and something that smells of home.</>}
          </p>
          {d.missing_inputs.length > 0 && (
            <div style={{ fontSize: 13, color: 'var(--muted)', marginTop: 10, lineHeight: 1.5 }}>
              <strong style={{ color: 'var(--ink)' }}>Sharper with:</strong> {d.missing_inputs.join(' · ')}
            </div>
          )}
          <div className="note" style={{ marginTop: 14 }}>
            <Icon name="science" />
            <span>Experimental model trained on simulated cases calibrated to published lost-pet studies, not yet on real
              reunions. Use it to decide where to search and put up posters first. Zones use the same approximate location as
              this page.</span>
          </div>
        </>
      )}
    </div>
  )
}
