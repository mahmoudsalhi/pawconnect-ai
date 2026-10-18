import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { useEffect, useMemo, type ReactNode } from 'react'
import { Circle, CircleMarker, MapContainer, Marker, TileLayer, useMap, useMapEvents } from 'react-leaflet'

// OpenStreetMap's tile servers are fine for development. For production, set VITE_MAP_TILES to a
// provider you have an agreement with (MapTiler, Stadia, CARTO with an API key, or self-hosted).
const TILES = import.meta.env.VITE_MAP_TILES ?? 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
const ATTRIBUTION = import.meta.env.VITE_MAP_ATTRIBUTION ?? '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

export type Pin =
  | { kind: 'photo'; id: string | number; lat: number; lng: number; photo: string; ring: string; big?: boolean; onClick?: () => void; z?: number }
  | { kind: 'number'; id: string | number; lat: number; lng: number; label: string; bg: string; size: number; onClick?: () => void; z?: number }
  | { kind: 'drop'; id: string | number; lat: number; lng: number; color: string }

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`)

function iconFor(p: Pin): L.DivIcon {
  if (p.kind === 'photo') {
    const s = p.big ? 50 : 44
    return L.divIcon({
      className: 'pc-icon',
      html: `<div class="pin-photo" style="width:${s}px;height:${s}px;border-color:${p.ring}"><img src="${esc(p.photo)}" alt=""></div><div class="pin-stem" style="background:${p.ring};height:${p.big ? 10 : 9}px"></div>`,
      iconSize: [s, s + 9],
      iconAnchor: [s / 2, s + 9],
    })
  }
  if (p.kind === 'number') {
    return L.divIcon({
      className: 'pc-icon',
      html: `<div class="pin-num" style="background:${p.bg}">${esc(p.label)}</div>`,
      iconSize: [p.size, p.size],
      iconAnchor: [p.size / 2, p.size / 2],
    })
  }
  return L.divIcon({
    className: 'pc-icon',
    html: `<span class="icon fill" style="font-size:46px;color:${p.color};filter:drop-shadow(0 4px 6px rgba(0,0,0,.3))">location_on</span>`,
    iconSize: [46, 46],
    iconAnchor: [23, 44],
  })
}

function Recenter({ center, zoom, bounds, maxZoom = 13 }: { center: [number, number]; zoom: number; bounds?: [number, number][]; maxZoom?: number }) {
  const map = useMap()
  const key = bounds?.length ? JSON.stringify(bounds) : `${center[0]},${center[1]},${zoom}`
  useEffect(() => {
    if (bounds && bounds.length > 1) map.fitBounds(L.latLngBounds(bounds), { padding: [48, 48], maxZoom })
    else map.setView(center, zoom)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, map])
  return null
}

export type ZoneBand = { cells: [number, number][]; color: string; opacity: number }

// Cells of a regular lattice (cellM metres) painted into one image, one pixel per cell, and laid
// over the map. Thousands of separate rectangles would be slow and show hairline seams.
function ZoneCells({ bands, cellM }: { bands: ZoneBand[]; cellM: number }) {
  const map = useMap()
  useEffect(() => {
    const all = bands.flatMap((b) => b.cells)
    if (!all.length) return
    const lat0 = all[0][0]
    const dLat = cellM / 110574
    const dLng = cellM / (111320 * Math.cos((lat0 * Math.PI) / 180))
    let minLat = Infinity, maxLat = -Infinity, minLng = Infinity, maxLng = -Infinity
    for (const [la, ln] of all) {
      minLat = Math.min(minLat, la); maxLat = Math.max(maxLat, la)
      minLng = Math.min(minLng, ln); maxLng = Math.max(maxLng, ln)
    }
    const w = Math.round((maxLng - minLng) / dLng) + 1
    const h = Math.round((maxLat - minLat) / dLat) + 1
    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    for (const band of bands) {
      ctx.fillStyle = band.color
      ctx.globalAlpha = band.opacity
      for (const [la, ln] of band.cells) {
        ctx.fillRect(Math.round((ln - minLng) / dLng), Math.round((maxLat - la) / dLat), 1, 1)
      }
    }
    const overlay = L.imageOverlay(canvas.toDataURL(), [
      [minLat - dLat / 2, minLng - dLng / 2],
      [maxLat + dLat / 2, maxLng + dLng / 2],
    ], { interactive: false, className: 'zone-overlay' }).addTo(map)
    return () => {
      overlay.remove()
    }
  }, [bands, cellM, map])
  return null
}

function ClickCatcher({ onPick }: { onPick: (lat: number, lng: number) => void }) {
  useMapEvents({ click: (e) => onPick(e.latlng.lat, e.latlng.lng) })
  return null
}

export function MapView({
  center,
  zoom,
  pins = [],
  fit,
  height,
  radiusKm,
  areaCircle,
  centerDot,
  onPick,
  interactive = true,
  border,
  cursor,
  zones,
  cellM,
  fitTo,
  maxZoom,
  children,
}: {
  center: [number, number]
  zoom: number
  pins?: Pin[]
  fit?: boolean
  height: number | string
  radiusKm?: number
  areaCircle?: { lat: number; lng: number; meters: number; color: string }
  centerDot?: boolean
  onPick?: (lat: number, lng: number) => void
  interactive?: boolean
  border?: string
  cursor?: string
  /** Search-zone bands, drawn under the pins. */
  zones?: ZoneBand[]
  cellM?: number
  /** Fit the view to these points (overrides `fit`). */
  fitTo?: [number, number][]
  maxZoom?: number
  children?: ReactNode
}) {
  const bounds = useMemo(() => fitTo ?? (fit ? pins.map((p) => [p.lat, p.lng] as [number, number]) : undefined), [fitTo, fit, pins])
  return (
    <div className="map-box" style={{ height, border, cursor }}>
      <MapContainer
        center={center}
        zoom={zoom}
        scrollWheelZoom={false}
        dragging={interactive}
        zoomControl={interactive}
        doubleClickZoom={interactive}
        attributionControl
        style={{ cursor }}
      >
        <TileLayer url={TILES} attribution={ATTRIBUTION} maxZoom={19} />
        <Recenter center={center} zoom={zoom} bounds={bounds} maxZoom={maxZoom} />
        {zones && cellM && <ZoneCells bands={zones} cellM={cellM} />}
        {onPick && <ClickCatcher onPick={onPick} />}
        {radiusKm !== undefined && (
          <Circle center={center} radius={radiusKm * 1000} pathOptions={{ color: '#176B63', weight: 1.5, dashArray: '6 6', fillColor: '#176B63', fillOpacity: 0.08 }} />
        )}
        {areaCircle && (
          <Circle center={[areaCircle.lat, areaCircle.lng]} radius={areaCircle.meters} pathOptions={{ color: areaCircle.color, weight: 1.5, dashArray: '6 6', fillColor: areaCircle.color, fillOpacity: 0.12 }} />
        )}
        {centerDot && <CircleMarker center={center} radius={8} pathOptions={{ color: '#FFFFFF', weight: 3, fillColor: '#D65A31', fillOpacity: 1 }} />}
        {pins.map((p) => (
          <Marker
            key={p.id}
            position={[p.lat, p.lng]}
            icon={iconFor(p)}
            zIndexOffset={'z' in p && p.z ? p.z * 100 : 0}
            eventHandlers={'onClick' in p && p.onClick ? { click: p.onClick } : undefined}
            keyboard={false}
          />
        ))}
      </MapContainer>
      {children}
    </div>
  )
}
