import { Link } from 'react-router-dom'
import type { Pet } from '../lib/api'
import { fmtKm, img, metaLabel, whenLabel } from '../lib/format'
import { useFavorite } from '../lib/queries'
import { Icon, StatusBadge } from './ui'

export function FavButton({ pet, className = 'round-btn', size }: { pet: Pet; className?: string; size?: number }) {
  const toggle = useFavorite()
  return (
    <button
      type="button"
      className={`${className} fav${pet.is_favorite ? ' on' : ''}`}
      aria-label={pet.is_favorite ? `Remove ${pet.name} from saved pets` : `Save ${pet.name}`}
      aria-pressed={pet.is_favorite}
      onClick={(e) => {
        e.preventDefault()
        e.stopPropagation()
        toggle(pet)
      }}
    >
      <Icon name="favorite" size={size} />
    </button>
  )
}

export function PetCard({ pet }: { pet: Pet }) {
  return (
    <Link to={`/pets/${pet.id}`} className="pet-card">
      <div className="pet-card-media">
        <img src={img(pet.photos[0], 640, 520)} alt={pet.name} loading="lazy" />
        <StatusBadge status={pet.status} resolved={pet.resolved} />
        <FavButton pet={pet} />
        {pet.reward && !pet.resolved && <div className="reward">{pet.reward}</div>}
      </div>
      <div className="pet-card-body">
        <div className="row between" style={{ alignItems: 'baseline', gap: 8 }}>
          <div className="pet-card-name">{pet.name}</div>
          <div className="mono muted nowrap" style={{ fontSize: 12 }}>{fmtKm(pet.distance_km)}</div>
        </div>
        <div className="pet-card-meta">{metaLabel(pet)}</div>
        <div className="pet-card-loc">
          <Icon name="location_on" />
          <span>{pet.city} · {whenLabel(pet)}</span>
        </div>
      </div>
    </Link>
  )
}

export function PetGrid({ pets, min = 250, gap = 24 }: { pets: Pet[]; min?: number; gap?: number }) {
  return (
    <div className="pet-grid" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${min}px, 1fr))`, gap }}>
      {pets.map((p) => <PetCard key={p.id} pet={p} />)}
    </div>
  )
}
