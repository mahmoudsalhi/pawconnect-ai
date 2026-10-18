// Typed client for the FastAPI backend. Types mirror backend/app/schemas.py.

export type PetStatus = 'lost' | 'found' | 'adopt'

export interface Questionnaire {
  species: 'Dog' | 'Cat' | 'Any'
  home: 'apartment' | 'house' | 'house_garden'
  children: 'none' | 'older' | 'young'
  other_pets: 'none' | 'dogs' | 'cats' | 'both'
  activity: 'relaxed' | 'moderate' | 'active'
  hours_alone: 'under4' | '4to8' | 'over8'
  experience: 'first' | 'some' | 'experienced'
  grooming: 'minimal' | 'some' | 'happy'
  age: 'young' | 'adult' | 'senior' | 'any'
  size: 'Small' | 'Medium' | 'Large' | 'any'
  ideal: string
}

export interface BreedGuess {
  label: string
  confidence: number
  species: string
}

export interface BreedResult {
  model: string
  predictions: BreedGuess[]
}

export interface LookalikeResult {
  model: string
  results: { pet: Pet; similarity: number }[]
}

export interface BreedInfo {
  model: string
  dataset: string
  breeds: BreedGuess[]
  metrics: { test_photos: number; top1_accuracy: number; top3_accuracy: number; species_accuracy: number; baseline_accuracy: number }
  most_confused: { count: number; true: string; predicted: string }[]
}

export interface CompatResult {
  model: string
  results: { pet: Pet; score: number; reasons: { text: string; good: boolean }[] }[]
}

export interface User {
  id: number
  name: string
  email: string
  account_type: 'owner' | 'adopter' | 'shelter'
  phone: string
  gov: string
  lang: string
  notify_email: boolean
  notify_sms: boolean
  notify_push: boolean
  show_phone: boolean
  min_match_score: number
  shelter_id: number | null
  created_at: string
}

export interface ShelterBrief {
  id: number
  slug: string
  name: string
  reply_note: string
  animal_count: number
}

export interface Pet {
  id: number
  ref: string
  status: PetStatus
  resolved: boolean
  name: string
  species: string
  breed: string
  age_label: string
  age_group: string
  sex: string
  size: string
  color: string
  chip: string
  description: string
  tags: string[]
  traits: string[]
  reward: string | null
  fee: string | null
  vaccinated: boolean | null
  dewormed: boolean | null
  sterilized: boolean | null
  fur_length: string | null
  ai_breeds: [string, number][]
  adopt_speed: number | null
  city: string
  gov: string
  lat: number
  lng: number
  event_date: string
  event_time: string
  days_ago: number
  contact_name: string
  contact_role: string
  contact_phone: string | null
  contact_pref: string
  photos: string[]
  shelter: ShelterBrief | null
  distance_km: number
  is_favorite: boolean
  is_mine: boolean
  views: number
  background_removed: boolean
  created_at: string
}

export interface Sighting {
  id: number
  place: string
  seen_at: string | null
  note: string
  by_name: string
  created_at: string
}

export interface PetDetail extends Pet {
  sightings: Sighting[]
  contact_note: string
}

export interface PetPage {
  items: Pet[]
  total: number
  page: number
  pages: number
  counts: Record<'all' | PetStatus, number>
}

export interface Shelter {
  id: number
  slug: string
  name: string
  city: string
  gov: string
  lat: number
  lng: number
  photo_url: string
  animal_count: number
  since: number
  reply_note: string
  focus: string
  listed: number
}

export interface InsightRow {
  pet_id: number
  name: string
  photo: string
  species: string
  age_label: string
  days_listed: number
  speed_class: number
  tips: string[]
}

export interface Application {
  id: number
  code: string
  pet_id: number
  pet_name: string
  pet_photo: string
  shelter_name: string
  stage: number
  stage_label: string
  next_step: string
  created_at: string
}

export interface Notification {
  id: number
  icon: string
  title: string
  body: string
  link: string
  unread: boolean
  created_at: string
}

export interface ActivityItem {
  id: number
  icon: string
  text: string
  created_at: string
}

// M6: scam check on a received message. The message is always delivered; this only adds a warning.
export interface ScamCheck {
  score: number
  level: 'none' | 'caution' | 'high'
  labels: string[]
  reasons: string[]
}

export interface InboxMessage {
  id: number
  pet_id: number
  pet_name: string
  sender_name: string
  body: string
  read: boolean
  created_at: string
  scam: ScamCheck | null
}

export interface MatchItem {
  id: number
  query: Pet
  candidate: Pet
  score: number
  is_new: boolean
  created_at: string
}

export interface Dashboard {
  reports: { pet: Pet; sightings: number; matches: number }[]
  matches: MatchItem[]
  applications: Application[]
  saved: Pet[]
  notifications: Notification[]
  activity: ActivityItem[]
  unread: number
}

export interface AnalyzeResult {
  model: string
  breeds: [string, number][]
  quality: { url: string; score: number; tip: string }[]
  cutout: string | null
  age?: { group: 'young' | 'adult' | 'senior' | null; confidence: number } | null
  health?: { status: 'healthy' | 'skin_problem' | 'unsure'; condition: string | null; confidence: number } | null
}

export interface MatchRun {
  model: string
  pool: number
  area_label: string
  area_lat: number
  area_lng: number
  signature: number[]
  signature_dim: number
  results: { pet: Pet; score: number; distance_from_area: number }[]
}

export interface AIInfo {
  provider: string
}

export interface Stats {
  lost: number
  found: number
  adopt: number
  reunited: number
  shelters: number
}

// M7 search zones for a lost pet (GET /api/ai/location/:id). Bands are exclusive:
// the 50 band holds the most likely cells adding up to 50%, the 80 band the next ones up to 80%, then 95.
export interface SearchZone {
  share: 50 | 80 | 95
  cells: [number, number][]
}

// Chances from now on, given the pet hasn't been found yet (M7 recovery outlook).
export interface RecoveryOutlook {
  found_alive: number
  found_alive_at_start: number
  within_24h: number
  within_7d: number
  median_hours_if_found: number | null
  beyond_horizon: boolean
  horizon_days: number
  by_lifestyle: { indoor_only: number; outdoor_access: number } | null
}

export interface LocationPrediction {
  model: string
  hours_missing: number
  cell_m: number
  zones: SearchZone[]
  peak: [number, number]
  self_return_share: number
  missing_inputs: string[]
  recovery: RecoveryOutlook | null
}

// AI search assistant (GET /api/ai/assistant/:id): Claude reads the description, M7 runs with what
// it found, Claude writes the plan. status "unavailable" (no API key) / "error" = M7 only.
export interface ReportFacts {
  goes_outside: 'yes' | 'no' | 'unknown'
  temperament: string
  circumstance: string
  collar_or_tag: 'yes' | 'no' | 'unknown'
  understood: string
}

export interface SearchPlan {
  summary: string
  steps: { when: string; where: string; what: string }[]
  poster_fr: { headline: string; text: string }
  poster_ar: { headline: string; text: string }
  share_text: string
}

export interface AssistantAnswer {
  status: 'ok' | 'unavailable' | 'error'
  message: string | null
  model: string | null
  facts: ReportFacts | null
  location: LocationPrediction
  plan: SearchPlan | null
}

export class ApiError extends Error {
  status: number
  errors: Record<string, string>
  constructor(status: number, message: string, errors: Record<string, string> = {}) {
    super(message)
    this.status = status
    this.errors = errors
  }
}

const TOKEN_KEY = 'pc_token'

export const tokenStore = {
  get(): string | null {
    try {
      return localStorage.getItem(TOKEN_KEY) ?? sessionStorage.getItem(TOKEN_KEY)
    } catch {
      return null
    }
  },
  set(token: string, remember: boolean) {
    try {
      this.clear()
      ;(remember ? localStorage : sessionStorage).setItem(TOKEN_KEY, token)
    } catch {
      /* storage unavailable: session works until reload */
    }
  },
  clear() {
    try {
      localStorage.removeItem(TOKEN_KEY)
      sessionStorage.removeItem(TOKEN_KEY)
    } catch {
      /* ignore */
    }
  },
}

let onUnauthorized: (() => void) | null = null
export function setUnauthorizedHandler(fn: () => void) {
  onUnauthorized = fn
}

type Query = Record<string, string | number | boolean | null | undefined>

async function request<T>(method: string, path: string, opts: { body?: unknown; query?: Query; form?: FormData } = {}): Promise<T> {
  const url = new URL(path, window.location.origin)
  for (const [k, v] of Object.entries(opts.query ?? {})) {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v))
  }
  const headers: Record<string, string> = {}
  const token = tokenStore.get()
  if (token) headers.Authorization = `Bearer ${token}`
  let body: BodyInit | undefined
  if (opts.form) body = opts.form
  else if (opts.body !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(opts.body)
  }

  let res: Response
  try {
    res = await fetch(url, { method, headers, body })
  } catch {
    throw new ApiError(0, 'Can’t reach PawConnect right now. Check your connection and try again.')
  }
  if (res.status === 204) return undefined as T
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    if (res.status === 401 && token) onUnauthorized?.()
    const detail = typeof data.detail === 'string' ? data.detail : 'Something went wrong. Please try again.'
    throw new ApiError(res.status, detail, data.errors ?? {})
  }
  return data as T
}

export const api = {
  get: <T>(path: string, query?: Query) => request<T>('GET', path, { query }),
  post: <T>(path: string, body?: unknown) => request<T>('POST', path, { body }),
  put: <T>(path: string, body?: unknown) => request<T>('PUT', path, { body }),
  patch: <T>(path: string, body?: unknown) => request<T>('PATCH', path, { body }),
  del: <T>(path: string) => request<T>('DELETE', path),
  upload: (files: File[]) => {
    const form = new FormData()
    files.forEach((f) => form.append('files', f))
    return request<{ urls: string[] }>('POST', '/api/uploads', { form })
  },
}
