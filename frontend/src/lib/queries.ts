import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  api,
  type Application,
  type Dashboard,
  type InsightRow,
  type LocationPrediction,
  type Notification,
  type Pet,
  type PetDetail,
  type PetPage,
  type Shelter,
  type Stats,
} from './api'
import { useAuth } from './auth'
import { useToast } from './toast'

export type PetQuery = Record<string, string | number | undefined>

export const usePets = (params: PetQuery) =>
  useQuery({ queryKey: ['pets', params], queryFn: () => api.get<PetPage>('/api/pets', params), placeholderData: keepPreviousData })

export const usePet = (id: number) =>
  useQuery({ queryKey: ['pet', id], queryFn: () => api.get<PetDetail>(`/api/pets/${id}`), enabled: Number.isFinite(id) })

export const useSimilar = (id: number) =>
  useQuery({ queryKey: ['similar', id], queryFn: () => api.get<Pet[]>(`/api/pets/${id}/similar`), enabled: Number.isFinite(id) })

// M7 search zones. Only asked for open lost reports; 4xx/503 answers mean "not available", so no retries.
export const useLocationPrediction = (id: number, enabled: boolean) =>
  useQuery({
    queryKey: ['location', id],
    queryFn: () => api.get<LocationPrediction>(`/api/ai/location/${id}`),
    enabled: enabled && Number.isFinite(id),
    retry: false,
    staleTime: 30 * 60_000,
  })

export const useShelters = () => useQuery({ queryKey: ['shelters'], queryFn: () => api.get<Shelter[]>('/api/shelters'), staleTime: 300_000 })

export const useInsights = (slug: string) =>
  useQuery({ queryKey: ['insights', slug], queryFn: () => api.get<InsightRow[]>(`/api/shelters/${slug}/insights`) })

export const useStats = () => useQuery({ queryKey: ['stats'], queryFn: () => api.get<Stats>('/api/stats'), staleTime: 60_000 })

export function useDashboard() {
  const { user } = useAuth()
  return useQuery({ queryKey: ['dashboard'], queryFn: () => api.get<Dashboard>('/api/me/dashboard'), enabled: !!user })
}

export function useNotifications() {
  const { user } = useAuth()
  return useQuery({
    queryKey: ['notifications'],
    queryFn: () => api.get<Notification[]>('/api/me/notifications'),
    enabled: !!user,
    refetchInterval: 60_000,
  })
}

export function useMyApplications() {
  const { user } = useAuth()
  return useQuery({ queryKey: ['applications'], queryFn: () => api.get<Application[]>('/api/adoptions/mine'), enabled: !!user })
}

/** Send signed-out visitors to sign in, then bring them back here. */
export function useRequireAuth() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  return useCallback(
    (note = 'Sign in to continue.') => {
      if (user) return true
      navigate('/login', { state: { from: location.pathname + location.search, note } })
      return false
    },
    [user, navigate, location],
  )
}

export function useFavorite() {
  const qc = useQueryClient()
  const toast = useToast()
  const requireAuth = useRequireAuth()
  const m = useMutation({
    mutationFn: (p: Pet) => (p.is_favorite ? api.del(`/api/me/favorites/${p.id}`) : api.put(`/api/me/favorites/${p.id}`)),
    onSuccess: (_, p) => {
      toast(p.is_favorite ? 'Removed from saved pets' : 'Saved. Find it in your dashboard')
      qc.invalidateQueries({ queryKey: ['pets'] })
      qc.invalidateQueries({ queryKey: ['pet', p.id] })
      qc.invalidateQueries({ queryKey: ['similar'] })
      qc.invalidateQueries({ queryKey: ['dashboard'] })
    },
    onError: (e: Error) => toast(e.message, true),
  })
  return useCallback(
    (p: Pet) => {
      if (requireAuth('Sign in to save pets to your dashboard.')) m.mutate(p)
    },
    [m, requireAuth],
  )
}
