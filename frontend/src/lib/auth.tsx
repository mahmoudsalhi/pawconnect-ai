import { useQueryClient } from '@tanstack/react-query'
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api, setUnauthorizedHandler, tokenStore, type User } from './api'

interface AuthState {
  user: User | null
  ready: boolean
  login: (email: string, password: string, remember: boolean) => Promise<User>
  signInWithToken: (r: TokenResponse, remember: boolean) => User
  register: (data: { name: string; email: string; password: string; account_type: string }) => Promise<User>
  logout: () => void
  setUser: (u: User) => void
}

const AuthContext = createContext<AuthState | null>(null)

interface TokenResponse {
  access_token: string
  user: User
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient()
  const [user, setUser] = useState<User | null>(null)
  const [ready, setReady] = useState(false)

  const logout = useCallback(() => {
    tokenStore.clear()
    setUser(null)
    qc.clear()
  }, [qc])

  useEffect(() => {
    setUnauthorizedHandler(logout)
    if (!tokenStore.get()) {
      setReady(true)
      return
    }
    api.get<User>('/api/auth/me')
      .then(setUser)
      .catch(() => tokenStore.clear())
      .finally(() => setReady(true))
  }, [logout])

  const login = useCallback(async (email: string, password: string, remember: boolean) => {
    const r = await api.post<TokenResponse>('/api/auth/login', { email, password, remember })
    tokenStore.set(r.access_token, remember)
    setUser(r.user)
    qc.invalidateQueries()
    return r.user
  }, [qc])

  const signInWithToken = useCallback((r: TokenResponse, remember: boolean) => {
    tokenStore.set(r.access_token, remember)
    setUser(r.user)
    qc.invalidateQueries()
    return r.user
  }, [qc])

  const register = useCallback(async (data: { name: string; email: string; password: string; account_type: string }) => {
    const r = await api.post<TokenResponse>('/api/auth/register', data)
    tokenStore.set(r.access_token, true)
    setUser(r.user)
    qc.invalidateQueries()
    return r.user
  }, [qc])

  const value = useMemo(() => ({ user, ready, login, signInWithToken, register, logout, setUser }), [user, ready, login, signInWithToken, register, logout])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
