// AI face login: the webcam photos are analysed by the server's face model (YuNet + SFace) and never stored;
// only the face embedding (128 numbers) is kept, after the user agrees.
import { api, type User } from './api'

export interface FaceStatus {
  available: boolean
  enrolled: boolean
  created_at: string | null
  last_used_at: string | null
}

export type TurnDirection = 'left' | 'right'

export const faceStatus = () => api.get<FaceStatus>('/api/face')
export const enrollFace = (images: string[]) => api.post<FaceStatus>('/api/face/enroll', { images, consent: true })
export const deleteFace = () => api.del('/api/face')
export const faceChallenge = (email: string) => api.post<{ direction: TurnDirection; state: string }>('/api/face/login/challenge', { email })
export const faceLogin = (body: { email: string; state: string; front: string; turned: string[]; remember: boolean }) =>
  api.post<{ access_token: string; user: User }>('/api/face/login', body)

export const cameraSupported = () => typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

/** Explain why the camera didn't open. */
export function cameraErrorMessage(e: unknown): string {
  const name = e instanceof Error ? e.name : ''
  if (name === 'NotAllowedError') return 'Camera access was blocked. Allow the camera in your browser’s address bar, then try again.'
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'No camera was found on this device.'
  if (name === 'NotReadableError') return 'The camera is being used by another app. Close it and try again.'
  return 'The camera couldn’t start. Try again, or use your password.'
}
