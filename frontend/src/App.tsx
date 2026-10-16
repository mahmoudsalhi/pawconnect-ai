import { lazy, Suspense } from 'react'
import { Link, Route, Routes } from 'react-router-dom'
import { ScrollToTop, SiteLayout } from './components/Layout'
import { Empty, Spinner } from './components/ui'
import Home from './pages/Home'

const Auth = lazy(() => import('./pages/Auth'))
const Browse = lazy(() => import('./pages/Browse'))
const PetDetail = lazy(() => import('./pages/PetDetail'))
const Report = lazy(() => import('./pages/Report'))
const Match = lazy(() => import('./pages/Match'))
const Adopt = lazy(() => import('./pages/Adopt'))
const Dashboard = lazy(() => import('./pages/Dashboard'))
const Poster = lazy(() => import('./pages/Poster'))

function Loading() {
  return <div className="row" style={{ minHeight: '60vh', justifyContent: 'center' }}><Spinner variant="teal" /></div>
}

function NotFound() {
  return (
    <main className="wrap" style={{ paddingTop: 60, paddingBottom: 88 }}>
      <Empty icon="pets" title="Page not found" actions={<Link to="/" className="btn btn-dark">Back to home</Link>}>
        The page you’re looking for doesn’t exist or has moved.
      </Empty>
    </main>
  )
}

export default function App() {
  return (
    <div className="app">
      <ScrollToTop />
      <Suspense fallback={<Loading />}>
        <Routes>
          <Route element={<SiteLayout />}>
            <Route index element={<Home />} />
            <Route path="pets" element={<Browse />} />
            <Route path="pets/:id" element={<PetDetail />} />
            <Route path="report" element={<Report />} />
            <Route path="match" element={<Match />} />
            <Route path="adopt" element={<Adopt />} />
            <Route path="*" element={<NotFound />} />
          </Route>
          <Route path="login" element={<Auth mode="login" />} />
          <Route path="register" element={<Auth mode="register" />} />
          <Route path="dashboard/:tab?" element={<Dashboard />} />
          <Route path="pets/:id/poster" element={<Poster />} />
        </Routes>
      </Suspense>
    </div>
  )
}
