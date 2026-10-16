import { useEffect, useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from '../lib/auth'
import { firstName, initials } from '../lib/format'
import { useNotifications } from '../lib/queries'
import { Icon, LogoMark } from './ui'

const NAV = [
  { to: '/', label: 'Home', end: true },
  { to: '/pets', label: 'Find a Pet' },
  { to: '/report', label: 'Report Lost/Found' },
  { to: '/adopt', label: 'Adoption' },
  { to: '/match', label: 'AI Matching' },
  { to: '/dashboard', label: 'Dashboard' },
]

export function Logo({ dark, to = '/' }: { dark?: boolean; to?: string }) {
  return (
    <Link to={to} className="logo" style={dark ? { color: 'var(--cream)' } : undefined} aria-label="PawConnect AI home">
      {!dark && <LogoMark />}
      <span className="logo-word">PawConnect</span>
      <span className="logo-ai" style={dark ? { background: 'var(--orange)' } : undefined}>AI</span>
    </Link>
  )
}

function Header() {
  const { user } = useAuth()
  const { data: notifs } = useNotifications()
  const unread = notifs?.some((n) => n.unread)
  const [menu, setMenu] = useState(false)
  const { pathname } = useLocation()
  useEffect(() => setMenu(false), [pathname])
  return (
    <header className="site-header">
      <div className="wrap">
        <Logo />
        <button className="icon-btn menu-btn" aria-label={menu ? 'Close menu' : 'Open menu'} aria-expanded={menu} onClick={() => setMenu(!menu)}>
          <Icon name={menu ? 'close' : 'menu'} />
        </button>
        <nav className={`main-nav${menu ? ' open' : ''}`} aria-label="Main">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => (isActive ? 'active' : '')}>
              {n.label}
            </NavLink>
          ))}
          {!user && <NavLink to="/register" className="menu-only">Register</NavLink>}
        </nav>
        {user ? (
          <div className="row" style={{ gap: 10, flexShrink: 0 }}>
            <Link to="/dashboard/notifications" className="icon-btn" aria-label={unread ? 'Notifications (unread)' : 'Notifications'} style={{ color: 'var(--ink)' }}>
              <Icon name="notifications" />
              {unread && <span className="unread-dot" />}
            </Link>
            <Link to="/dashboard" className="user-chip">
              <span className="avatar">{initials(user.name)}</span>
              <span className="hide-sm">{firstName(user.name)}</span>
            </Link>
          </div>
        ) : (
          <div className="row" style={{ gap: 8, flexShrink: 0 }}>
            <Link to="/login" className="btn btn-outline" style={{ height: 42, padding: '0 16px' }}>Sign in</Link>
            <Link to="/register" className="btn btn-dark hide-sm" style={{ height: 42, padding: '0 16px' }}>Register</Link>
          </div>
        )}
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="site-footer">
      <div className="wrap">
        <div className="footer-grid">
          <div style={{ gridColumn: 'span 2', minWidth: 0, maxWidth: 360 }}>
            <div style={{ marginBottom: 14 }}><Logo dark /></div>
            <p style={{ fontSize: 14, lineHeight: 1.6 }}>An AI-assisted lost pets and adoption platform for Tunisia. A software engineering project by a team of six students.</p>
          </div>
          <div className="footer-col">
            <div className="footer-head">Platform</div>
            <Link to="/pets">Find a pet</Link>
            <Link to="/report">Report lost or found</Link>
            <Link to="/adopt">Adoption Center</Link>
            <Link to="/match">AI Matching</Link>
          </div>
          <div className="footer-col">
            <div className="footer-head">Project</div>
            <span>Model cards</span><span>Datasets</span><span>The team</span><span>Responsible AI</span>
          </div>
          <div className="footer-col">
            <div className="footer-head">Support</div>
            <span>Safety tips for meetups</span><span>Privacy</span><span>Terms</span><span>contact@pawconnect.tn</span>
          </div>
        </div>
        <div className="wrap-row between" style={{ marginTop: 48, paddingTop: 24, borderTop: '1px solid rgba(255,255,255,.1)', gap: 12, fontSize: 13, color: 'var(--muted-2)' }}>
          <span>© 2026 PawConnect AI · Tunis</span>
          <span>AI results are demo outputs until our models are trained and evaluated.</span>
        </div>
      </div>
    </footer>
  )
}

export function SiteLayout() {
  return (
    <>
      <Header />
      <Outlet />
      <Footer />
    </>
  )
}

/** Scroll to top on route change (keeps position when only the query string changes). */
export function ScrollToTop() {
  const { pathname } = useLocation()
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])
  return null
}
