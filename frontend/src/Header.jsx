// Page header: disclaimer banner, logo and product name, navigation and model status.

export const PRODUCT_NAME = "LoanLens";
export const TAGLINE = "Risk checks for vehicle-loan officers";

export const PAGES = [
  { path: "/overview", label: "Overview" },
  { path: "/", label: "New application" },
  { path: "/how-to-use", label: "How to use" },
];

export function Logo({ size = 40 }) {
  return (
    <svg className="logo" width={size} height={size} viewBox="0 0 64 64" aria-hidden="true" focusable="false">
      <rect width="64" height="64" rx="14" fill="#0E4C7A" />
      <path
        d="M14 38l4-11a5 5 0 0 1 4.7-3.3h14.6a5 5 0 0 1 4.7 3.3l4 11"
        fill="none"
        stroke="#ffffff"
        strokeWidth="3.5"
        strokeLinejoin="round"
      />
      <rect x="11" y="37" width="38" height="10" rx="3" fill="#ffffff" />
      <circle cx="19" cy="48" r="4" fill="#0E4C7A" stroke="#ffffff" strokeWidth="2.5" />
      <circle cx="41" cy="48" r="4" fill="#0E4C7A" stroke="#ffffff" strokeWidth="2.5" />
      <path
        d="M44 8l11 4.5v8.5c0 7-4.7 11.7-11 14-6.3-2.3-11-7-11-14v-8.5z"
        fill="#2FA36B"
        stroke="#ffffff"
        strokeWidth="2.5"
        strokeLinejoin="round"
      />
      <path
        d="M39 21l3.6 3.6L49.5 17.5"
        fill="none"
        stroke="#ffffff"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ModelStatus({ health, loadError }) {
  if (health) {
    const version = health.model_version.match(/\(([^)]+)\)/)?.[1] ?? health.model_version;
    return (
      <span className="status status-online" title={`Model ${health.model_version}`}>
        <span className="status-dot" aria-hidden="true" />
        Model online · {version}
      </span>
    );
  }
  if (loadError) {
    return (
      <span className="status status-offline" title={loadError}>
        <span className="status-dot" aria-hidden="true" />
        Model offline
      </span>
    );
  }
  return (
    <span className="status">
      <span className="status-dot" aria-hidden="true" />
      Connecting...
    </span>
  );
}

export default function Header({ route, health, loadError }) {
  return (
    <header className="site-header">
      <div className="top-banner" role="note">
        <strong>Academic prototype.</strong> <span className="banner-long">Trained on historical vehicle-loan data. </span>
        Decision support only: never use it to refuse an application automatically.
      </div>
      <div className="site-header-inner">
        <a className="brand" href="#/">
          <Logo />
          <span className="brand-text">
            <span className="brand-name">{PRODUCT_NAME}</span>
            <span className="brand-tagline">{TAGLINE}</span>
          </span>
        </a>
        <nav aria-label="Main">
          {PAGES.map((page) => (
            <a
              key={page.path}
              href={`#${page.path}`}
              className={route === page.path ? "active" : undefined}
              aria-current={route === page.path ? "page" : undefined}
            >
              {page.label}
            </a>
          ))}
        </nav>
        <ModelStatus health={health} loadError={loadError} />
      </div>
    </header>
  );
}
