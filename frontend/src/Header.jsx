// Page header: logo, product name and navigation between the two pages.

export const PRODUCT_NAME = "Vehicle Loan Risk Check";

export const PAGES = [
  { path: "/", label: "Check an application" },
  { path: "/how-to-use", label: "How to use" },
];

function Logo() {
  return (
    <svg className="logo" viewBox="0 0 64 64" role="img" aria-hidden="true" focusable="false">
      <rect width="64" height="64" rx="14" fill="#1f5f99" />
      <path
        d="M14 38l4-11a5 5 0 0 1 4.7-3.3h14.6a5 5 0 0 1 4.7 3.3l4 11"
        fill="none"
        stroke="#ffffff"
        strokeWidth="3.5"
        strokeLinejoin="round"
      />
      <rect x="11" y="37" width="38" height="10" rx="3" fill="#ffffff" />
      <circle cx="19" cy="48" r="4" fill="#1f5f99" stroke="#ffffff" strokeWidth="2.5" />
      <circle cx="41" cy="48" r="4" fill="#1f5f99" stroke="#ffffff" strokeWidth="2.5" />
      <path
        d="M44 8l11 4.5v8.5c0 7-4.7 11.7-11 14-6.3-2.3-11-7-11-14v-8.5z"
        fill="#2fa36b"
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

export default function Header({ route }) {
  return (
    <header className="site-header">
      <div className="site-header-inner">
        <a className="brand" href="#/">
          <Logo />
          <span>
            <span className="brand-name">{PRODUCT_NAME}</span>
            <span className="brand-tagline">Decision support for vehicle-loan officers</span>
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
      </div>
    </header>
  );
}
