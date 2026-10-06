// Overview: what LoanLens is, how well it works and how to use it responsibly.
// Every figure is read from GET /schema, so it always matches the deployed model.

import { PRODUCT_NAME } from "./Header.jsx";

const perHundred = (value) => Math.round(value * 100);

const BANDS = [
  { band: "Low", action: "Standard processing." },
  { band: "Medium", action: "Standard review; check any missing documents." },
  { band: "High", action: "Refer to a senior credit officer; verify income; consider a guarantor." },
];

const STEPS = [
  { title: "Enter the application", body: "Type in the key details from the form. Add anything else you know; blanks are fine." },
  { title: "Read the risk band", body: "Low, Medium or High, with how often past applicants in that band defaulted." },
  { title: "Check the reasons", body: "See which parts of the application raised or lowered the risk, then decide." },
];

export default function OverviewPage({ schema }) {
  const { performance, risk_bands: bands } = schema;
  const lift = performance.recall / performance.share_flagged;
  const stats = [
    { value: `${perHundred(performance.recall)} in 100`, label: "defaulters caught on past applications the model had never seen" },
    { value: `${perHundred(performance.share_flagged)} in 100`, label: "applications sent for a closer review to catch them" },
    { value: `${lift.toFixed(1)}×`, label: "more defaulters found than checking the same number at random" },
    { value: schema.training_rows.toLocaleString(), label: "past vehicle-loan applications it learned from" },
  ];

  return (
    <div className="overview">
      <section className="hero">
        <div className="hero-text">
          <p className="eyebrow">Vehicle-loan default risk</p>
          <h1>
            See the risk in a loan <span className="hero-accent">before the money is paid out.</span>
          </h1>
          <p className="lead">
            {PRODUCT_NAME} compares a new vehicle-loan application with tens of thousands of past ones, shows how risky it
            looks, explains why, and suggests the next step. The decision always stays with you.
          </p>
          <div className="hero-actions">
            <a className="button" href="#/">
              Start a new application
            </a>
            <a className="button secondary" href="#/how-to-use">
              How to use it
            </a>
          </div>
        </div>
        <div className="stat-grid">
          {stats.map((stat) => (
            <div className="stat-card" key={stat.label}>
              <span className="stat-value">{stat.value}</span>
              <span className="stat-label">{stat.label}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <p className="eyebrow">How it works</p>
        <h2>Three steps, under a minute</h2>
        <ol className="step-cards">
          {STEPS.map((step, i) => (
            <li className="step-card" key={step.title}>
              <span className="step-number" aria-hidden="true">
                {i + 1}
              </span>
              <h3>{step.title}</h3>
              <p>{step.body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="panel">
        <p className="eyebrow">Risk bands</p>
        <h2>Every result comes with a clear next step</h2>
        <div className="band-cards">
          {BANDS.map(({ band, action }) => {
            const outcome = bands.outcomes?.[band];
            return (
              <div className={`band-card band-card-${band.toLowerCase()}`} key={band}>
                <span className={`badge badge-${band.toLowerCase()}`}>{band} risk</span>
                {outcome && (
                  <p className="band-fact">
                    <strong>About {perHundred(outcome.default_rate)} in 100</strong> past applicants in this band
                    defaulted ({perHundred(outcome.share_of_applicants)} in 100 applicants fall here).
                  </p>
                )}
                <p className="band-action">{action}</p>
              </div>
            );
          })}
        </div>
      </section>

      <section className="panel responsible">
        <p className="eyebrow">Responsible use</p>
        <h2>Built to support officers, not replace them</h2>
        <ul className="check-list">
          <li>A person makes every decision. {PRODUCT_NAME} flags applications for review; it never refuses one.</li>
          <li>Every result shows its main reasons, so a flag can be questioned and explained to the applicant.</li>
          <li>
            About {perHundred(1 - performance.precision)} in 100 flagged applicants still repay, so a flag means{" "}
            <em>check more closely</em>, not <em>refuse</em>.
          </li>
          <li>Nothing you enter is stored on the server.</li>
        </ul>
      </section>
    </div>
  );
}
