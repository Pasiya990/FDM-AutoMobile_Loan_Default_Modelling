// The result of one check: a gauge with the risk score, the verdict and band,
// how often past applicants in that band defaulted, the next step, and the main
// reasons. The band is always shown in words, so colour is never the only signal.

import { useState } from "react";
import { PRODUCT_NAME } from "./Header.jsx";

const BAND_TEXT = { Low: "Low risk", Medium: "Medium risk", High: "High risk" };

// A short checklist per band; the officer ticks what they have done
const NEXT_STEPS = {
  Low: ["Continue standard processing", "Check identity and income documents as usual"],
  Medium: ["Verify income documents", "Check for missing documents", "Note anything unusual in the file"],
  High: ["Refer to a senior credit officer", "Verify income", "Consider a guarantor or a larger down-payment"],
};
const DECISIONS = ["Proceed", "Escalate", "More documents"];

function nextSteps(band, missingScores) {
  const steps = [...NEXT_STEPS[band]];
  if (missingScores > 0) steps.push("Request the missing credit-bureau scores");
  return steps;
}

// ---------------------------------------------------------------- gauge
const CX = 100;
const CY = 100;
const R = 78;

function point(t, radius = R) {
  const angle = Math.PI * (1 - t);
  return [CX + radius * Math.cos(angle), CY - radius * Math.sin(angle)];
}

function arc(from, to) {
  const [x1, y1] = point(from);
  const [x2, y2] = point(to);
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${R} ${R} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

function Gauge({ score, bands, band }) {
  const low = bands.low_upper_bound;
  const high = bands.high_lower_bound;
  const [nx, ny] = point(Math.min(Math.max(score, 0), 1), R - 20);
  const [tx1, ty1] = point(high, R - 13);
  const [tx2, ty2] = point(high, R + 13);
  const label = `Risk score ${score.toFixed(2)} out of 1, ${BAND_TEXT[band].toLowerCase()}. Applications at or above ${high.toFixed(2)} are flagged for review.`;
  return (
    <svg className="gauge" viewBox="0 0 200 152" role="img" aria-label={label}>
      <path d={arc(0, low)} className="gauge-zone gauge-low" />
      <path d={arc(low, high)} className="gauge-zone gauge-medium" />
      <path d={arc(high, 1)} className="gauge-zone gauge-high" />
      <line x1={tx1} y1={ty1} x2={tx2} y2={ty2} className="gauge-tick" />
      <line x1={CX} y1={CY} x2={nx} y2={ny} className="gauge-needle" />
      <circle cx={CX} cy={CY} r="6" className="gauge-hub" />
      <text x={CX} y={CY + 36} className="gauge-score" textAnchor="middle">
        {score.toFixed(2)}
      </text>
      <text x={CX} y={CY + 51} className="gauge-caption" textAnchor="middle">
        risk score
      </text>
      <text x="22" y={CY + 16} className="gauge-end" textAnchor="middle">
        0
      </text>
      <text x="178" y={CY + 16} className="gauge-end" textAnchor="middle">
        1
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------- text helpers
function pastOutcome(band, bands) {
  const outcome = bands.outcomes?.[band];
  if (!outcome || !outcome.default_rate) return null;
  const perHundred = Math.round(outcome.default_rate * 100);
  return `About ${perHundred} in every 100 past applicants in this band defaulted (roughly 1 in ${Math.round(1 / outcome.default_rate)}).`;
}

function summaryText(result, bands, reference, checkedAt, review) {
  const lines = [
    `${PRODUCT_NAME} risk check${reference ? ` - ${reference}` : ""} - ${checkedAt.toLocaleString()}`,
    `Result: ${result.prediction} (${BAND_TEXT[result.risk_band]}), risk score ${result.risk_score.toFixed(2)}; review line ${result.threshold.toFixed(2)}.`,
  ];
  const outcome = pastOutcome(result.risk_band, bands);
  if (outcome) lines.push(outcome);
  if (result.top_factors.length > 0) {
    lines.push("Main factors:");
    result.top_factors.forEach((f) => lines.push(`- ${f.factor}: ${f.effect}`));
  }
  lines.push(`Suggested action: ${result.suggested_action}`);
  const done = review.steps.filter((s) => review.done[s]);
  if (done.length > 0) lines.push(`Steps done: ${done.join("; ")}`);
  if (review.decision) lines.push(`Officer's decision: ${review.decision}`);
  if (review.note.trim()) lines.push(`Note: ${review.note.trim()}`);
  lines.push("Decision support only; not for automatic refusal.");
  return lines.join("\n");
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Older browsers, or a page opened without a secure context
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(area);
    return ok;
  }
}

// ---------------------------------------------------------------- panel
export default function ResultPanel({ result, bands, details, reference, checkedAt, stale, onNewApplication }) {
  const [copied, setCopied] = useState(null);
  const [done, setDone] = useState({});
  const [decision, setDecision] = useState("");
  const [note, setNote] = useState("");
  const steps = nextSteps(result.risk_band, result.missingScores ?? 0);
  const review = { steps, done, decision, note };
  const band = result.risk_band;
  const key = band.toLowerCase();
  const largest = Math.max(...result.top_factors.map((factor) => Math.abs(factor.impact)), 1e-9);

  async function handleCopy() {
    const ok = await copyText(summaryText(result, bands, reference, checkedAt, review));
    setCopied(ok ? "Summary copied" : "Copy did not work in this browser");
    setTimeout(() => setCopied(null), 2500);
  }

  return (
    <section className={`result result-${key}${stale ? " is-stale" : ""}`} aria-labelledby="result-title">
      {stale && (
        <p className="stale-banner" role="status">
          Details changed since this check. Press <strong>Check risk</strong> to update the result.
        </p>
      )}
      <div className="result-top">
        <p className="eyebrow">Risk assessment</p>
        <p className="result-meta">
          {reference ? `${reference} · ` : ""}
          {checkedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </p>
      </div>

      <div className="verdict">
        <Gauge score={result.risk_score} bands={bands} band={band} />
        <div className="verdict-text" aria-live="polite">
          <span className={`badge badge-${key}`}>{BAND_TEXT[band]}</span>
          <h2 id="result-title">{result.prediction}</h2>
          <p>{result.flagged ? "Flagged for review." : "Not flagged."}</p>
        </div>
      </div>

      <p className="gauge-legend">
        <span className="legend-mark" aria-hidden="true" /> Review line {result.threshold.toFixed(2)}: applications at or
        above it are flagged.
      </p>

      {pastOutcome(band, bands) && <p className="band-outcome">{pastOutcome(band, bands)}</p>}

      <div className={`action-card action-${key}`}>
        <span className="action-label">Suggested action</span>
        <p>{result.suggested_action}</p>
      </div>

      {result.top_factors.length > 0 && (
        <>
          <div className="factors-head">
            <h3>Why this score</h3>
            <span>strongest first</span>
          </div>
          <ul className="factors">
            {result.top_factors.map((factor) => {
              const share = Math.abs(factor.impact) / largest;
              const raises = factor.impact > 0;
              return (
                <li key={factor.factor}>
                  <span className="factor-name">{factor.factor}</span>
                  <span className={`factor-effect ${raises ? "raises" : "lowers"}`}>
                    {raises ? "▲ raises risk" : "▼ lowers risk"}
                  </span>
                  <span className="factor-bar-track" aria-hidden="true">
                    <span
                      className={`factor-bar ${raises ? "raises" : "lowers"}`}
                      style={{ width: `${Math.max(share * 100, 4)}%` }}
                    />
                  </span>
                </li>
              );
            })}
          </ul>
          <p className="result-note">
            The factors that moved this score most, compared with an average applicant. They explain the model, not the
            causes of default.
          </p>
        </>
      )}

      <fieldset className="review-box">
        <legend>Suggested next steps</legend>
        {steps.map((step) => (
          <label className="check-item" key={step}>
            <input type="checkbox" checked={Boolean(done[step])} onChange={(e) => setDone({ ...done, [step]: e.target.checked })} />
            <span>{step}</span>
          </label>
        ))}
      </fieldset>

      <fieldset className="review-box">
        <legend>Your decision</legend>
        <div className="decision-options" role="radiogroup">
          {DECISIONS.map((option) => (
            <label key={option} className={`decision${decision === option ? " selected" : ""}`}>
              <input type="radio" name="decision" value={option} checked={decision === option} onChange={() => setDecision(option)} />
              {option}
            </label>
          ))}
        </div>
        <label className="note-label" htmlFor="review-note">
          Note for the file (printed and copied with the summary)
        </label>
        <textarea id="review-note" rows={2} value={note} maxLength={500} onChange={(e) => setNote(e.target.value)} />
      </fieldset>

      <div className="print-only">
        <h3>Application details entered</h3>
        <dl className="print-details">
          {details.map(({ label, value }) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
        <h3>Officer review</h3>
        <ul className="print-review">
          {steps.map((step) => (
            <li key={step}>
              {done[step] ? "[x]" : "[ ]"} {step}
            </li>
          ))}
          <li>Decision: {decision || "not recorded"}</li>
          {note.trim() && <li>Note: {note.trim()}</li>}
        </ul>
        <p className="result-note">
          Decision support only: this result must not be used to refuse an application automatically. Printed{" "}
          {new Date().toLocaleString()}.
        </p>
      </div>

      <div className="result-actions">
        <button type="button" className="secondary" onClick={handleCopy}>
          Copy summary
        </button>
        <button type="button" className="secondary" onClick={() => window.print()}>
          Print
        </button>
        <button type="button" onClick={onNewApplication}>
          New application
        </button>
      </div>
      <p className="copy-status" role="status">
        {copied}
      </p>

      <p className="result-note">
        The risk score ranks applicants from lower to higher risk; it is not the chance of default. Model{" "}
        {result.model_version}.
      </p>
    </section>
  );
}
