// The result of one check: what the model says, where the risk score sits
// against the band cut-offs, and what the loan officer should do next.
// The band is always shown as text, so colour is never the only signal.

const BAND_TEXT = {
  Low: "Low risk",
  Medium: "Medium risk",
  High: "High risk",
};

function explanation(result, bands) {
  const low = bands.low_upper_bound.toFixed(2);
  const high = result.threshold.toFixed(2);
  if (result.risk_band === "High") {
    return `The risk score is at or above ${high}, the level at which the model flags an application for review.`;
  }
  if (result.risk_band === "Medium") {
    return `The risk score is between ${low} and ${high}: not flagged, but above the low-risk range.`;
  }
  return `The risk score is below ${low}, the low-risk range.`;
}

function ScoreScale({ score, bands }) {
  const low = bands.low_upper_bound * 100;
  const high = bands.high_lower_bound * 100;
  return (
    <div className="scale" aria-hidden="true">
      <div className="scale-track">
        <span className="scale-zone zone-low" style={{ width: `${low}%` }} />
        <span className="scale-zone zone-medium" style={{ width: `${high - low}%` }} />
        <span className="scale-zone zone-high" style={{ width: `${100 - high}%` }} />
        <span className="scale-marker" style={{ left: `${score * 100}%` }} />
      </div>
      <div className="scale-labels">
        <span>0</span>
        <span style={{ left: `${low}%` }}>{bands.low_upper_bound.toFixed(2)}</span>
        <span style={{ left: `${high}%` }}>{bands.high_lower_bound.toFixed(2)}</span>
        <span>1</span>
      </div>
    </div>
  );
}

export default function ResultPanel({ result, bands }) {
  const band = result.risk_band.toLowerCase();
  const largest = Math.max(...result.top_factors.map((factor) => Math.abs(factor.impact)), 1e-9);
  return (
    <section className={`result result-${band}`} aria-live="polite" aria-labelledby="result-title">
      <div className="result-header">
        <h2 id="result-title">{result.prediction}</h2>
        <span className={`badge badge-${band}`}>{BAND_TEXT[result.risk_band]}</span>
      </div>

      <p className="score-line">
        Risk score <strong>{result.risk_score.toFixed(2)}</strong>
        {result.flagged ? " - flagged for review" : " - not flagged"}
      </p>
      <ScoreScale score={result.risk_score} bands={bands} />
      <p>{explanation(result, bands)}</p>

      <h3>Suggested action</h3>
      <p className="action">{result.suggested_action}</p>

      {result.top_factors.length > 0 && (
        <>
          <h3>Main factors behind this score</h3>
          <ul className="factors">
            {result.top_factors.map((factor) => {
              const share = Math.abs(factor.impact) / largest;
              const raises = factor.impact > 0;
              return (
                <li key={factor.factor}>
                  <span className="factor-name">{factor.factor}</span>
                  <span className="factor-bar-track" aria-hidden="true">
                    <span
                      className={`factor-bar ${raises ? "raises" : "lowers"}`}
                      style={{ width: `${Math.max(share * 100, 4)}%` }}
                    />
                  </span>
                  <span className={`factor-effect ${raises ? "raises" : "lowers"}`}>
                    {raises ? "▲ raises risk" : "▼ lowers risk"}
                  </span>
                </li>
              );
            })}
          </ul>
          <p className="result-note">
            Factors are the parts of the application that moved this score most, compared with an average applicant.
            They explain the model, not the causes of default.
          </p>
        </>
      )}

      <p className="result-note">
        The risk score ranks applicants from lower to higher risk; it is not the chance of default. Model{" "}
        {result.model_version}.
      </p>
    </section>
  );
}
