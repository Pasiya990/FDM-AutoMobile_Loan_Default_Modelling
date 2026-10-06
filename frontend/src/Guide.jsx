// "How to use" and "About the model": the instructions a loan officer needs,
// with the model's figures taken from GET /schema rather than typed here.

const percent = (value) => `${Math.round(value * 100)}%`;
const perHundred = (value) => Math.round(value * 100);

export function HowToUse() {
  return (
    <section className="guide" aria-labelledby="how-to-use">
      <h2 id="how-to-use">How to use</h2>
      <ol>
        <li>
          Fill in the <strong>key details</strong>. They are the only required fields.
        </li>
        <li>
          Add any other details you know. Leave unknown fields as "Not given"; the model handles missing
          information, but results are more reliable with the external credit scores.
        </li>
        <li>
          Press <strong>Check risk</strong>, then read the risk band, the main factors and the suggested action.
        </li>
        <li>The decision stays with you: the tool flags applications for review, it never refuses them.</li>
      </ol>
    </section>
  );
}

export function AboutModel({ schema }) {
  const { performance, risk_bands: bands } = schema;
  return (
    <details className="guide about">
      <summary>About the model</summary>
      <p>
        The tool compares an application with about {schema.training_rows.toLocaleString()} past vehicle-loan
        applications whose outcome is known, and gives a <strong>risk score</strong> from 0 to 1. The score ranks
        applicants from lower to higher risk; it is not the chance of default.
      </p>
      <ul>
        <li>
          <strong>Low risk</strong>: score below {bands.low_upper_bound.toFixed(2)}.
        </li>
        <li>
          <strong>Medium risk</strong>: score from {bands.low_upper_bound.toFixed(2)} to{" "}
          {bands.high_lower_bound.toFixed(2)}.
        </li>
        <li>
          <strong>High risk</strong>: score of {bands.high_lower_bound.toFixed(2)} or more. These applications are
          flagged for review.
        </li>
      </ul>
      <p>
        On past applications it had not seen before, the tool flagged {percent(performance.share_flagged)} of
        applicants and caught about {perHundred(performance.recall)} in every 100 who later defaulted. About{" "}
        {perHundred(performance.precision)} in every 100 flagged applicants defaulted, so most flagged applicants
        still repay: a flag means "check more closely", not "refuse".
      </p>
      <p>
        <strong>Limits.</strong> The external credit scores carry the most weight, so applicants without them are
        scored less reliably. Some groups (for example younger applicants) are flagged more often, partly because
        they defaulted more often in the historical data. The data comes from a public dataset with an undocumented
        source and period.
      </p>
      <p className="muted">Model: {schema.model_version}</p>
    </details>
  );
}
