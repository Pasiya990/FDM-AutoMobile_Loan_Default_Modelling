// The "How to use" page: step-by-step instructions, how to read a result, and
// what the model can and cannot do. Every figure comes from GET /schema.

import { FIELDS } from "./fields.js";
import { PRODUCT_NAME } from "./Header.jsx";

const perHundred = (value) => Math.round(value * 100);

function BandTable({ bands }) {
  const low = bands.low_upper_bound.toFixed(2);
  const high = bands.high_lower_bound.toFixed(2);
  const rows = [
    { band: "Low", scores: `below ${low}`, action: "Standard processing." },
    { band: "Medium", scores: `${low} to ${high}`, action: "Standard review; verify any missing documents." },
    {
      band: "High",
      scores: `${high} or more`,
      action: "Flagged: refer to a senior credit officer, verify income, consider a guarantor or larger down-payment.",
    },
  ];
  return (
    <table className="guide-table">
      <thead>
        <tr>
          <th scope="col">Band</th>
          <th scope="col">Risk score</th>
          <th scope="col">Past applicants who defaulted</th>
          <th scope="col">Suggested action</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const outcome = bands.outcomes?.[row.band];
          return (
            <tr key={row.band}>
              <th scope="row">
                <span className={`badge badge-${row.band.toLowerCase()}`}>{row.band} risk</span>
              </th>
              <td>{row.scores}</td>
              <td>{outcome ? `about ${perHundred(outcome.default_rate)} in 100` : "-"}</td>
              <td>{row.action}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default function HowToUsePage({ schema }) {
  const { performance, risk_bands: bands } = schema;
  const required = schema.required.map((field) => FIELDS[field]?.label ?? field);

  return (
    <article className="guide-page">
      <h1>How to use the {PRODUCT_NAME}</h1>
      <p className="lead">
        The tool helps loan officers decide which vehicle-loan applications need a closer look. It compares an
        application with about {schema.training_rows.toLocaleString()} past applications whose outcome is known.
        It supports your decision; it never approves or refuses an application.
      </p>

      <section>
        <h2>1. Enter the application</h2>
        <ol>
          <li>
            Open <a href="#/">Check an application</a>.
          </li>
          <li>
            Fill in the <strong>Key details</strong>: {required.join(", ")}. If the applicant is employed, also give
            their years in the current job.
          </li>
          <li>
            Add any other details you know. Leave unknown answers as "Not given": the model handles missing
            information.
          </li>
          <li>
            Fill in the <strong>external credit scores</strong> whenever a bureau record exists. They are the strongest
            signal the model has, so results without them are less reliable.
          </li>
          <li>
            To see how the tool works first, press one of the <strong>Try an example</strong> buttons. Each fills the
            form with a real past application.
          </li>
        </ol>
      </section>

      <section>
        <h2>2. Check the risk</h2>
        <p>
          Press <strong>Check risk</strong>. If something is missing or out of range, a red box at the top lists every
          problem; click a problem to jump to that field. Fields are also checked as soon as you leave them. Some
          answers switch other fields off: for example, "Not employed or retired" disables the years in job.
        </p>
      </section>

      <section>
        <h2>3. Read the result</h2>
        <ul>
          <li>
            <strong>Risk band</strong>: Low, Medium or High, shown in words and colour.
          </li>
          <li>
            <strong>Risk score</strong> from 0 to 1. It ranks applicants from lower to higher risk; it is not the chance
            of default.
          </li>
          <li>
            <strong>Main factors</strong>: the parts of the application that moved the score most, and whether each
            raised or lowered the risk. They explain the model's score, not the causes of default.
          </li>
          <li>
            <strong>Suggested action</strong> for the band.
          </li>
        </ul>
        <BandTable bands={bands} />
        <p>
          Use <strong>Print summary</strong> to keep a copy of the result and the details entered, and{" "}
          <strong>Start a new application</strong> to clear the form.
        </p>
      </section>

      <section>
        <h2>4. How reliable is it?</h2>
        <p>
          On past applications it had not seen before, the tool flagged {perHundred(performance.share_flagged)} in
          every 100 applicants and caught about {perHundred(performance.recall)} in every 100 who later defaulted.
          About {perHundred(performance.precision)} in every 100 flagged applicants defaulted, so most flagged
          applicants still repay: a flag means <em>check more closely</em>, not <em>refuse</em>.
        </p>
      </section>

      <section>
        <h2>5. Use it responsibly</h2>
        <ul>
          <li>The final decision is always made by a person. Never refuse an application only because it is flagged.</li>
          <li>
            Some groups, for example younger applicants, are flagged more often, partly because they defaulted more
            often in the historical data. Review flagged applications on their merits.
          </li>
          <li>
            The model was trained on a public historical dataset whose source and period are not documented, so it
            may not reflect today's applicants.
          </li>
        </ul>
      </section>

      <p className="muted">Model: {schema.model_version}</p>
    </article>
  );
}
