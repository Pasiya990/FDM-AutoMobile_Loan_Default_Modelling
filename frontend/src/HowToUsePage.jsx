// "How to use": a quick guide in four steps, what each band means, and answers
// to the questions officers ask most. Every figure comes from GET /schema.

import { FIELDS } from "./fields.js";
import { PRODUCT_NAME } from "./Header.jsx";

const perHundred = (value) => Math.round(value * 100);

const BAND_ACTIONS = {
  Low: "Standard processing.",
  Medium: "Standard review; check any missing documents.",
  High: "Refer to a senior credit officer; verify income; consider a guarantor or a larger down-payment.",
};

function steps(required) {
  return [
    {
      title: "Enter the application",
      body: `Fill in the key details (${required.join(", ").toLowerCase()}). Add anything else you know. Not sure? Load one of the example applicants.`,
    },
    {
      title: "Check the risk",
      body: "Press Check risk. If something is missing or out of range, a red box lists each problem; click one to jump to it.",
    },
    {
      title: "Read the result",
      body: "The gauge shows the risk score and band. Why this score lists what pushed the risk up or down for this applicant.",
    },
    {
      title: "Act on the band",
      body: "Follow the suggested action: standard processing, a standard review, or a senior officer's review. The decision is yours.",
    },
  ];
}

function questions(schema) {
  const { performance, risk_bands: bands } = schema;
  const low = bands.low_upper_bound.toFixed(2);
  const high = bands.high_lower_bound.toFixed(2);
  return [
    {
      q: "What does the risk score mean?",
      a: `It ranks applicants from lower (0) to higher (1) risk compared with past applicants. It is not a percentage chance of default. Scores below ${low} are Low risk, ${low} to ${high} Medium, and ${high} or more High risk, which is the review line.`,
    },
    {
      q: "Why can an applicant be Medium but \"not likely to default\"?",
      a: `Only High-risk applications are flagged. Medium means the score is above the low-risk range but below the review line, so a standard review is enough. In the past, about ${perHundred(bands.outcomes?.Medium?.default_rate ?? 0)} in 100 Medium applicants defaulted.`,
    },
    {
      q: "Which details matter most?",
      a: "The three external credit scores carry the most weight. Without them the tool is much less accurate, so fill them in whenever a bureau record exists. The loan size compared with income and the time in the current job also matter.",
    },
    {
      q: "What if I don't know some details?",
      a: "Leave them as \"Not given\". The tool fills them the same way it did during training. Only the key details are required; the more you add, the more reliable the result.",
    },
    {
      q: "An application looks fine but was flagged. Why?",
      a: `Check Why this score: it shows the reasons. Remember that about ${perHundred(1 - performance.precision)} in 100 flagged applicants still repay. A flag means "check more closely", never "refuse".`,
    },
    {
      q: "How reliable is it?",
      a: `On past applications it had never seen, it flagged ${perHundred(performance.share_flagged)} in 100 applicants and caught about ${perHundred(performance.recall)} in every 100 who later defaulted. That is about ${(performance.recall / performance.share_flagged).toFixed(1)} times more than checking the same number at random.`,
    },
    {
      q: "Is it fair to every applicant?",
      a: "Younger applicants and women are flagged more often, partly because they defaulted more often in the historical data. A flag means about the same in every group, but review every flagged application on its own merits.",
    },
    {
      q: "Is the application stored anywhere?",
      a: `No. ${PRODUCT_NAME} scores the application and forgets it. Your reference number never leaves the browser. Use Copy summary or Print to keep a record.`,
    },
    {
      q: "What do the \"days since ... changed\" fields mean?",
      a: "How many days ago the applicant last changed their registration, identity document or phone number. Leave them blank if you do not know.",
    },
  ];
}

export default function HowToUsePage({ schema }) {
  const required = schema.required.map((field) => FIELDS[field]?.label.replace(" (years)", "") ?? field);
  const bands = schema.risk_bands;
  const low = bands.low_upper_bound.toFixed(2);
  const high = bands.high_lower_bound.toFixed(2);
  const bandRows = [
    { band: "Low", scores: `below ${low}` },
    { band: "Medium", scores: `${low} to ${high}` },
    { band: "High", scores: `${high} or more` },
  ];

  return (
    <div className="guide">
      <section className="guide-intro">
        <p className="eyebrow">Quick guide</p>
        <h1>How to use {PRODUCT_NAME}</h1>
        <p className="lead">
          Four steps, under a minute per application. {PRODUCT_NAME} supports your decision; it never approves or refuses an
          application.
        </p>
        <a className="button" href="#/">
          Start a new application
        </a>
      </section>

      <ol className="step-cards guide-steps">
        {steps(required).map((step, i) => (
          <li className="step-card" key={step.title}>
            <span className="step-number" aria-hidden="true">
              {i + 1}
            </span>
            <h3>
              <span className="visually-hidden">Step {i + 1}: </span>
              {step.title}
            </h3>
            <p>{step.body}</p>
          </li>
        ))}
      </ol>

      <section className="panel">
        <p className="eyebrow">Reading the result</p>
        <h2>What each band means</h2>
        <table className="guide-table">
          <thead>
            <tr>
              <th scope="col">Band</th>
              <th scope="col">Risk score</th>
              <th scope="col">Past applicants who defaulted</th>
              <th scope="col">What to do</th>
            </tr>
          </thead>
          <tbody>
            {bandRows.map(({ band, scores }) => {
              const outcome = bands.outcomes?.[band];
              return (
                <tr key={band}>
                  <th scope="row">
                    <span className={`badge badge-${band.toLowerCase()}`}>{band} risk</span>
                  </th>
                  <td data-label="Risk score">{scores}</td>
                  <td data-label="Past applicants who defaulted">{outcome ? `about ${perHundred(outcome.default_rate)} in 100` : "not available"}</td>
                  <td data-label="What to do">{BAND_ACTIONS[band]}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <section className="panel">
        <p className="eyebrow">Questions</p>
        <h2>Common questions</h2>
        <div className="faq">
          {questions(schema).map(({ q, a }) => (
            <details className="faq-item" key={q}>
              <summary>{q}</summary>
              <p>{a}</p>
            </details>
          ))}
        </div>
      </section>

      <section className="panel responsible">
        <p className="eyebrow">Responsible use</p>
        <h2>Keep in mind</h2>
        <ul className="check-list">
          <li>A person makes every decision. Never refuse an application only because it is flagged.</li>
          <li>Results are estimates from historical data of an undocumented source and period; they can be wrong.</li>
          <li>If the result surprises you, read Why this score and check the details you entered.</li>
        </ul>
      </section>
    </div>
  );
}
