import { useEffect, useState } from "react";
import ErrorSummary from "./ErrorSummary.jsx";
import Field from "./Field.jsx";
import { FIELDS, SECTIONS } from "./fields.js";
import { filledCount, isDisabled, keyProgress } from "./formValues.js";

// Only the key details stay open all the time; the other sections can be folded away
const startsOpen = (section) => section.id === "key" || section.id === "bureau";
const SCORE_FIELDS = ["score_source_1", "score_source_2", "score_source_3"];

const toNumber = (raw) => (raw === undefined || raw === "" ? NaN : Number(raw));

// Affordability ratios worked out from the key details as the officer types.
// They mirror ratios the model itself uses, but are shown only as a guide.
function KeyRatios({ values }) {
  const income = toNumber(values.client_income);
  const credit = toNumber(values.credit_amount);
  const annuity = toNumber(values.loan_annuity);
  const ok = (n) => Number.isFinite(n) && n > 0;
  const ratios = [
    {
      label: "Loan to income",
      value: ok(credit) && ok(income) ? `${(credit / income).toFixed(2)}×` : "–",
      note: "credit amount ÷ income",
    },
    {
      label: "Repayment burden",
      value: ok(annuity) && ok(income) ? `${((annuity / income) * 100).toFixed(1)}%` : "–",
      note: "loan annuity ÷ income",
    },
    {
      label: "Implied term",
      value: ok(credit) && ok(annuity) ? `${(credit / annuity).toFixed(1)}` : "–",
      note: "credit amount ÷ annuity (repayments)",
    },
  ];
  return (
    <div className="ratios" aria-live="polite">
      {ratios.map((r) => (
        <div className="ratio" key={r.label}>
          <span className="ratio-label">{r.label}</span>
          <span className="ratio-value">{r.value}</span>
          <span className="ratio-note">{r.note}</span>
        </div>
      ))}
    </div>
  );
}

export default function ApplicationForm({
  specs, required, values, errors, showSummary, summaryRef, reference, onReference, onChange, onBlur, onSubmit, onClear,
  busy, status,
}) {
  const sectionOf = (field) => FIELDS[field]?.section ?? "applicant";
  const fieldsOf = (section) => Object.keys(FIELDS).filter((field) => specs[field] && sectionOf(field) === section.id);
  const [open, setOpen] = useState(() => Object.fromEntries(SECTIONS.map((s) => [s.id, startsOpen(s)])));
  const progress = keyProgress(values, required);
  const allFields = Object.keys(FIELDS).filter((field) => specs[field]);
  const started = progress.done > 0;
  const missingScores = SCORE_FIELDS.filter((field) => specs[field] && (values[field] ?? "") === "");

  // A section with a problem opens by itself, so the problem is never hidden
  useEffect(() => {
    const withProblems = SECTIONS.filter((s) => fieldsOf(s).some((field) => errors[field])).map((s) => s.id);
    if (withProblems.length > 0) {
      setOpen((current) => ({ ...current, ...Object.fromEntries(withProblems.map((id) => [id, true])) }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [errors]);

  const setAll = (value) => setOpen(Object.fromEntries(SECTIONS.map((s) => [s.id, s.id === "key" ? true : value])));

  function jumpTo(id) {
    setOpen((current) => ({ ...current, [id]: true }));
    requestAnimationFrame(() => {
      const target = document.getElementById(`section-${id}`);
      target?.scrollIntoView({ behavior: "smooth", block: "start" });
      target?.querySelector("input, select")?.focus({ preventScroll: true });
    });
  }

  const sectionStatus = (section) => {
    if (section.id === "key") return `${progress.done}/${progress.total}`;
    const fields = fieldsOf(section);
    return `${filledCount(fields, values)}/${fields.length}`;
  };

  return (
    <div className="form-layout">
      <nav className="section-nav" aria-label="Form sections">
        <div className="completeness">
          <span>Completeness</span>
          <strong>
            {filledCount(allFields, values)} / {allFields.length}
          </strong>
        </div>
        <progress max={allFields.length} value={filledCount(allFields, values)} aria-label="Fields filled in" />
        <ul>
          {SECTIONS.map((section) => {
            const problems = fieldsOf(section).filter((field) => errors[field]).length;
            const keyDone = section.id === "key" && progress.done === progress.total;
            return (
              <li key={section.id}>
                <button type="button" className={`nav-item${problems ? " nav-problem" : ""}`} onClick={() => jumpTo(section.id)}>
                  <span>{section.title}</span>
                  <span className="nav-count">{keyDone ? "✓" : problems ? `${problems} to fix` : sectionStatus(section)}</span>
                </button>
              </li>
            );
          })}
        </ul>
        <p className="nav-note">Blank answers are fine; the model handles missing details. Credit scores matter most.</p>
      </nav>

      <form
        className="application-form"
        noValidate
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
        }}
      >
        {showSummary && <ErrorSummary errors={errors} ref={summaryRef} />}

        <div className="reference-row">
          <label htmlFor="application-reference">Application reference (optional)</label>
          <input
            id="application-reference"
            type="text"
            maxLength={40}
            value={reference}
            onChange={(event) => onReference(event.target.value)}
            placeholder="Your own reference or file number"
            autoComplete="off"
          />
          <small className="help">For your own records. It is printed with the summary and is never sent to the model.</small>
        </div>

        {SECTIONS.map((section) => {
          const fields = fieldsOf(section);
          const body = (
            <>
              {section.note && <p className="section-note">{section.note}</p>}
              <div className="field-grid">
                {fields.map((field) => (
                  <Field
                    key={field}
                    name={field}
                    spec={specs[field]}
                    value={values[field]}
                    error={errors[field]}
                    warning={started && missingScores.includes(field) ? "Missing: the result is less reliable." : undefined}
                    disabled={isDisabled(field, values)}
                    onChange={onChange}
                    onBlur={onBlur}
                  />
                ))}
              </div>
              {section.id === "key" && <KeyRatios values={values} />}
            </>
          );
          if (section.id === "key") {
            return (
              <fieldset key={section.id} id={`section-${section.id}`} className="section section-key">
                <legend>{section.title}</legend>
                {body}
              </fieldset>
            );
          }
          const filled = filledCount(fields, values);
          const problems = fields.filter((field) => errors[field]).length;
          return (
            <details
              key={section.id}
              id={`section-${section.id}`}
              className={`section section-fold${problems ? " has-problems" : ""}${section.id === "bureau" ? " section-bureau" : ""}`}
              open={open[section.id]}
              onToggle={(event) => {
                const isOpen = event.currentTarget.open;
                setOpen((current) => (current[section.id] === isOpen ? current : { ...current, [section.id]: isOpen }));
              }}
            >
              <summary>
                <span className="section-title">{section.title}</span>
                <span className="section-count">
                  {section.id === "bureau" && <span className="signal-badge">Strongest signal</span>}
                  {section.id === "bureau" && missingScores.length > 0 && (
                    <span className="count-missing">{missingScores.length} score{missingScores.length > 1 ? "s" : ""} missing</span>
                  )}
                  {problems > 0 ? <span className="count-problem">{problems} to fix</span> : null}
                  {filled} of {fields.length} filled
                </span>
              </summary>
              {body}
            </details>
          );
        })}

        <div className="fold-controls">
          <button type="button" className="link-button" onClick={() => setAll(true)}>
            Expand all sections
          </button>
          <button type="button" className="link-button" onClick={() => setAll(false)}>
            Collapse optional sections
          </button>
        </div>

        <div className="action-bar">
          <div className="progress" role="status">
            <span className="progress-label">
              {progress.done === progress.total
                ? "All key details entered"
                : `${progress.done} of ${progress.total} key details entered`}
              {status && <span className={`run-status run-${status.kind}`}> · {status.text}</span>}
            </span>
            <progress max={progress.total} value={progress.done} aria-label="Key details entered" />
          </div>
          <div className="form-actions">
            <button type="button" className="secondary" onClick={onClear} disabled={busy}>
              Clear form
            </button>
            <button type="submit" disabled={busy}>
              {busy ? "Checking..." : "Check risk"}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
