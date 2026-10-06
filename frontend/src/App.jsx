import { useEffect, useRef, useState } from "react";
import ApplicationForm from "./ApplicationForm.jsx";
import Header, { Logo, PRODUCT_NAME } from "./Header.jsx";
import HowToUsePage from "./HowToUsePage.jsx";
import OverviewPage from "./OverviewPage.jsx";
import ResultPanel from "./ResultPanel.jsx";
import { getExamples, getHealth, getSchema, predict } from "./api.js";
import { FIELDS, WEEKDAYS, valueLabel } from "./fields.js";
import useRoute from "./useRoute.js";
import { DISABLED_WHEN, apiErrors, checkField, checkValues, fromApplication, toApplication } from "./formValues.js";

export default function App() {
  const route = useRoute();
  const [health, setHealth] = useState(null);
  const [schema, setSchema] = useState(null);
  const [examples, setExamples] = useState([]);
  const [loadError, setLoadError] = useState(null);
  const [values, setValues] = useState({});
  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [showSummary, setShowSummary] = useState(false);
  const [reference, setReference] = useState("");
  const [stale, setStale] = useState(false);
  const resultRef = useRef(null);
  const summaryRef = useRef(null);

  useEffect(() => {
    Promise.all([getHealth(), getSchema()])
      .then(([healthBody, schemaBody]) => {
        setHealth(healthBody);
        setSchema(schemaBody);
      })
      .catch((error) => setLoadError(error.message));
    getExamples()
      .then(setExamples)
      .catch(() => setExamples([]));
  }, []);

  useEffect(() => {
    if (result) resultRef.current?.focus();
  }, [result]);

  // After a failed check, move to the summary so the problems are seen first
  const [summaryFocus, setSummaryFocus] = useState(0);
  useEffect(() => {
    if (summaryFocus) summaryRef.current?.focus();
  }, [summaryFocus]);

  function showProblems(problems) {
    setErrors(problems);
    setShowSummary(true);
    setSummaryFocus((n) => n + 1);
  }

  function handleChange(field, value) {
    // Answers that switch another field off also clear it (no car -> no car age)
    const cleared = Object.keys(DISABLED_WHEN).filter(
      (other) => DISABLED_WHEN[other].field === field && DISABLED_WHEN[other].value === value,
    );
    setValues((current) => ({ ...current, [field]: value, ...Object.fromEntries(cleared.map((f) => [f, ""])) }));
    const dependants = Object.keys(DISABLED_WHEN).filter((other) => DISABLED_WHEN[other].field === field);
    setErrors((current) => ({
      ...current,
      ...Object.fromEntries([field, ...dependants].map((f) => [f, undefined])),
      _form: undefined,
    }));
    // The last result stays on screen, marked out of date, until the officer checks again
    if (result) setStale(true);
  }

  // Check a field as soon as the user leaves it
  function handleBlur(field) {
    setErrors((current) => ({ ...current, [field]: checkField(field, values, schema.fields) }));
  }

  async function handleSubmit() {
    const found = checkValues(values, schema.fields);
    setResult(null);
    if (Object.keys(found).length > 0) {
      showProblems(found);
      return;
    }
    setErrors({});
    setShowSummary(false);
    setBusy(true);
    try {
      const { status, body } = await predict(toApplication(values, schema.fields));
      if (status === 200) {
        const missingScores = ["score_source_1", "score_source_2", "score_source_3"].filter((f) => (values[f] ?? "") === "").length;
        setResult({ ...body, checkedAt: new Date(), missingScores });
        setStale(false);
      }
      else showProblems({ ...apiErrors(body?.details), _form: "The service could not accept this application." });
    } catch (error) {
      showProblems({ _form: error.message });
    } finally {
      setBusy(false);
    }
  }

  function loadExample(example) {
    setStale(false);
    setValues(fromApplication(example.application, schema.fields));
    setErrors({});
    setShowSummary(false);
    setResult(null);
  }

  function handleClear() {
    setStale(false);
    setReference("");
    setValues({});
    setErrors({});
    setShowSummary(false);
    setResult(null);
  }

  // Shown next to the progress in the action bar
  function runStatus() {
    if (busy) return { kind: "busy", text: "checking..." };
    if (!result) return null;
    const time = result.checkedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return stale
      ? { kind: "stale", text: "details changed since the last check" }
      : { kind: "fresh", text: `result updated ${time}` };
  }

  function startNewApplication() {
    handleClear();
    window.scrollTo({ top: 0, behavior: "smooth" });
    document.getElementById("field-age_years")?.focus({ preventScroll: true });
  }

  // The details entered, in form order and readable form, for the printed summary
  function enteredDetails() {
    const entered = Object.keys(FIELDS)
      .filter((field) => (values[field] ?? "") !== "")
      .map((field) => {
        const raw = values[field];
        let shown = valueLabel(field, raw);
        if (field === "not_employed") shown = raw === "yes" ? "Not employed or retired" : "Employed";
        else if (FIELDS[field].kind === "weekday") shown = WEEKDAYS[Number(raw)];
        else if (raw === "yes" || raw === "no") shown = raw === "yes" ? "Yes" : "No";
        return { label: FIELDS[field].label, value: shown };
      });
    return reference.trim() ? [{ label: "Application reference", value: reference.trim() }, ...entered] : entered;
  }

  const page = route === "/overview" || route === "/how-to-use" ? route : "/";
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <Header route={page} health={health} loadError={loadError} />
      <main className="page" id="main">
        {loadError && (
          <div className="message error" role="alert">
            <strong>The prediction service is not available.</strong> {loadError} Start it with{" "}
            <code>uvicorn backend.app:app</code> and reload this page.
          </div>
        )}
        {!schema && !loadError && <p className="message">Connecting to the prediction service...</p>}

        {schema && page === "/overview" && <OverviewPage schema={schema} />}
        {schema && page === "/how-to-use" && <HowToUsePage schema={schema} />}

        {schema && page === "/" && (
          <>
            <div className="page-intro">
              <div>
                <p className="eyebrow">Step 1 · Enter the application</p>
                <h1>New application</h1>
                <p>
                  Enter what you know when the application comes in. Nothing you enter is stored on the server. New here?
                  Read <a href="#/how-to-use">How to use</a>.
                </p>
              </div>
              {examples.length > 0 && (
                <div className="examples">
                  <span>Try an example:</span>
                  {examples.map((example) => (
                    <button key={example.label} type="button" className="chip" onClick={() => loadExample(example)}>
                      {example.label.replace(" applicant", "")}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="workspace">
              <ApplicationForm
                specs={schema.fields}
                required={schema.required}
                reference={reference}
                onReference={setReference}
                values={values}
                errors={errors}
                showSummary={showSummary}
                summaryRef={summaryRef}
                onChange={handleChange}
                onBlur={handleBlur}
                onSubmit={handleSubmit}
                onClear={handleClear}
                busy={busy}
                status={runStatus()}
              />

              <aside className="result-column" aria-label="Result">
                {result ? (
                  <div ref={resultRef} tabIndex={-1} className="result-anchor">
                    <ResultPanel
                      key={result.checkedAt.getTime()}
                      stale={stale}
                      result={result}
                      bands={schema.risk_bands}
                      details={enteredDetails()}
                      reference={reference.trim()}
                      checkedAt={result.checkedAt}
                      onNewApplication={startNewApplication}
                    />
                  </div>
                ) : (
                  <div className="result-placeholder">
                    <p className="eyebrow">Step 2 · Risk assessment</p>
                    <p>{busy ? "Checking the application..." : "Fill in the key details and press Check risk. The result appears here."}</p>
                  </div>
                )}
              </aside>
            </div>
          </>
        )}
      </main>
      <footer className="site-footer">
        <div className="site-footer-inner">
          <span className="footer-brand">
            <Logo size={22} /> {PRODUCT_NAME} · IT3051 Fundamentals of Data Mining mini project
          </span>
          <span>
            Statistical estimates from historical data; they can be wrong. Model {health?.model_version ?? "not connected"} ·{" "}
            <a href="/docs">API documentation</a>
          </span>
        </div>
      </footer>
    </>
  );
}
