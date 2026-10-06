import { useEffect, useRef, useState } from "react";
import ApplicationForm from "./ApplicationForm.jsx";
import Header from "./Header.jsx";
import HowToUsePage from "./HowToUsePage.jsx";
import ResultPanel from "./ResultPanel.jsx";
import { getExamples, getHealth, getSchema, predict } from "./api.js";
import { FIELDS, WEEKDAYS, valueLabel } from "./fields.js";
import useRoute from "./useRoute.js";
import { DISABLED_WHEN, apiErrors, checkField, checkValues, fromApplication, toApplication } from "./formValues.js";

const DISCLAIMER =
  "Academic prototype trained on a historical dataset. Decision support only: it must not be used to refuse an application automatically.";

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
    setResult(null);
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
      if (status === 200) setResult(body);
      else showProblems({ ...apiErrors(body?.details), _form: "The service could not accept this application." });
    } catch (error) {
      showProblems({ _form: error.message });
    } finally {
      setBusy(false);
    }
  }

  function loadExample(example) {
    setValues(fromApplication(example.application, schema.fields));
    setErrors({});
    setShowSummary(false);
    setResult(null);
  }

  function handleClear() {
    setValues({});
    setErrors({});
    setShowSummary(false);
    setResult(null);
  }

  function startNewApplication() {
    handleClear();
    window.scrollTo({ top: 0, behavior: "smooth" });
    document.getElementById("field-age_years")?.focus({ preventScroll: true });
  }

  // The details entered, in form order and readable form, for the printed summary
  function enteredDetails() {
    return Object.keys(FIELDS)
      .filter((field) => (values[field] ?? "") !== "")
      .map((field) => {
        const raw = values[field];
        let shown = valueLabel(field, raw);
        if (field === "not_employed") shown = raw === "yes" ? "Not employed or retired" : "Employed";
        else if (FIELDS[field].kind === "weekday") shown = WEEKDAYS[Number(raw)];
        else if (raw === "yes" || raw === "no") shown = raw === "yes" ? "Yes" : "No";
        return { label: FIELDS[field].label, value: shown };
      });
  }

  const onGuide = route === "/how-to-use";
  return (
    <>
      <Header route={onGuide ? "/how-to-use" : "/"} />
      <main className="page">
        {!onGuide && (
          <div className="page-intro">
            <h1>Check an application</h1>
            <p>
              Enter an application to see how risky it looks compared with thousands of past vehicle-loan applicants.
              New to the tool? Read <a href="#/how-to-use">How to use</a>.
            </p>
          </div>
        )}

        <p className="disclaimer" role="note">
          {DISCLAIMER}
        </p>

        {loadError && (
          <p className="message error" role="alert">
            {loadError}
          </p>
        )}
        {!schema && !loadError && <p className="message">Connecting to the prediction service...</p>}

        {schema && onGuide && <HowToUsePage schema={schema} />}

        {schema && !onGuide && (
          <>
            {examples.length > 0 && (
              <div className="examples">
                <span>Try an example:</span>
                {examples.map((example) => (
                  <button key={example.label} type="button" className="secondary" onClick={() => loadExample(example)}>
                    {example.label}
                  </button>
                ))}
              </div>
            )}

            <div className="workspace">
              <ApplicationForm
                specs={schema.fields}
                values={values}
                errors={errors}
                showSummary={showSummary}
                summaryRef={summaryRef}
                onChange={handleChange}
                onBlur={handleBlur}
                onSubmit={handleSubmit}
                onClear={handleClear}
                busy={busy}
              />

              <aside className="result-column" aria-label="Result">
                {result ? (
                  <div ref={resultRef} tabIndex={-1} className="result-anchor">
                    <ResultPanel
                      result={result}
                      bands={schema.risk_bands}
                      details={enteredDetails()}
                      onNewApplication={startNewApplication}
                    />
                  </div>
                ) : (
                  <p className="result-placeholder">
                    {busy ? "Checking the application..." : "Fill in the form and press Check risk. The result appears here."}
                  </p>
                )}
              </aside>
            </div>
          </>
        )}

        <footer>Model: {health?.model_version ?? "not connected"}</footer>
      </main>
    </>
  );
}
