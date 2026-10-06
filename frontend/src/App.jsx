import { useEffect, useRef, useState } from "react";
import ApplicationForm from "./ApplicationForm.jsx";
import { AboutModel, HowToUse } from "./Guide.jsx";
import ResultPanel from "./ResultPanel.jsx";
import { getExamples, getHealth, getSchema, predict } from "./api.js";
import { apiErrors, checkValues, fromApplication, toApplication } from "./formValues.js";

const DISCLAIMER =
  "Academic prototype trained on a historical dataset. Decision support only: it must not be used to refuse an application automatically.";

export default function App() {
  const [health, setHealth] = useState(null);
  const [schema, setSchema] = useState(null);
  const [examples, setExamples] = useState([]);
  const [loadError, setLoadError] = useState(null);
  const [values, setValues] = useState({});
  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const resultRef = useRef(null);

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

  function handleChange(field, value) {
    setValues((current) => ({ ...current, [field]: value }));
    setErrors((current) => ({ ...current, [field]: undefined, _form: undefined }));
    setResult(null);
  }

  async function handleSubmit() {
    const found = checkValues(values, schema.fields);
    setResult(null);
    if (Object.keys(found).length > 0) {
      setErrors({ ...found, _form: "Please correct the highlighted fields." });
      return;
    }
    setErrors({});
    setBusy(true);
    try {
      const { status, body } = await predict(toApplication(values, schema.fields));
      if (status === 200) setResult(body);
      else setErrors({ ...apiErrors(body?.details), _form: "The service could not accept this application." });
    } catch (error) {
      setErrors({ _form: error.message });
    } finally {
      setBusy(false);
    }
  }

  function loadExample(example) {
    setValues(fromApplication(example.application, schema.fields));
    setErrors({});
    setResult(null);
  }

  function handleClear() {
    setValues({});
    setErrors({});
    setResult(null);
  }

  return (
    <main className="page">
      <header>
        <h1>Vehicle Loan Risk Check</h1>
        <p>
          Enter an application to see how risky it looks compared with thousands of past vehicle-loan applicants.
        </p>
      </header>

      <p className="disclaimer" role="note">
        {DISCLAIMER}
      </p>

      {loadError && (
        <p className="message error" role="alert">
          {loadError}
        </p>
      )}
      {!schema && !loadError && <p className="message">Connecting to the prediction service...</p>}

      {schema && (
        <>
          <div className="guides">
            <HowToUse />
            <AboutModel schema={schema} />
          </div>

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

          <ApplicationForm
            specs={schema.fields}
            values={values}
            errors={errors}
            onChange={handleChange}
            onSubmit={handleSubmit}
            onClear={handleClear}
            busy={busy}
          />

          {result && (
            <div ref={resultRef} tabIndex={-1} className="result-anchor">
              <ResultPanel result={result} bands={schema.risk_bands} />
            </div>
          )}
        </>
      )}

      <footer>Model: {health?.model_version ?? "not connected"}</footer>
    </main>
  );
}
