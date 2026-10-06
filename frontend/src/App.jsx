import { useEffect, useState } from "react";
import ApplicationForm from "./ApplicationForm.jsx";
import { getHealth, getSchema, predict } from "./api.js";
import { apiErrors, checkValues, toApplication } from "./formValues.js";

export default function App() {
  const [health, setHealth] = useState(null);
  const [schema, setSchema] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [values, setValues] = useState({});
  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    Promise.all([getHealth(), getSchema()])
      .then(([healthBody, schemaBody]) => {
        setHealth(healthBody);
        setSchema(schemaBody);
      })
      .catch((error) => setLoadError(error.message));
  }, []);

  function handleChange(field, value) {
    setValues((current) => ({ ...current, [field]: value }));
    setErrors((current) => ({ ...current, [field]: undefined, _form: undefined }));
  }

  async function handleSubmit() {
    const found = checkValues(values, schema.fields);
    setErrors(found);
    setResult(null);
    if (Object.keys(found).length > 0) {
      setErrors({ ...found, _form: "Please correct the highlighted fields." });
      return;
    }
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

  function handleClear() {
    setValues({});
    setErrors({});
    setResult(null);
  }

  return (
    <main className="page">
      <header>
        <h1>Vehicle Loan Risk Check</h1>
        <p>Decision support for loan officers. It does not approve or refuse applications.</p>
      </header>

      {loadError && (
        <p className="message error" role="alert">
          {loadError}
        </p>
      )}
      {!schema && !loadError && <p className="message">Connecting to the prediction service...</p>}

      {schema && (
        <>
          <p className="status">Model: {health?.model_version}</p>
          <ApplicationForm
            specs={schema.fields}
            values={values}
            errors={errors}
            onChange={handleChange}
            onSubmit={handleSubmit}
            onClear={handleClear}
            busy={busy}
          />
        </>
      )}

      {result && (
        <section className="message" aria-live="polite">
          <strong>{result.prediction}</strong> - risk band {result.risk_band}, risk score {result.risk_score}
        </section>
      )}
    </main>
  );
}
