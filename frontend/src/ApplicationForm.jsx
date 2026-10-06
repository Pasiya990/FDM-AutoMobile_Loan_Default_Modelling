import ErrorSummary from "./ErrorSummary.jsx";
import Field from "./Field.jsx";
import { FIELDS, SECTIONS } from "./fields.js";
import { isDisabled } from "./formValues.js";

export default function ApplicationForm({
  specs, values, errors, showSummary, summaryRef, onChange, onBlur, onSubmit, onClear, busy,
}) {
  const sectionOf = (field) => FIELDS[field]?.section ?? "applicant";

  return (
    <form
      className="application-form"
      noValidate
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      {showSummary && <ErrorSummary errors={errors} ref={summaryRef} />}

      {SECTIONS.map((section) => {
        const fields = Object.keys(FIELDS).filter((field) => specs[field] && sectionOf(field) === section.id);
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
                  disabled={isDisabled(field, values)}
                  onChange={onChange}
                  onBlur={onBlur}
                />
              ))}
            </div>
          </>
        );
        return section.collapsed ? (
          <details key={section.id} className="section">
            <summary>{section.title}</summary>
            {body}
          </details>
        ) : (
          <fieldset key={section.id} className="section">
            <legend>{section.title}</legend>
            {body}
          </fieldset>
        );
      })}

      <div className="form-actions">
        <button type="submit" disabled={busy}>
          {busy ? "Checking..." : "Check risk"}
        </button>
        <button type="button" className="secondary" onClick={onClear} disabled={busy}>
          Clear form
        </button>
      </div>
    </form>
  );
}
