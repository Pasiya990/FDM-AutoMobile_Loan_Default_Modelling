import Field from "./Field.jsx";
import { FIELDS, SECTIONS } from "./fields.js";

export default function ApplicationForm({ specs, values, errors, onChange, onSubmit, onClear, busy }) {
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
                  onChange={onChange}
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

      {errors._form && (
        <p className="message error" role="alert">
          {errors._form}
        </p>
      )}
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
