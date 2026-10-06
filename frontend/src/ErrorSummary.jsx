import { forwardRef } from "react";
import { FIELDS } from "./fields.js";

// Lists every problem in form order, each linking to its field, so the user can
// see at once what to fix and jump straight to it.
const ErrorSummary = forwardRef(function ErrorSummary({ errors }, ref) {
  const fieldProblems = Object.keys(FIELDS).filter((field) => errors[field]);
  const unknownProblems = Object.keys(errors).filter((key) => errors[key] && key !== "_form" && !FIELDS[key]);
  if (fieldProblems.length === 0 && unknownProblems.length === 0 && !errors._form) return null;

  function jumpTo(event, field) {
    event.preventDefault();
    const input = document.getElementById(`field-${field}`);
    input?.closest("details")?.setAttribute("open", "");
    input?.focus();
  }

  const count = fieldProblems.length + unknownProblems.length;
  return (
    <div className="error-summary" role="alert" tabIndex={-1} ref={ref}>
      <h2>{count > 0 ? `Please check ${count === 1 ? "1 field" : `${count} fields`}` : "The application was not checked"}</h2>
      {errors._form && <p>{errors._form}</p>}
      {count > 0 && (
        <ul>
          {fieldProblems.map((field) => (
            <li key={field}>
              <a href={`#field-${field}`} onClick={(event) => jumpTo(event, field)}>
                {FIELDS[field].label}: {errors[field]}
              </a>
            </li>
          ))}
          {unknownProblems.map((key) => (
            <li key={key}>
              {key}: {errors[key]}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
});

export default ErrorSummary;
