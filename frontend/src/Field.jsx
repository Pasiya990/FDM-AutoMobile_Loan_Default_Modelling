import { FIELDS, WEEKDAYS, valueLabel } from "./fields.js";

function hint(spec) {
  if (spec.typical_min === undefined) return null;
  if (spec.typical_min === spec.min && spec.typical_max === spec.max) return null;
  return `Typical: ${format(spec.typical_min)} to ${format(spec.typical_max)}`;
}

function format(number) {
  return Number(number).toLocaleString(undefined, { maximumFractionDigits: 3 });
}

export default function Field({ name, spec, value, error, onChange }) {
  const info = FIELDS[name] ?? { label: name };
  const id = `field-${name}`;
  const describedBy = error ? `${id}-error` : undefined;
  const common = {
    id,
    name,
    value: value ?? "",
    onChange: (event) => onChange(name, event.target.value),
    "aria-invalid": error ? "true" : undefined,
    "aria-describedby": describedBy,
  };

  let input;
  if (info.kind === "employment") {
    input = (
      <select {...common}>
        <option value="">Choose...</option>
        <option value="no">Employed</option>
        <option value="yes">Not employed or retired</option>
      </select>
    );
  } else if (info.kind === "weekday") {
    input = (
      <select {...common}>
        <option value="">Use submission time</option>
        {WEEKDAYS.map((day, index) => (
          <option key={day} value={String(index)}>
            {day}
          </option>
        ))}
      </select>
    );
  } else if (spec.type === "boolean") {
    input = (
      <select {...common}>
        <option value="">Not given</option>
        <option value="yes">Yes</option>
        <option value="no">No</option>
      </select>
    );
  } else if (spec.type === "category") {
    const options = [...spec.allowed].sort((a, b) => valueLabel(name, a).localeCompare(valueLabel(name, b)));
    input = (
      <select {...common}>
        <option value="">{spec.required ? "Choose..." : "Not given"}</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {valueLabel(name, option)}
          </option>
        ))}
      </select>
    );
  } else {
    input = (
      <input
        {...common}
        type="number"
        inputMode="decimal"
        min={spec.min}
        max={spec.max}
        step={spec.type === "integer" ? 1 : "any"}
      />
    );
  }

  const helpText = [info.help, hint(spec)].filter(Boolean).join(" ");
  return (
    <div className={`field${error ? " has-error" : ""}`}>
      <label htmlFor={id}>
        {info.label}
        {spec.required && <span className="required"> (required)</span>}
      </label>
      {input}
      {helpText && <small className="help">{helpText}</small>}
      {error && (
        <small className="field-error" id={`${id}-error`} role="alert">
          {error}
        </small>
      )}
    </div>
  );
}
