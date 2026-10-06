// Converting between what the form holds (text from inputs, "yes"/"no" from
// yes/no selects) and the application the API expects, plus friendly checks
// before sending. The API checks everything again; these only catch mistakes early.

export function toApplication(values, specs) {
  const application = {};
  for (const [field, raw] of Object.entries(values)) {
    const spec = specs[field];
    if (!spec || raw === "" || raw === undefined || raw === null) continue;
    if (spec.type === "boolean") application[field] = raw === "yes";
    else if (spec.type === "category") application[field] = raw;
    else application[field] = Number(raw);
  }
  return application;
}

export function fromApplication(application, specs) {
  const values = {};
  for (const [field, value] of Object.entries(application)) {
    const spec = specs[field];
    if (!spec || value === null || value === undefined) continue;
    values[field] = spec.type === "boolean" ? (value ? "yes" : "no") : String(value);
  }
  return values;
}

const isBlank = (raw) => raw === undefined || raw === null || raw === "";

// Fields that are switched off by another answer, and the answer that switches them off
export const DISABLED_WHEN = {
  years_employed: { field: "not_employed", value: "yes" },
  car_age: { field: "car_owned", value: "no" },
};

export function isDisabled(field, values) {
  const rule = DISABLED_WHEN[field];
  return Boolean(rule) && values[rule.field] === rule.value;
}

// The problem with one field, or undefined
export function checkField(field, values, specs) {
  const spec = specs[field];
  const raw = values[field];
  if (!spec) return undefined;

  if (isBlank(raw)) {
    if (spec.required) return "Required.";
    if (field === "years_employed" && values.not_employed === "no") return "Required when the applicant is employed.";
    return undefined;
  }
  if (isDisabled(field, values)) {
    return field === "car_age" ? "Leave blank when the applicant has no car." : undefined;
  }
  if (spec.type === "integer" || spec.type === "number") {
    const number = Number(raw);
    if (!Number.isFinite(number)) return "Enter a number.";
    if (spec.type === "integer" && !Number.isInteger(number)) return "Enter a whole number.";
    if (number < spec.min || number > spec.max) return `Must be between ${spec.min} and ${spec.max}.`;
  }
  return undefined;
}

export function checkValues(values, specs) {
  const errors = {};
  for (const field of Object.keys(specs)) {
    const problem = checkField(field, values, specs);
    if (problem) errors[field] = problem;
  }
  return errors;
}

// The API's 422 details ({field, problem, received}) as messages per field
export function apiErrors(details) {
  const errors = {};
  for (const { field, problem } of details ?? []) {
    const key = field ?? "_form";
    errors[key] = errors[key] ? `${errors[key]} ${problem}` : capitalise(problem);
  }
  return errors;
}

function capitalise(text) {
  return text.charAt(0).toUpperCase() + text.slice(1) + ".";
}
