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

export function checkValues(values, specs) {
  const errors = {};
  for (const [field, spec] of Object.entries(specs)) {
    const raw = values[field];
    const blank = raw === undefined || raw === "";
    if (blank) {
      if (spec.required) errors[field] = "Required.";
      continue;
    }
    if (spec.type === "integer" || spec.type === "number") {
      const number = Number(raw);
      if (!Number.isFinite(number)) errors[field] = "Enter a number.";
      else if (spec.type === "integer" && !Number.isInteger(number)) errors[field] = "Enter a whole number.";
      else if (number < spec.min || number > spec.max) errors[field] = `Must be between ${spec.min} and ${spec.max}.`;
    }
  }
  if (values.not_employed === "no" && (values.years_employed ?? "") === "") {
    errors.years_employed = "Required when the applicant is employed.";
  }
  if (values.car_owned === "no" && (values.car_age ?? "") !== "") {
    errors.car_age = "Leave blank when the applicant has no car.";
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
