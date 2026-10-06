// How each API field is shown in the form: section, label, help text, and
// readable names for coded values. Types, limits, allowed values and which
// fields are required all come from GET /schema, so they are never typed here.

export const SECTIONS = [
  {
    id: "key",
    title: "Key details",
    note: "These are needed for every check. Amounts are in the currency of the historical data.",
  },
  { id: "applicant", title: "About the applicant" },
  { id: "work", title: "Work" },
  { id: "assets", title: "Assets" },
  { id: "contact", title: "Contact details and documents" },
  {
    id: "bureau",
    title: "Credit bureau",
    note: "Fill these in whenever a bureau record exists: the external scores are the strongest signal the model has. Leave blank if there is no record.",
  },
  {
    id: "timing",
    title: "Application time",
    note: "Filled in automatically from the submission time if left blank.",
  },
];

export const FIELDS = {
  age_years: { section: "key", label: "Age (years)" },
  client_income: { section: "key", label: "Income", help: "The applicant's income.", prefix: "AMT" },
  contract_type: {
    section: "key",
    label: "Contract type",
    values: { CL: "Cash loan", RL: "Revolving loan" },
  },
  credit_amount: { section: "key", label: "Credit amount", help: "Total amount of the loan.", prefix: "AMT" },
  loan_annuity: { section: "key", label: "Loan annuity", help: "The regular repayment amount.", prefix: "AMT" },
  not_employed: { section: "key", label: "Employment status", kind: "employment" },
  years_employed: { section: "key", label: "Years in current job", help: "Needed when employed." },

  gender: { section: "applicant", label: "Gender" },
  marital_status: {
    section: "applicant",
    label: "Marital status",
    values: { D: "Divorced", M: "Married", S: "Single", W: "Widowed" },
  },
  education: { section: "applicant", label: "Education" },
  child_count: { section: "applicant", label: "Number of children" },
  family_members: { section: "applicant", label: "Family members", help: "Including the applicant." },
  housing_type: { section: "applicant", label: "Housing type" },
  accompany_client: { section: "applicant", label: "Came to apply with", help: "Who came with the applicant." },

  income_type: { section: "work", label: "Income type" },
  occupation: { section: "work", label: "Occupation" },
  organization_type: {
    section: "work",
    label: "Employer type",
    values: { XNA: "Not applicable" },
  },

  car_owned: { section: "assets", label: "Owns a car" },
  car_age: { section: "assets", label: "Car age (years)", help: "Only for car owners." },
  bike_owned: { section: "assets", label: "Owns a bike" },
  house_owned: { section: "assets", label: "Owns a house" },
  active_loan: { section: "assets", label: "Has another active loan" },

  mobile_phone: { section: "contact", label: "Gave a mobile number" },
  homephone: { section: "contact", label: "Gave a home phone number" },
  workphone: { section: "contact", label: "Work phone is reachable" },
  permanent_address_match: { section: "contact", label: "Permanent and contact address match" },
  work_contact_match: { section: "contact", label: "Work and contact address match" },
  days_since_registration_change: {
    section: "contact",
    label: "Days since registration changed",
    help: "Days since the applicant last changed their registration.",
  },
  days_since_id_change: {
    section: "contact",
    label: "Days since ID document changed",
    help: "Days since the identity document was last changed.",
  },
  days_since_phone_change: {
    section: "contact",
    label: "Days since phone number changed",
  },
  population_region_relative: {
    section: "contact",
    label: "Region population (relative)",
    help: "How populated the applicant's region is, on the data's scale.",
  },
  city_rating: { section: "contact", label: "City rating", help: "1, 2 or 3, as rated in the data." },

  score_source_1: {
    section: "bureau",
    label: "External credit score 1",
    help: "From 0 to 1; a higher score means a better credit record.",
  },
  score_source_2: { section: "bureau", label: "External credit score 2", help: "From 0 to 1." },
  score_source_3: { section: "bureau", label: "External credit score 3", help: "From 0 to 1." },
  social_circle_default: {
    section: "bureau",
    label: "Defaults in social circle",
    help: "Share of the applicant's contacts who defaulted, from 0 to 1.",
  },
  credit_bureau_enquiries: {
    section: "bureau",
    label: "Credit bureau enquiries",
    help: "Number of enquiries about the applicant at the bureau.",
  },

  application_day: { section: "timing", label: "Day of application", kind: "weekday" },
  application_hour: { section: "timing", label: "Hour of application (0 to 23)" },
};

// The dataset codes days 0 to 6; 0 is taken to be Sunday (see backend/adapter.py)
export const WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

export function valueLabel(field, value) {
  return FIELDS[field]?.values?.[value] ?? value;
}

// Readable numbers for hints: whole numbers above 100, otherwise up to 3 significant digits
export function formatNumber(value) {
  const number = Number(value);
  if (Math.abs(number) >= 100) return Math.round(number).toLocaleString();
  return Number(number.toPrecision(3)).toLocaleString(undefined, { maximumFractionDigits: 4 });
}
