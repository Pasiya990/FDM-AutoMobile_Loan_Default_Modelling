// How each API field is shown in the form: section, label, help text, and
// readable names for coded values. Types, limits, allowed values and which
// fields are required all come from GET /schema, so they are never typed here.

export const SECTIONS = [
  { id: "applicant", title: "Applicant" },
  { id: "loan", title: "Loan" },
  { id: "employment", title: "Employment and income" },
  { id: "assets", title: "Assets" },
  { id: "contact", title: "Contact and documents" },
  { id: "bureau", title: "Credit bureau", note: "Leave blank if no bureau record is available." },
  { id: "timing", title: "Application time", note: "Filled in automatically from the submission time if left blank.", collapsed: true },
];

export const FIELDS = {
  age_years: { section: "applicant", label: "Age (years)" },
  gender: { section: "applicant", label: "Gender" },
  marital_status: {
    section: "applicant",
    label: "Marital status",
    values: { D: "Divorced", M: "Married", S: "Single", W: "Widowed" },
  },
  education: { section: "applicant", label: "Education" },
  child_count: { section: "applicant", label: "Number of children" },
  family_members: { section: "applicant", label: "Family members" },
  housing_type: { section: "applicant", label: "Housing type" },
  accompany_client: { section: "applicant", label: "Came to apply with" },

  contract_type: {
    section: "loan",
    label: "Contract type",
    values: { CL: "Cash loan", RL: "Revolving loan" },
  },
  credit_amount: { section: "loan", label: "Credit amount" },
  loan_annuity: { section: "loan", label: "Loan annuity (yearly repayment)" },
  active_loan: { section: "loan", label: "Has another active loan" },

  not_employed: { section: "employment", label: "Employment status", kind: "employment" },
  years_employed: { section: "employment", label: "Years in current job", help: "Required when employed." },
  client_income: { section: "employment", label: "Income" },
  income_type: { section: "employment", label: "Income type" },
  occupation: { section: "employment", label: "Occupation" },
  organization_type: {
    section: "employment",
    label: "Employer type",
    values: { XNA: "Not applicable" },
  },

  car_owned: { section: "assets", label: "Owns a car" },
  car_age: { section: "assets", label: "Car age (years)", help: "Only for car owners." },
  bike_owned: { section: "assets", label: "Owns a bike" },
  house_owned: { section: "assets", label: "Owns a house" },

  mobile_phone: { section: "contact", label: "Gave a mobile phone number" },
  homephone: { section: "contact", label: "Gave a home phone number" },
  workphone: { section: "contact", label: "Work phone is reachable" },
  permanent_address_match: { section: "contact", label: "Permanent address matches contact address" },
  work_contact_match: { section: "contact", label: "Work address matches contact address" },
  days_since_registration_change: { section: "contact", label: "Days since registration changed" },
  days_since_id_change: { section: "contact", label: "Days since ID document changed" },
  days_since_phone_change: { section: "contact", label: "Days since phone number changed" },
  population_region_relative: { section: "contact", label: "Region population (relative)" },
  city_rating: { section: "contact", label: "City rating (1 to 3)" },

  score_source_1: { section: "bureau", label: "External credit score 1 (0 to 1)" },
  score_source_2: { section: "bureau", label: "External credit score 2 (0 to 1)" },
  score_source_3: { section: "bureau", label: "External credit score 3 (0 to 1)" },
  social_circle_default: { section: "bureau", label: "Share of social circle who defaulted (0 to 1)" },
  credit_bureau_enquiries: { section: "bureau", label: "Credit bureau enquiries" },

  application_day: {
    section: "timing",
    label: "Day of application",
    kind: "weekday",
  },
  application_hour: { section: "timing", label: "Hour of application (0 to 23)" },
};

// The dataset codes days 0 to 6; 0 is taken to be Sunday (see backend/adapter.py)
export const WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

export function valueLabel(field, value) {
  return FIELDS[field]?.values?.[value] ?? value;
}
