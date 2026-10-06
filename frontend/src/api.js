// Calls to the prediction API. The page is served by the API itself, so the
// paths are relative; in development Vite forwards them (see vite.config.js).

async function request(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("The prediction service cannot be reached. Is the API running?");
  }
  const body = await response.json().catch(() => null);
  if (response.status === 503) {
    throw new Error(body?.message || "The model is not available on the server.");
  }
  if (response.status >= 500) {
    throw new Error("The prediction service had a problem. Please try again.");
  }
  return { status: response.status, body };
}

export async function getHealth() {
  return (await request("/health")).body;
}

export async function getSchema() {
  return (await request("/schema")).body;
}

export async function getExamples() {
  return (await request("/examples")).body ?? [];
}

// Returns {status, body}: 200 with the result, or 422 with per-field details
export async function predict(application) {
  return request("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(application),
  });
}
