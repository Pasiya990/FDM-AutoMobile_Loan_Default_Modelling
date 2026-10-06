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
  return { status: response.status, body };
}

export async function getHealth() {
  const { body } = await request("/health");
  return body;
}
