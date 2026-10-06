import { useEffect, useState } from "react";
import { getHealth } from "./api.js";

export default function App() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setError(e.message));
  }, []);

  return (
    <main className="page">
      <header>
        <h1>Vehicle Loan Risk Check</h1>
        <p>Decision support for loan officers. It does not approve or refuse applications.</p>
      </header>
      {error && <p className="message error">{error}</p>}
      {health && <p className="message">Connected to model {health.model_version}.</p>}
      {!health && !error && <p className="message">Connecting to the prediction service...</p>}
    </main>
  );
}
