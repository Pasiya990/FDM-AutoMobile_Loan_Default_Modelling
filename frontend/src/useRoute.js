import { useEffect, useState } from "react";

// The current page from the address after "#" ("#/how-to-use" -> "/how-to-use").
// Links within the page need no extra package, and refreshing never asks the
// server for a path it does not have.
function currentRoute() {
  const path = window.location.hash.replace(/^#/, "");
  return path || "/";
}

export default function useRoute() {
  const [route, setRoute] = useState(currentRoute);

  useEffect(() => {
    function onChange() {
      setRoute(currentRoute());
      window.scrollTo({ top: 0 });
    }
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  return route;
}
