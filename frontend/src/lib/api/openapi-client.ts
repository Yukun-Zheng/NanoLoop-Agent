import createClient from "openapi-fetch";

import type { paths } from "./schema";
import { parseEnvelope, type SuccessfulApiEnvelope } from "./envelope";
import type { HealthData } from "./types";

const openApiClient = createClient<paths>({
  baseUrl: "/api/nanoloop",
  fetch: sameOriginBffFetch
});

export async function getHealth(): Promise<SuccessfulApiEnvelope<HealthData>> {
  const { data, error, response } = await openApiClient.GET("/api/v1/health", {
    cache: "no-store"
  });
  return parseEnvelope<HealthData>(data ?? error, response.status);
}

async function sameOriginBffFetch(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> {
  const request =
    input instanceof Request
      ? input
      : new Request(new URL(String(input), window.location.origin), init);
  const url = new URL(request.url);
  const generatedPrefix = "/api/nanoloop/api/v1/";
  if (!url.pathname.startsWith(generatedPrefix)) {
    throw new TypeError("Generated OpenAPI request did not target the NanoLoop API prefix");
  }
  url.pathname = `/api/nanoloop/${url.pathname.slice(generatedPrefix.length)}`;
  return fetch(new Request(url, request));
}
