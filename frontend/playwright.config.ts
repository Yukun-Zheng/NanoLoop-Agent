import { defineConfig, devices } from "@playwright/test";

const portValue = process.env.NANOLOOP_E2E_PORT || "3000";
if (!/^\d{2,5}$/.test(portValue) || Number(portValue) > 65_535) {
  throw new Error("NANOLOOP_E2E_PORT must be a valid TCP port");
}
const baseURL = `http://127.0.0.1:${portValue}`;

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL,
    trace: "retain-on-failure"
  },
  webServer: {
    command: `./node_modules/.bin/next dev --hostname 127.0.0.1 --port ${portValue}`,
    url: `${baseURL}/api/healthz`,
    reuseExistingServer: !process.env.CI && !process.env.NANOLOOP_E2E_PORT,
    timeout: 120_000
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] }
    }
  ]
});
