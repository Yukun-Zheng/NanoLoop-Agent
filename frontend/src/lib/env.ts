import { z } from "zod";

const httpOriginSchema = z
  .url()
  .transform((value) => new URL(value))
  .refine(
    (value) =>
      ["http:", "https:"].includes(value.protocol) &&
      value.origin === value.href.replace(/\/$/, ""),
    { message: "frontend origins must be exact HTTP(S) origins without paths" }
  )
  .transform((value) => value.origin);

const serverEnvSchema = z.object({
  NANOLOOP_API_INTERNAL_URL: z
    .url()
    .refine((value) => value.startsWith("http://") || value.startsWith("https://"), {
      message: "NANOLOOP_API_INTERNAL_URL must be HTTP(S)"
    })
    .default("http://127.0.0.1:8000"),
  NANOLOOP_API_KEY: z.string().default(""),
  NANOLOOP_FRONTEND_ALLOWED_ORIGINS: z
    .string()
    .default("http://127.0.0.1:3000,http://localhost:3000")
    .transform((value, context) => {
      const values = value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
      if (!values.length) {
        context.addIssue({
          code: "custom",
          message: "at least one frontend origin is required"
        });
        return z.NEVER;
      }
      const parsed = values.map((item) => httpOriginSchema.safeParse(item));
      for (const result of parsed) {
        if (!result.success) {
          context.addIssue({
            code: "custom",
            message: result.error.issues[0]?.message ?? "invalid frontend origin"
          });
          return z.NEVER;
        }
      }
      return [...new Set(parsed.map((result) => result.data as string))];
    })
});

export function getServerEnv() {
  return serverEnvSchema.parse({
    NANOLOOP_API_INTERNAL_URL: process.env.NANOLOOP_API_INTERNAL_URL,
    NANOLOOP_API_KEY: process.env.NANOLOOP_API_KEY,
    NANOLOOP_FRONTEND_ALLOWED_ORIGINS:
      process.env.NANOLOOP_FRONTEND_ALLOWED_ORIGINS
  });
}
