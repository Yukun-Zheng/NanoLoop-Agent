import { describe, expect, it } from "vitest";

import {
  analysisMetadataSchema,
  knowledgeMetadataSchema
} from "@/lib/contracts/metadata";

describe("metadata string contracts omitted by OpenAPI", () => {
  it("accepts one valid analysis image", () => {
    expect(
      analysisMetadataSchema.parse({
        job_name: "SEM",
        images: [
          {
            filename: "a.png",
            sample_id: "sample-a",
            experiment_conditions: {},
            scale: { mode: "pixel_only" }
          }
        ]
      }).images
    ).toHaveLength(1);
  });

  it("rejects duplicate filenames and invalid physical scale", () => {
    expect(() =>
      analysisMetadataSchema.parse({
        job_name: "SEM",
        images: [
          { filename: "a.png", sample_id: "1", scale: { mode: "pixel_only" } },
          {
            filename: "a.png",
            sample_id: "2",
            scale: { mode: "nm_per_pixel", value: 0 }
          }
        ]
      })
    ).toThrow();
  });

  it("rejects more than 20 analysis images", () => {
    expect(() =>
      analysisMetadataSchema.parse({
        job_name: "SEM",
        images: Array.from({ length: 21 }, (_, index) => ({
          filename: `${index}.png`,
          sample_id: `sample-${index}`,
          scale: { mode: "pixel_only" }
        }))
      })
    ).toThrow();
  });

  it("defaults knowledge demo permission to false", () => {
    expect(
      knowledgeMetadataSchema.parse({
        title: "Team note",
        source_type: "material_note",
        citation_text: "NanoLoop team, 2026",
        license_note: "Team-authored"
      }).allowed_for_demo
    ).toBe(false);
  });
});
