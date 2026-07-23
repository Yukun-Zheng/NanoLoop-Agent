import { z } from "zod";

export const analysisMetadataSchema = z.object({
  job_name: z.string().trim().min(1).max(255),
  images: z
    .array(
      z.object({
        filename: z.string().trim().min(1).max(255),
        sample_id: z.string().trim().min(1).max(120),
        material_name: z.string().trim().max(255).nullable().optional(),
        material_formula: z.string().trim().max(255).nullable().optional(),
        experiment_conditions: z.record(z.string(), z.unknown()).default({}),
        scale: z.discriminatedUnion("mode", [
          z.object({ mode: z.literal("pixel_only"), value: z.null().optional() }),
          z.object({ mode: z.literal("nm_per_pixel"), value: z.number().positive() })
        ])
      })
    )
    .min(1)
    .max(20)
    .refine(
      (images) => new Set(images.map((image) => image.filename)).size === images.length,
      "metadata filenames must be unique"
    )
});

export const knowledgeMetadataSchema = z.object({
  title: z.string().trim().min(1).max(500),
  source_type: z.enum(["paper", "report", "material_note", "other"]),
  year: z.number().int().min(1000).max(3000).nullable().optional(),
  citation_text: z.string().trim().min(1).max(2000),
  material_aliases: z.array(z.string().trim().min(1).max(255)).max(32).default([]),
  license_note: z.string().trim().min(1).max(1000),
  allowed_for_demo: z.boolean().default(false)
});

export const knowledgeFormSchema = z.object({
  title: z.string().trim().min(1, "请填写标题").max(500),
  source_type: z.enum(["paper", "report", "material_note", "other"]),
  year: z
    .string()
    .refine(
      (value) =>
        !value ||
        (Number.isInteger(Number(value)) && Number(value) >= 1000 && Number(value) <= 3000),
      "年份需为 1000–3000 的整数"
    ),
  citation_text: z.string().trim().min(1, "请填写规范引用").max(2000),
  material_aliases_text: z.string().max(8192),
  license_note: z.string().trim().min(1, "请填写许可与来源说明").max(1000),
  allowed_for_demo: z.boolean()
});

export type KnowledgeMetadata = z.infer<typeof knowledgeMetadataSchema>;
export type KnowledgeFormValues = z.infer<typeof knowledgeFormSchema>;
