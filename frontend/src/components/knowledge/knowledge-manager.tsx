"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BookOpen,
  FileText,
  RefreshCw,
  ShieldCheck,
  ToggleLeft,
  ToggleRight,
  Upload
} from "lucide-react";
import Link from "next/link";
import { useRef, useState } from "react";
import { useForm } from "react-hook-form";

import { Brand } from "@/components/shell/brand";
import { HealthIndicator } from "@/components/shell/health-indicator";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { RequestError } from "@/components/ui/request-error";
import { StatusBadge } from "@/components/ui/status-badge";
import { apiRequest } from "@/lib/api/client";
import { getHealth } from "@/lib/api/openapi-client";
import { queryKeys } from "@/lib/api/query-keys";
import type {
  KnowledgeDocument,
  KnowledgeDocumentList,
  ReindexReport
} from "@/lib/api/types";
import {
  knowledgeFormSchema,
  knowledgeMetadataSchema,
  type KnowledgeFormValues
} from "@/lib/contracts/metadata";
import { compactId, formatDate } from "@/lib/format/value";
import { coreMutationBlocker } from "@/lib/health";

export function KnowledgeManager() {
  const queryClient = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [lastReport, setLastReport] = useState<Record<string, unknown> | null>(null);
  const form = useForm<KnowledgeFormValues>({
    resolver: zodResolver(knowledgeFormSchema),
    defaultValues: {
      title: "",
      source_type: "paper",
      year: "",
      citation_text: "",
      material_aliases_text: "",
      license_note: "",
      allowed_for_demo: false
    }
  });

  const documents = useQuery({
    queryKey: queryKeys.knowledge,
    queryFn: () =>
      apiRequest<KnowledgeDocumentList>("knowledge/documents").then(
        (response) => response.data
      )
  });
  const health = useQuery({
    queryKey: queryKeys.health,
    queryFn: () => getHealth().then((response) => response.data),
    refetchInterval: 15_000
  });
  const writeBlocker = coreMutationBlocker(health.data, {
    failed: health.isError,
    pending: health.isPending
  });
  const documentItems = documents.data?.documents ?? [];

  const ingest = useMutation({
    mutationFn: async (values: KnowledgeFormValues) => {
      if (!file) throw new Error("请选择 PDF、TXT 或 Markdown 文件");
      const metadata = knowledgeMetadataSchema.parse({
        title: values.title,
        source_type: values.source_type,
        year: values.year ? Number(values.year) : null,
        citation_text: values.citation_text,
        material_aliases: values.material_aliases_text
          .split(/[,，]/)
          .map((item) => item.trim())
          .filter(Boolean),
        license_note: values.license_note,
        allowed_for_demo: values.allowed_for_demo
      });
      const body = new FormData();
      body.append("file", file, file.name);
      body.append("metadata_json", JSON.stringify(metadata));
      return apiRequest<Record<string, unknown>>("knowledge/documents", {
        method: "POST",
        body
      });
    },
    async onSuccess(response) {
      setLastReport(response.data);
      setFile(null);
      form.reset();
      if (fileInput.current) fileInput.current.value = "";
      await queryClient.invalidateQueries({ queryKey: queryKeys.knowledge });
    }
  });

  const toggle = useMutation({
    mutationFn: ({ docId, enabled }: { docId: string; enabled: boolean }) =>
      apiRequest<KnowledgeDocument>(`knowledge/documents/${encodeURIComponent(docId)}`, {
        method: "PATCH",
        body: { enabled }
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.knowledge })
  });

  const reindex = useMutation({
    mutationFn: () =>
      apiRequest<ReindexReport>("knowledge/reindex", {
        method: "POST",
        body: { force: true }
      }),
    onSuccess(response) {
      setLastReport(response.data as unknown as Record<string, unknown>);
      void queryClient.invalidateQueries({ queryKey: queryKeys.knowledge });
    }
  });

  return (
    <main className="knowledge-page">
      <header className="app-topbar">
        <Brand />
        <div className="topbar-actions">
          <HealthIndicator />
          <Button asChild tone="ghost">
            <Link href="/">返回任务首页</Link>
          </Button>
        </div>
      </header>

      <div className="knowledge-shell">
        <section className="knowledge-intro">
          <div className="eyebrow"><BookOpen size={14} />全局知识资产</div>
          <h1>材料知识库</h1>
          <p>
            管理获准使用的论文、报告和材料笔记。许可信息、索引状态和引用身份会随知识证据一起审查。
          </p>
        </section>

        <div className="knowledge-layout">
          <section className="panel ingest-panel">
            <div className="panel-header">
              <div>
                <h2>导入知识文档</h2>
                <p>许可未明确时，不会默认允许用于演示。</p>
              </div>
              <Upload size={19} />
            </div>
            <form
              className="panel-body form-stack"
              onSubmit={form.handleSubmit((values) => {
                if (!writeBlocker) ingest.mutate(values);
              })}
            >
              {writeBlocker ? (
                <p className="form-warning" role="status">
                  {writeBlocker}
                </p>
              ) : null}
              <label className="field">
                <span>文档文件 *</span>
                <input
                  ref={fileInput}
                  className="input file-input"
                  type="file"
                  accept=".pdf,.txt,.md,.markdown"
                  onChange={(event) => setFile(event.target.files?.[0] || null)}
                />
              </label>
              <label className="field">
                <span>标题 *</span>
                <input
                  className="input"
                  aria-invalid={Boolean(form.formState.errors.title)}
                  aria-describedby={
                    form.formState.errors.title ? "knowledge-title-error" : undefined
                  }
                  {...form.register("title")}
                />
                <FieldError
                  id="knowledge-title-error"
                  message={form.formState.errors.title?.message}
                />
              </label>
              <div className="two-fields">
                <label className="field">
                  <span>来源类型</span>
                  <select
                    className="select"
                    {...form.register("source_type")}
                  >
                    <option value="paper">论文</option>
                    <option value="report">报告</option>
                    <option value="material_note">材料笔记</option>
                    <option value="other">其他</option>
                  </select>
                </label>
                <label className="field">
                  <span>年份</span>
                  <input
                    className="input"
                    type="number"
                    min="1000"
                    max="3000"
                    aria-invalid={Boolean(form.formState.errors.year)}
                    aria-describedby={
                      form.formState.errors.year ? "knowledge-year-error" : undefined
                    }
                    {...form.register("year")}
                  />
                  <FieldError
                    id="knowledge-year-error"
                    message={form.formState.errors.year?.message}
                  />
                </label>
              </div>
              <label className="field">
                <span>规范引用 *</span>
                <textarea
                  className="textarea"
                  aria-invalid={Boolean(form.formState.errors.citation_text)}
                  aria-describedby={
                    form.formState.errors.citation_text
                      ? "knowledge-citation-error"
                      : undefined
                  }
                  {...form.register("citation_text")}
                  placeholder="作者、标题、期刊/来源、年份、DOI 等"
                />
                <FieldError
                  id="knowledge-citation-error"
                  message={form.formState.errors.citation_text?.message}
                />
              </label>
              <label className="field">
                <span>材料别名（逗号分隔）</span>
                <input
                  className="input"
                  {...form.register("material_aliases_text")}
                  placeholder="LaNiO3, LNO"
                />
              </label>
              <label className="field">
                <span>许可与来源说明 *</span>
                <textarea
                  className="textarea"
                  aria-invalid={Boolean(form.formState.errors.license_note)}
                  aria-describedby={
                    form.formState.errors.license_note
                      ? "knowledge-license-error"
                      : undefined
                  }
                  {...form.register("license_note")}
                  placeholder="说明公开许可、内部授权或使用限制"
                />
                <FieldError
                  id="knowledge-license-error"
                  message={form.formState.errors.license_note?.message}
                />
              </label>
              <label className="toggle-field">
                <input
                  type="checkbox"
                  {...form.register("allowed_for_demo")}
                />
                <span>确认该文档获准用于项目演示</span>
              </label>
              <Button
                tone="primary"
                type="submit"
                disabled={Boolean(writeBlocker) || !file || ingest.isPending}
                title={writeBlocker || undefined}
              >
                <ShieldCheck size={16} />
                {ingest.isPending ? "正在提取和索引…" : "导入并建立引用"}
              </Button>
              {ingest.isError ? <RequestError error={ingest.error} /> : null}
            </form>
          </section>

          <section className="panel document-panel">
            <div className="panel-header">
              <div>
                <h2>知识资产清单</h2>
                <p>{documentItems.length} 份受管文档</p>
              </div>
              <Button
                onClick={() => reindex.mutate()}
                disabled={Boolean(writeBlocker) || reindex.isPending}
                title={writeBlocker || undefined}
              >
                <RefreshCw size={15} />
                {reindex.isPending ? "重建中…" : "强制重建索引"}
              </Button>
            </div>

            {documents.isError ? (
              <div className="panel-body"><RequestError error={documents.error} /></div>
            ) : documentItems.length ? (
              <div className="document-list">
                {documentItems.map((document) => {
                  const enabled = document.status !== "disabled";
                  return (
                    <article key={document.doc_id}>
                      <span className="document-icon"><FileText size={18} /></span>
                      <div className="document-main">
                        <div>
                          <strong>{document.title}</strong>
                          <StatusBadge value={document.status} />
                        </div>
                        <p>{document.citation_text}</p>
                        <div className="document-meta">
                          <span>{document.source_type}</span>
                          <span>{document.year || "年份未填"}</span>
                          <span>{compactId(document.doc_id)}</span>
                          <span>{formatDate(document.created_at)}</span>
                        </div>
                        <small>{document.license_note}</small>
                      </div>
                      <button
                        className="document-toggle"
                        onClick={() => toggle.mutate({ docId: document.doc_id, enabled: !enabled })}
                        aria-label={`${enabled ? "停用" : "启用"} ${document.title}`}
                        disabled={Boolean(writeBlocker) || toggle.isPending}
                        title={writeBlocker || undefined}
                      >
                        {enabled ? <ToggleRight size={24} /> : <ToggleLeft size={24} />}
                      </button>
                    </article>
                  );
                })}
              </div>
            ) : (
              <EmptyState
                icon={FileText}
                title="知识库尚无文档"
                detail="导入获准使用的 PDF、TXT 或 Markdown 后，索引报告会显示页数、chunks 和警告。"
              />
            )}

            {toggle.isError ? <div className="panel-body"><RequestError error={toggle.error} /></div> : null}
            {reindex.isError ? <div className="panel-body"><RequestError error={reindex.error} /></div> : null}
          </section>
        </div>

        {lastReport ? (
          <section className="panel ingest-report">
            <div className="panel-header">
              <div>
                <h2>最近一次知识操作报告</h2>
                <p>该报告来自后端索引服务。</p>
              </div>
            </div>
            <pre>{JSON.stringify(lastReport, null, 2)}</pre>
          </section>
        ) : null}
      </div>
    </main>
  );
}

function FieldError({
  id,
  message
}: {
  id: string;
  message: string | undefined;
}) {
  return message ? (
    <small className="field-error" id={id}>
      {message}
    </small>
  ) : null;
}
