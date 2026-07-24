"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowUp, Database, Library, Merge, Sparkles } from "lucide-react";
import { useState } from "react";

import { RequestError } from "@/components/ui/request-error";
import { apiRequest } from "@/lib/api/client";
import { queryKeys } from "@/lib/api/query-keys";
import type {
  ImageAsset,
  UnifiedQueryRequest,
  UnifiedQueryResponse
} from "@/lib/api/types";
import { buildQueryScopeKey } from "@/lib/runs/selection";
import {
  type QueryMode,
  useWorkspaceStore
} from "@/lib/store/workspace";

const modes: Array<{ value: QueryMode; label: string; icon: typeof Sparkles }> = [
  { value: "auto", label: "自动", icon: Sparkles },
  { value: "analysis_data", label: "数据", icon: Database },
  { value: "material_knowledge", label: "知识", icon: Library },
  { value: "mixed", label: "混合", icon: Merge }
];

export function CommandComposer({
  jobId,
  image,
  runIds,
  writeBlocker,
  clarification,
  onAnswer
}: {
  jobId: string;
  image: ImageAsset | null;
  runIds: string[];
  writeBlocker: string | null;
  clarification: UnifiedQueryResponse | null;
  onAnswer: (answer: UnifiedQueryResponse, scope: string) => void;
}) {
  const queryClient = useQueryClient();
  const [question, setQuestion] = useState("");
  const [materialName, setMaterialName] = useState("");
  const [materialFormula, setMaterialFormula] = useState("");
  const [materialAliases, setMaterialAliases] = useState("");
  const mode = useWorkspaceStore((state) => state.queryMode);
  const setMode = useWorkspaceStore((state) => state.setQueryMode);
  const aliases = materialAliases
    .split(/[,，]/)
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 32);
  const hasConfirmedContext = Boolean(
    materialName.trim() || materialFormula.trim() || aliases.length
  );
  const needsClarification = Boolean(clarification?.needs_clarification);
  const needsMaterialContext = isMaterialContextClarification(clarification);

  const query = useMutation({
    mutationFn: ({
      requestJobId,
      body
    }: {
      scope: string;
      requestJobId: string;
      body: UnifiedQueryRequest;
    }) =>
      apiRequest<UnifiedQueryResponse>(`analyses/${encodeURIComponent(requestJobId)}/query`, {
        method: "POST",
        body
      }),
    onSuccess(response, variables) {
      onAnswer(response.data, variables.scope);
      setQuestion("");
      void queryClient.invalidateQueries({
        queryKey: queryKeys.queryHistory(variables.requestJobId)
      });
    }
  });

  function submitQuery() {
    const scopedRunIds = runIds.slice(0, 50);
    const imageId = image?.image_id ?? null;
    query.mutate({
      scope: buildQueryScopeKey(jobId, imageId, scopedRunIds),
      requestJobId: jobId,
      body: {
        question: question.trim(),
        query_type: mode,
        image_id: imageId,
        run_ids: scopedRunIds,
        material_context: hasConfirmedContext
          ? {
              name: materialName.trim() || null,
              formula: materialFormula.trim() || null,
              aliases,
              source: "user_confirmation"
            }
          : image?.material_name || image?.material_formula
            ? {
                name: image.material_name,
                formula: image.material_formula,
                aliases: [],
                source: "image_metadata"
              }
            : null
      }
    });
  }

  return (
    <div className="composer-wrap">
      {query.isError ? <RequestError error={query.error} /> : null}
      {writeBlocker ? (
        <p className="form-warning" role="status">
          {writeBlocker}
        </p>
      ) : null}
      {needsMaterialContext ? (
        <section className="clarification-context" aria-label="补充材料上下文">
          <div>
            <strong>需要补充材料上下文</strong>
            <span>确认后再次提问；这些字段只随本次请求发送。</span>
          </div>
          <input
            className="input"
            value={materialName}
            maxLength={255}
            onChange={(event) => setMaterialName(event.target.value)}
            placeholder="材料名称"
            aria-label="补充材料名称"
          />
          <input
            className="input"
            value={materialFormula}
            maxLength={255}
            onChange={(event) => setMaterialFormula(event.target.value)}
            placeholder="化学式"
            aria-label="补充材料化学式"
          />
          <input
            className="input"
            value={materialAliases}
            onChange={(event) => setMaterialAliases(event.target.value)}
            placeholder="别名，逗号分隔"
            aria-label="补充材料别名"
          />
        </section>
      ) : needsClarification ? (
        <p className="clarification-guidance" role="status">
          回答需要进一步澄清。请根据上方回答中的限制，补充运行范围、指标或实验条件后重试。
        </p>
      ) : null}
      <div className="command-composer">
        <div className="composer-modes">
          {modes.map((item) => {
            const Icon = item.icon;
            return (
              <button
                className={mode === item.value ? "active" : undefined}
                key={item.value}
                onClick={() => setMode(item.value)}
                aria-pressed={mode === item.value}
              >
                <Icon size={13} />
                {item.label}
              </button>
            );
          })}
        </div>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              if (question.trim() && !writeBlocker) submitQuery();
            }
          }}
          maxLength={2000}
          placeholder="询问当前实验，或让 NanoLoop 比较所选运行的确定性结果……"
          aria-label="询问当前实验"
        />
        <div className="composer-context">
          <span>
            {image ? image.filename : "未选图像"} · {runIds.length} 个运行作用域
          </span>
          <button
            type="button"
            aria-label="发送问题"
            onClick={submitQuery}
            disabled={Boolean(writeBlocker) || !question.trim() || query.isPending}
            title={writeBlocker || undefined}
          >
            <ArrowUp size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}

export function isMaterialContextClarification(
  response: UnifiedQueryResponse | null
): boolean {
  if (!response?.needs_clarification) return false;
  return (response.limitations ?? []).some(
    (limitation) =>
      limitation.includes("材料上下文") || limitation.includes("material_context")
  );
}
