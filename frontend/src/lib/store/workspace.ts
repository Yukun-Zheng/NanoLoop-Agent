import { create } from "zustand";

export type WorkspaceStage = "project" | "roi" | "models" | "runs" | "results" | "agent";
export type InspectorTab = "system" | "model" | "quality" | "provenance" | "evidence";
export type QueryMode = "auto" | "analysis_data" | "material_knowledge" | "mixed";

type WorkspaceState = {
  activeImageId: string | null;
  activeRunId: string | null;
  selectedRunIds: string[];
  stage: WorkspaceStage;
  inspectorTab: InspectorTab;
  queryMode: QueryMode;
  queryDraft: string;
  railCollapsed: boolean;
  setActiveImage: (value: string | null) => void;
  setActiveRun: (value: string | null) => void;
  setSelectedRuns: (value: string[]) => void;
  setStage: (value: WorkspaceStage) => void;
  setInspectorTab: (value: InspectorTab) => void;
  setQueryMode: (value: QueryMode) => void;
  setQueryDraft: (value: string) => void;
  toggleRail: () => void;
};

export const useWorkspaceStore = create<WorkspaceState>((set) => ({
  activeImageId: null,
  activeRunId: null,
  selectedRunIds: [],
  stage: "project",
  inspectorTab: "system",
  queryMode: "auto",
  queryDraft: "",
  railCollapsed: false,
  setActiveImage: (activeImageId) => set({ activeImageId }),
  setActiveRun: (activeRunId) => set({ activeRunId }),
  setSelectedRuns: (selectedRunIds) => set({ selectedRunIds }),
  setStage: (stage) => set({ stage }),
  setInspectorTab: (inspectorTab) => set({ inspectorTab }),
  setQueryMode: (queryMode) => set({ queryMode }),
  setQueryDraft: (queryDraft) => set({ queryDraft }),
  toggleRail: () => set((state) => ({ railCollapsed: !state.railCollapsed }))
}));
