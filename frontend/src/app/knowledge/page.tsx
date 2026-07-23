import type { Metadata } from "next";

import { KnowledgeManager } from "@/components/knowledge/knowledge-manager";

export const metadata: Metadata = {
  title: "材料知识库"
};

export default function KnowledgePage() {
  return <KnowledgeManager />;
}
