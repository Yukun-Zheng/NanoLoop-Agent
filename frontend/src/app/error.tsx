"use client";

import { RotateCcw } from "lucide-react";

export default function GlobalError({
  error,
  reset
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <main className="centered-state">
      <div className="empty-symbol">!</div>
      <h1>界面遇到了问题</h1>
      <p>{error.message || "当前页面无法继续渲染。"}</p>
      {error.digest ? <code>事件 {error.digest}</code> : null}
      <button className="button button-primary" onClick={reset}>
        <RotateCcw size={16} />
        重新载入
      </button>
    </main>
  );
}
