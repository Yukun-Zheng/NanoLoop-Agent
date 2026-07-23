import type { Metadata, Viewport } from "next";

import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "NanoLoop Agent",
    template: "%s · NanoLoop Agent"
  },
  description: "面向材料显微分析的可追溯科研智能体指挥中心"
};

export const viewport: Viewport = {
  colorScheme: "light",
  themeColor: "#f7f8fc",
  width: "device-width",
  initialScale: 1
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
