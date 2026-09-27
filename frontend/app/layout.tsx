import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AERIS — Environmental Risk & Intelligence System",
  description: "AI-Powered Air Quality Forecasting, Fuzzy Attention & Spatio-Temporal Intelligence Platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0B1120] text-slate-100 antialiased min-h-screen flex flex-col selection:bg-cyan-500/20 selection:text-cyan-300">
        <header className="sticky top-0 z-40 border-b border-slate-800 bg-[#0B1120]/80 backdrop-blur-md">
          <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
            <div className="flex items-center space-x-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 font-bold text-white shadow-lg shadow-cyan-500/20">
                A
              </div>
              <div>
                <span className="font-bold tracking-tight text-white text-lg">AERIS</span>
                <span className="ml-2 text-xs font-medium text-cyan-400 border border-cyan-500/30 px-2 py-0.5 rounded-full bg-cyan-950/40">
                  Phase 1 Dev Shell
                </span>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="hidden sm:flex items-center space-x-2 text-xs text-slate-400 bg-slate-900/80 px-3 py-1.5 rounded-md border border-slate-800">
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
                <span>System Initialized (Phase 1)</span>
              </div>
            </div>
          </div>
        </header>

        <main className="flex-1">{children}</main>

        <footer className="border-t border-slate-800/80 py-6 text-center text-xs text-slate-500">
          <div className="mx-auto max-w-7xl px-4">
            AERIS: Adaptive Environmental Risk & Intelligence System &bull; Research & Development Environment
          </div>
        </footer>
      </body>
    </html>
  );
}
