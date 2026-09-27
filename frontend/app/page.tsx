"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  CheckCircle2,
  Layers,
  Server,
  Sparkles,
  Workflow,
} from "lucide-react";
import { ApiClient } from "@/lib/api";
import { HealthStatus } from "@/types";

export default function HomePage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    ApiClient.getHealth()
      .then((data) => {
        if (isMounted) {
          setHealth(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || "Failed to reach backend");
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8 space-y-10">
      {/* Hero Header */}
      <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900/90 to-slate-950 p-8 shadow-2xl relative overflow-hidden">
        <div className="absolute -right-16 -top-16 h-64 w-64 rounded-full bg-cyan-500/10 blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center space-x-2 rounded-full border border-cyan-500/30 bg-cyan-950/30 px-3 py-1 text-xs font-semibold text-cyan-400">
            <Sparkles className="h-3.5 w-3.5" />
            <span>Applied Research & Engineering Workspace</span>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            Adaptive Environmental Risk & Intelligence System
          </h1>
          <p className="text-slate-400 text-sm leading-relaxed sm:text-base">
            Full-stack research platform for air quality forecasting, hybrid
            spatio-temporal neural architectures (1D-CNN + BiLSTM), and fuzzy
            logic-modulated temporal attention.
          </p>
        </div>
      </div>

      {/* Grid: System Status & Architecture Overview */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Backend Connectivity Card */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow-md flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-4 border-b border-slate-800/80">
              <div className="flex items-center space-x-2.5">
                <Server className="h-5 w-5 text-cyan-400" />
                <h3 className="font-semibold text-white">FastAPI Backend Status</h3>
              </div>
              <span className="text-xs font-mono text-slate-500">/api/v1/health</span>
            </div>

            <div className="mt-4 space-y-3">
              {loading && (
                <div className="flex items-center space-x-2 text-sm text-slate-400">
                  <div className="h-4 w-4 rounded-full border-2 border-cyan-500 border-t-transparent animate-spin" />
                  <span>Connecting to backend service...</span>
                </div>
              )}

              {error && (
                <div className="rounded-lg border border-rose-900/50 bg-rose-950/30 p-3 text-xs text-rose-300">
                  <p className="font-semibold">Backend Service Offline</p>
                  <p className="mt-1 text-slate-400">{error}</p>
                  <p className="mt-2 text-[10px] text-slate-500 font-mono">
                    Start backend: uvicorn backend.app.main:app --port 8000
                  </p>
                </div>
              )}

              {health && (
                <div className="space-y-2.5 text-xs font-mono">
                  <div className="flex justify-between py-1 border-b border-slate-800/40">
                    <span className="text-slate-400">Health Status:</span>
                    <span className="text-emerald-400 font-semibold">{health.status}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-800/40">
                    <span className="text-slate-400">API Version:</span>
                    <span className="text-slate-200">{health.version}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-800/40">
                    <span className="text-slate-400">Execution Target:</span>
                    <span className="text-cyan-300 font-semibold">{health.device}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-800/40">
                    <span className="text-slate-400">Environment:</span>
                    <span className="text-slate-200">{health.environment}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-slate-400">Uptime:</span>
                    <span className="text-slate-200">{health.uptime_seconds}s</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="mt-6 pt-3 border-t border-slate-800 text-[11px] text-slate-500 flex items-center justify-between">
            <span>REST API (OpenAPI 3.1)</span>
            <span className="text-emerald-500 flex items-center gap-1">
              <CheckCircle2 className="h-3 w-3" /> Operational
            </span>
          </div>
        </div>

        {/* Development Milestones Overview */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow-md flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2.5 pb-4 border-b border-slate-800/80">
              <Workflow className="h-5 w-5 text-cyan-400" />
              <h3 className="font-semibold text-white">Engineering Milestones</h3>
            </div>

            <div className="mt-4 space-y-2 text-xs">
              <div className="flex items-center justify-between p-2 rounded-md bg-emerald-950/20 border border-emerald-800/30">
                <span className="font-semibold text-emerald-300">M1: Project Foundation</span>
                <span className="text-[10px] text-emerald-400 font-bold">COMPLETE</span>
              </div>
              <div className="flex items-center justify-between p-2 rounded-md bg-emerald-950/20 border border-emerald-800/30">
                <span className="font-semibold text-emerald-300">M2: App Infrastructure</span>
                <span className="text-[10px] text-emerald-400 font-bold">COMPLETE</span>
              </div>
              <div className="flex items-center justify-between p-2 rounded-md bg-cyan-950/30 border border-cyan-800/40">
                <span className="font-semibold text-cyan-300">M3: Data Ingestion & Checks</span>
                <span className="text-[10px] text-cyan-400 font-bold">NEXT</span>
              </div>
              <div className="flex items-center justify-between p-2 rounded-md bg-slate-950/40 border border-slate-800/40 text-slate-400">
                <span>M4–M9: Models, XAI & Serving</span>
                <span className="text-[10px] text-slate-500">PLANNED</span>
              </div>
            </div>
          </div>

          <div className="mt-6 pt-3 border-t border-slate-800 text-[11px] text-slate-500 flex items-center justify-between">
            <span>Milestone-Driven Roadmap</span>
            <span className="font-mono text-cyan-400">DEVELOPMENT_PLAN.md</span>
          </div>
        </div>

        {/* Planned Dashboard Views Preview */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow-md flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2.5 pb-4 border-b border-slate-800/80">
              <Layers className="h-5 w-5 text-cyan-400" />
              <h3 className="font-semibold text-white">Planned Dashboard Modules</h3>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-2 text-[11px] text-slate-300">
              <div className="p-2 rounded bg-slate-950/40 border border-slate-800/40 flex items-center gap-1.5">
                <div className="h-1.5 w-1.5 rounded-full bg-cyan-400" /> Overview
              </div>
              <div className="p-2 rounded bg-slate-950/40 border border-slate-800/40 flex items-center gap-1.5">
                <div className="h-1.5 w-1.5 rounded-full bg-cyan-400" /> Air Quality Map
              </div>
              <div className="p-2 rounded bg-slate-950/40 border border-slate-800/40 flex items-center gap-1.5">
                <div className="h-1.5 w-1.5 rounded-full bg-cyan-400" /> Forecast Studio
              </div>
              <div className="p-2 rounded bg-slate-950/40 border border-slate-800/40 flex items-center gap-1.5">
                <div className="h-1.5 w-1.5 rounded-full bg-cyan-400" /> Model Lab
              </div>
              <div className="p-2 rounded bg-slate-950/40 border border-slate-800/40 flex items-center gap-1.5">
                <div className="h-1.5 w-1.5 rounded-full bg-cyan-400" /> Explainability
              </div>
              <div className="p-2 rounded bg-slate-950/40 border border-slate-800/40 flex items-center gap-1.5">
                <div className="h-1.5 w-1.5 rounded-full bg-cyan-400" /> Fuzzy Intelligence
              </div>
              <div className="p-2 rounded bg-slate-950/40 border border-slate-800/40 flex items-center gap-1.5">
                <div className="h-1.5 w-1.5 rounded-full bg-cyan-400" /> What-If Simulator
              </div>
              <div className="p-2 rounded bg-slate-950/40 border border-slate-800/40 flex items-center gap-1.5">
                <div className="h-1.5 w-1.5 rounded-full bg-cyan-400" /> Data Explorer
              </div>
            </div>
          </div>

          <div className="mt-6 pt-3 border-t border-slate-800 text-[11px] text-slate-500 flex items-center justify-between">
            <span>Interface Blueprint</span>
            <span className="font-mono text-cyan-400">DASHBOARD_SPECIFICATION.md</span>
          </div>
        </div>
      </div>
    </div>
  );
}
