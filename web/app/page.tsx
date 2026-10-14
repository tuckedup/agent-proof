import React from "react";
import { ShieldAlert, Activity, CheckCircle2, Terminal } from "lucide-react";

export default function HomePage() {
  return (
    <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
      <header className="border-b border-neutral-800 pb-5 mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
              <ShieldAlert className="h-6 w-6 text-blue-500" />
              agentproof Lab
            </h1>
            <p className="text-sm text-neutral-400 mt-1">
              Empirical reliability & failure analysis for ChemDesk agent
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="inline-flex items-center rounded-md bg-blue-950/80 px-2.5 py-1 text-xs font-medium text-blue-400 ring-1 ring-inset ring-blue-500/20">
              Dual-Control Active
            </span>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-neutral-900/60 border border-neutral-800 p-4 rounded-lg">
          <div className="flex items-center justify-between text-neutral-400 mb-2">
            <span className="text-xs uppercase font-semibold">Pass^1 Rate</span>
            <Activity className="h-4 w-4" />
          </div>
          <div className="text-2xl font-bold text-white">Pending</div>
          <div className="text-xs text-neutral-500 mt-1">Dual-mode benchmark</div>
        </div>

        <div className="bg-neutral-900/60 border border-neutral-800 p-4 rounded-lg">
          <div className="flex items-center justify-between text-neutral-400 mb-2">
            <span className="text-xs uppercase font-semibold">Policy Violations</span>
            <ShieldAlert className="h-4 w-4" />
          </div>
          <div className="text-2xl font-bold text-white">0.0%</div>
          <div className="text-xs text-neutral-500 mt-1">Zero hazmat/credit breaches</div>
        </div>

        <div className="bg-neutral-900/60 border border-neutral-800 p-4 rounded-lg">
          <div className="flex items-center justify-between text-neutral-400 mb-2">
            <span className="text-xs uppercase font-semibold">Human Labeled</span>
            <CheckCircle2 className="h-4 w-4" />
          </div>
          <div className="text-2xl font-bold text-white">0</div>
          <div className="text-xs text-neutral-500 mt-1">Ground truth traces</div>
        </div>

        <div className="bg-neutral-900/60 border border-neutral-800 p-4 rounded-lg">
          <div className="flex items-center justify-between text-neutral-400 mb-2">
            <span className="text-xs uppercase font-semibold">Total Trajectories</span>
            <Terminal className="h-4 w-4" />
          </div>
          <div className="text-2xl font-bold text-white">0</div>
          <div className="text-xs text-neutral-500 mt-1">Executed episodes</div>
        </div>
      </div>
    </main>
  );
}
