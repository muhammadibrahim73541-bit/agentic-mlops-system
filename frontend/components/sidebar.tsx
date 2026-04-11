"use client"

import { Zap, LayoutDashboard, History, BarChart3, Brain } from "lucide-react"

export function Sidebar({ activeTab, setActiveTab }: { activeTab: string, setActiveTab: (t: string) => void }) {
  const items = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "history", label: "History", icon: History },
    { id: "analytics", label: "Analytics", icon: BarChart3 },
    { id: "performance", label: "Model", icon: Brain },
  ]

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 p-4">
      <div className="flex items-center gap-2 mb-8 px-2">
        <Zap className="w-6 h-6 text-cyan-400" />
        <span className="font-bold text-lg">ENERGY AI</span>
      </div>
      <nav className="space-y-1">
        {items.map((item) => {
          const Icon = item.icon
          const isActive = activeTab === item.id
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                isActive ? "bg-cyan-600 text-white" : "text-slate-400 hover:bg-slate-800"
              }`}
            >
              <Icon className="w-4 h-4" />
              {item.label}
            </button>
          )
        })}
      </nav>
    </aside>
  )
}
