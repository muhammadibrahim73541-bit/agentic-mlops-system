"use client"

import { useState, useEffect } from "react"
import { Sidebar } from "@/components/sidebar"
import { PredictionPanel } from "@/components/prediction-panel"

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("dashboard")

  return (
    <div className="flex h-screen bg-slate-950 text-white">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="flex-1 p-6 overflow-auto">
        <h1 className="text-2xl font-bold mb-6">Energy Demand Forecasting</h1>
        <PredictionPanel />
      </main>
    </div>
  )
}
