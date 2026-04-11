"use client"

import { useState } from "react"
import { Zap, Thermometer, Droplets, Wind } from "lucide-react"

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export function PredictionPanel() {
  const [form, setForm] = useState({
    temp: 12, humidity: 75, pressure: 1013,
    wind_speed: 6, hour: 14, is_weekend: 0, is_holiday: 0, month: 4
  })
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const predict = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      })
      const data = await res.json()
      setResult(data)
    } catch (e) {
      alert("API not running - start backend first!")
    }
    setLoading(false)
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-cyan-400" /> Input Parameters
        </h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="flex items-center gap-2 text-sm text-slate-400 mb-1">
              <Thermometer className="w-4 h-4" /> Temp (°C)
            </label>
            <input
              type="number"
              value={form.temp}
              onChange={(e) => setForm({...form, temp: Number(e.target.value)})}
              className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-white"
            />
          </div>
          <div>
            <label className="flex items-center gap-2 text-sm text-slate-400 mb-1">
              <Droplets className="w-4 h-4" /> Humidity (%)
            </label>
            <input
              type="number"
              value={form.humidity}
              onChange={(e) => setForm({...form, humidity: Number(e.target.value)})}
              className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-white"
            />
          </div>
          <div>
            <label className="flex items-center gap-2 text-sm text-slate-400 mb-1">
              <Wind className="w-4 h-4" /> Wind (m/s)
            </label>
            <input
              type="number"
              value={form.wind_speed}
              onChange={(e) => setForm({...form, wind_speed: Number(e.target.value)})}
              className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-white"
            />
          </div>
          <div>
            <label className="text-sm text-slate-400 mb-1 block">Hour</label>
            <input
              type="number"
              min={0} max={23}
              value={form.hour}
              onChange={(e) => setForm({...form, hour: Number(e.target.value)})}
              className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-white"
            />
          </div>
        </div>
        <button
          onClick={predict}
          disabled={loading}
          className="w-full mt-4 bg-cyan-600 hover:bg-cyan-500 text-white py-2 rounded-lg font-medium transition-colors"
        >
          {loading ? "Predicting..." : "Generate Prediction"}
        </button>
      </div>

      {result && (
        <div className="bg-slate-900 p-6 rounded-xl border border-cyan-500/30">
          <h2 className="text-lg font-semibold mb-4">Prediction Result</h2>
          <div className="text-center py-8">
            <div className="text-5xl font-bold text-cyan-400 mb-2">
              {result.predicted_demand_mw.toFixed(1)}
              <span className="text-xl text-slate-400 ml-2">MW</span>
            </div>
            <p className="text-slate-400">{result.interpretation}</p>
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="bg-slate-800 p-3 rounded">
              <span className="text-slate-400">Confidence</span>
              <p className="font-medium capitalize">{result.confidence}</p>
            </div>
            <div className="bg-slate-800 p-3 rounded">
              <span className="text-slate-400">Weather Score</span>
              <p className="font-medium">{result.weather_score}/100</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
