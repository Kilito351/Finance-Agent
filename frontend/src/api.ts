const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options
  })
  if (!response.ok) throw new Error(`请求失败：${response.status}`)
  return response.json() as Promise<T>
}

export interface Step { agent: string; title: string; content: string }
export interface Analysis {
  trace_id: string
  mode: 'demo' | 'llm'
  research: Step
  risk: Step
  strategy: Step
  report: string
  sources: { title: string; category: string; score: number }[]
  disclaimer: string
}

export const analyze = (query: string, stockCode: string) => request<Analysis>('/agents/analyze', {
  method: 'POST', body: JSON.stringify({ query, stock_code: stockCode || null })
})
export const getStocks = () => request<{ as_of: string; items: Stock[] }>('/market/stocks')
export const getKnowledge = () => request<{ total: number; items: Knowledge[] }>('/knowledge')
export const getMetrics = () => request<Metrics>('/llmops/metrics')

export interface Stock { code: string; name: string; price: number; change_pct: number }
export interface Knowledge { title: string; category: string; content: string }
export interface Metrics { calls: number; success_rate: number; total_tokens: number; average_latency_ms: number; mode: string }

