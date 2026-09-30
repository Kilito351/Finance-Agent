<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { analyze, getKnowledge, getMetrics, getStocks, type Analysis, type Knowledge, type Metrics, type Stock } from './api'

type Page = 'analysis' | 'market' | 'knowledge' | 'ops'
const page = ref<Page>('analysis')
const query = ref('分析贵州茅台的基本面、主要风险与配置策略')
const stockCode = ref('600519')
const loading = ref(false)
const error = ref('')
const result = ref<Analysis | null>(null)
const stocks = ref<Stock[]>([])
const knowledge = ref<Knowledge[]>([])
const metrics = ref<Metrics | null>(null)

const nav = [
  { id: 'analysis' as Page, icon: '✦', label: '智能分析' },
  { id: 'market' as Page, icon: '⌁', label: '市场看板' },
  { id: 'knowledge' as Page, icon: '▤', label: '知识库' },
  { id: 'ops' as Page, icon: '◉', label: 'LLMOps' }
]
const title = computed(() => nav.find(item => item.id === page.value)?.label)

async function runAnalysis() {
  loading.value = true; error.value = ''
  try { result.value = await analyze(query.value, stockCode.value); await loadMetrics() }
  catch (reason) { error.value = reason instanceof Error ? reason.message : '分析失败' }
  finally { loading.value = false }
}
async function loadMetrics() { metrics.value = await getMetrics() }

onMounted(async () => {
  const settled = await Promise.allSettled([getStocks(), getKnowledge(), getMetrics()])
  if (settled[0].status === 'fulfilled') stocks.value = settled[0].value.items
  if (settled[1].status === 'fulfilled') knowledge.value = settled[1].value.items
  if (settled[2].status === 'fulfilled') metrics.value = settled[2].value
})
</script>

<template>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand"><span class="brand-mark">F</span><div><strong>Finance Agent</strong><small>AI INVESTMENT LAB</small></div></div>
      <nav>
        <button v-for="item in nav" :key="item.id" :class="{ active: page === item.id }" @click="page = item.id">
          <span>{{ item.icon }}</span>{{ item.label }}
        </button>
      </nav>
      <div class="status"><i></i><div><strong>服务已连接</strong><small>{{ metrics?.mode === 'llm' ? 'LLM 模式' : '本地演示模式' }}</small></div></div>
    </aside>

    <main>
      <header><div><p>FINANCIAL INTELLIGENCE</p><h1>{{ title }}</h1></div><span class="demo-badge">演示数据 · 非实时</span></header>

      <section v-if="page === 'analysis'" class="analysis-page">
        <div class="hero">
          <div><span class="eyebrow">MULTI-AGENT RESEARCH</span><h2>把问题交给一支<br><em>AI 投研团队</em></h2><p>投研、风控与策略 Agent 并行协作，输出带检索依据的结构化观点。</p></div>
          <div class="orbit"><span>投研</span><span>风控</span><span>策略</span><b>AI</b></div>
        </div>
        <div class="composer">
          <div class="input-row"><input v-model="stockCode" aria-label="股票代码" placeholder="股票代码（可选）" /><textarea v-model="query" aria-label="分析问题" rows="2" placeholder="输入你的金融分析问题"></textarea><button :disabled="loading || query.length < 2" @click="runAnalysis">{{ loading ? '协作中…' : '开始分析 →' }}</button></div>
          <div class="suggestions"><span>试试：</span><button @click="query='分析该公司的盈利质量和现金流'">盈利质量</button><button @click="query='识别该标的的主要下行风险'">下行风险</button><button @click="query='给出审慎的仓位管理框架'">仓位框架</button></div>
        </div>
        <p v-if="error" class="error">{{ error }}。请确认后端已在 8000 端口启动。</p>
        <div v-if="result" class="results">
          <article v-for="step in [result.research, result.risk, result.strategy]" :key="step.agent" class="agent-card">
            <div class="card-top"><span>{{ step.agent === 'researcher' ? '01' : step.agent === 'risk' ? '02' : '03' }}</span><div><small>{{ step.agent.toUpperCase() }}</small><h3>{{ step.title }}</h3></div></div><p>{{ step.content }}</p>
          </article>
          <article class="report"><div><small>FINAL SYNTHESIS</small><h3>综合研判</h3></div><p>{{ result.report.replaceAll('## ', '').replaceAll('\n', ' ') }}</p><footer><span>Trace {{ result.trace_id.slice(0, 8) }}</span><span>{{ result.mode === 'llm' ? '真实模型' : '本地演示' }}</span></footer></article>
          <div class="sources"><strong>检索依据</strong><span v-for="source in result.sources" :key="source.title">{{ source.title }} · {{ source.category }}</span></div>
          <p class="disclaimer">{{ result.disclaimer }}</p>
        </div>
      </section>

      <section v-else-if="page === 'market'" class="grid-page">
        <div class="section-intro"><span class="eyebrow">MARKET SNAPSHOT</span><h2>市场概览</h2><p>用于界面演示的静态行情卡片，不代表实时价格。</p></div>
        <div class="stock-grid"><article v-for="stock in stocks" :key="stock.code"><div><span>{{ stock.code }}</span><b :class="stock.change_pct >= 0 ? 'up' : 'down'">{{ stock.change_pct >= 0 ? '+' : '' }}{{ stock.change_pct }}%</b></div><h3>{{ stock.name }}</h3><p>¥ {{ stock.price.toFixed(2) }}</p></article></div>
      </section>

      <section v-else-if="page === 'knowledge'" class="grid-page">
        <div class="section-intro"><span class="eyebrow">RETRIEVAL KNOWLEDGE</span><h2>知识库</h2><p>{{ knowledge.length }} 个可检索知识片段，为 Agent 提供可追溯上下文。</p></div>
        <div class="knowledge-list"><article v-for="doc in knowledge" :key="doc.title"><span>{{ doc.category }}</span><h3>{{ doc.title }}</h3><p>{{ doc.content }}</p></article></div>
      </section>

      <section v-else class="grid-page">
        <div class="section-intro"><span class="eyebrow">MODEL OPERATIONS</span><h2>LLMOps 运行指标</h2><p>轻量进程内指标；生产环境可替换为 Prometheus 与持久化存储。</p></div>
        <div class="metric-grid"><article><span>总调用</span><strong>{{ metrics?.calls ?? 0 }}</strong></article><article><span>成功率</span><strong>{{ ((metrics?.success_rate ?? 1) * 100).toFixed(0) }}%</strong></article><article><span>Token</span><strong>{{ metrics?.total_tokens ?? 0 }}</strong></article><article><span>平均延迟</span><strong>{{ metrics?.average_latency_ms ?? 0 }} ms</strong></article></div>
      </section>
    </main>
  </div>
</template>

