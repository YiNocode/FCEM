<template>
  <div class="canvas-wrap" ref="wrapRef">
    <canvas ref="canvasRef"></canvas>
    <div class="hud" v-if="meta">
      <div><strong>{{ meta.method }}</strong> · {{ meta.scenario }}</div>
      <div>步数 {{ summary.step }} / {{ meta.max_steps }}</div>
      <div v-if="lastFrame">
        D_ang={{ lastFrame.metrics.D_ang.toFixed(3) }}
        C_cov={{ lastFrame.metrics.C_cov.toFixed(3) }}
        G_max={{ lastFrame.metrics.G_max_deg.toFixed(0) }}°
        C_sync={{ lastFrame.metrics.C_sync.toFixed(3) }}
      </div>
      <div v-if="summary.captured" class="ok">已捕获 @ step {{ summary.capture_step }}</div>
      <div v-else-if="summary.failed" class="bad">失败: {{ summary.failure_reason }}</div>
      <div v-else-if="summary.done" class="bad">超时未捕获</div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref, watch } from 'vue'

const props = defineProps({
  meta: { type: Object, default: null },
  frames: { type: Array, default: () => [] },
  summary: { type: Object, default: () => ({}) },
})

const wrapRef = ref(null)
const canvasRef = ref(null)
let ro = null

const lastFrame = ref(null)

const COLORS = ['#38bdf8', '#f472b6', '#a78bfa', '#fbbf24', '#34d399', '#fb7185']

function resize() {
  const canvas = canvasRef.value
  const wrap = wrapRef.value
  if (!canvas || !wrap) return
  const dpr = window.devicePixelRatio || 1
  const w = wrap.clientWidth
  const h = wrap.clientHeight
  canvas.width = Math.max(1, Math.floor(w * dpr))
  canvas.height = Math.max(1, Math.floor(h * dpr))
  canvas.style.width = `${w}px`
  canvas.style.height = `${h}px`
  draw()
}

function worldToScreen(x, y, world, w, h, pad = 28) {
  const xmin = world.xmin ?? 0
  const xmax = world.xmax ?? 40
  const ymin = world.ymin ?? 0
  const ymax = world.ymax ?? 40
  const sx = pad + ((x - xmin) / (xmax - xmin)) * (w - 2 * pad)
  const sy = h - pad - ((y - ymin) / (ymax - ymin)) * (h - 2 * pad)
  return [sx, sy]
}

function draw() {
  const canvas = canvasRef.value
  if (!canvas || !props.meta) return
  const ctx = canvas.getContext('2d')
  const dpr = window.devicePixelRatio || 1
  const w = canvas.width
  const h = canvas.height
  const world = props.meta.world

  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.clearRect(0, 0, w, h)
  ctx.scale(dpr, dpr)
  const cw = w / dpr
  const ch = h / dpr

  ctx.fillStyle = '#0b1015'
  ctx.fillRect(0, 0, cw, ch)

  // grid
  ctx.strokeStyle = '#1e293b'
  ctx.lineWidth = 1
  const gridN = 8
  for (let i = 0; i <= gridN; i++) {
    const t = i / gridN
    const x0 = 28 + t * (cw - 56)
    const y0 = 28 + t * (ch - 56)
    ctx.beginPath()
    ctx.moveTo(x0, 28)
    ctx.lineTo(x0, ch - 28)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(28, y0)
    ctx.lineTo(cw - 28, y0)
    ctx.stroke()
  }

  // obstacles
  for (const obs of props.meta.obstacles || []) {
    const [cx, cy] = worldToScreen(obs.center[0], obs.center[1], world, cw, ch)
    const [rx] = worldToScreen(obs.center[0] + obs.radius, obs.center[1], world, cw, ch)
    const rPx = Math.abs(rx - cx)
    ctx.beginPath()
    ctx.arc(cx, cy, rPx, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(100, 116, 139, 0.35)'
    ctx.fill()
    ctx.strokeStyle = '#64748b'
    ctx.setLineDash([4, 4])
    ctx.stroke()
    ctx.setLineDash([])
  }

  const frames = props.frames
  if (!frames.length) return

  // trajectories
  const nP = frames[0].pursuers?.length || 0
  for (let pi = 0; pi < nP; pi++) {
    ctx.strokeStyle = COLORS[pi % COLORS.length]
    ctx.lineWidth = 1.5
    ctx.globalAlpha = 0.55
    ctx.beginPath()
    frames.forEach((f, idx) => {
      const p = f.pursuers[pi]
      const [sx, sy] = worldToScreen(p[0], p[1], world, cw, ch)
      if (idx === 0) ctx.moveTo(sx, sy)
      else ctx.lineTo(sx, sy)
    })
    ctx.stroke()
    ctx.globalAlpha = 1
  }

  ctx.strokeStyle = '#e2e8f0'
  ctx.lineWidth = 2
  ctx.beginPath()
  frames.forEach((f, idx) => {
    const [sx, sy] = worldToScreen(f.evader[0], f.evader[1], world, cw, ch)
    if (idx === 0) ctx.moveTo(sx, sy)
    else ctx.lineTo(sx, sy)
  })
  ctx.stroke()

  const frame = frames[frames.length - 1]
  lastFrame.value = frame

  // capture radius
  const capR = props.meta.capture_radius || 1.8
  const [ex, ey] = worldToScreen(frame.evader[0], frame.evader[1], world, cw, ch)
  const [rx] = worldToScreen(frame.evader[0] + capR, frame.evader[1], world, cw, ch)
  ctx.beginPath()
  ctx.arc(ex, ey, Math.abs(rx - ex), 0, Math.PI * 2)
  ctx.strokeStyle = '#4ade80'
  ctx.setLineDash([6, 4])
  ctx.stroke()
  ctx.setLineDash([])

  // FCEM ring R
  if (frame.R > 0 && frame.center) {
    const [cx, cy] = worldToScreen(frame.center[0], frame.center[1], world, cw, ch)
    const [rx2] = worldToScreen(frame.center[0] + frame.R, frame.center[1], world, cw, ch)
    ctx.beginPath()
    ctx.arc(cx, cy, Math.abs(rx2 - cx), 0, Math.PI * 2)
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)'
    ctx.lineWidth = 1
    ctx.stroke()
  }

  // slots + assignment lines
  if (frame.slots && frame.assignment) {
    frame.slots.forEach((slot, si) => {
      const [sx, sy] = worldToScreen(slot[0], slot[1], world, cw, ch)
      ctx.beginPath()
      ctx.arc(sx, sy, 4, 0, Math.PI * 2)
      ctx.fillStyle = '#38bdf8'
      ctx.fill()
      const pi = frame.assignment.indexOf(si)
      if (pi >= 0 && frame.pursuers[pi]) {
        const p = frame.pursuers[pi]
        const [px, py] = worldToScreen(p[0], p[1], world, cw, ch)
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(px, py)
        ctx.lineTo(sx, sy)
        ctx.stroke()
      }
    })
  }

  // agents
  frame.pursuers.forEach((p, i) => {
    const [px, py] = worldToScreen(p[0], p[1], world, cw, ch)
    ctx.fillStyle = COLORS[i % COLORS.length]
    ctx.beginPath()
    ctx.moveTo(px, py - 7)
    ctx.lineTo(px - 6, py + 5)
    ctx.lineTo(px + 6, py + 5)
    ctx.closePath()
    ctx.fill()
  })

  ctx.fillStyle = '#f8fafc'
  ctx.beginPath()
  const spikes = 5
  const outer = 9
  const inner = 4
  for (let i = 0; i < spikes * 2; i++) {
    const r = i % 2 === 0 ? outer : inner
    const ang = (Math.PI / spikes) * i - Math.PI / 2
    const sx = ex + Math.cos(ang) * r
    const sy = ey + Math.sin(ang) * r
    if (i === 0) ctx.moveTo(sx, sy)
    else ctx.lineTo(sx, sy)
  }
  ctx.closePath()
  ctx.fill()
}

watch(() => [props.frames, props.meta], draw, { deep: true })

onMounted(() => {
  resize()
  ro = new ResizeObserver(resize)
  if (wrapRef.value) ro.observe(wrapRef.value)
})

onUnmounted(() => {
  if (ro) ro.disconnect()
})
</script>
