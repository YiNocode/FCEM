<template>
  <div class="app-shell" v-if="ready">
    <ControlPanel
      v-model="config"
      :options="options"
      v-model:play-speed="playSpeed"
      @apply="applyAndReset"
    />
    <div class="main">
      <div class="toolbar">
        <button class="primary" :disabled="!connected || playing" @click="play">播放</button>
        <button :disabled="!connected" @click="pause">暂停</button>
        <button :disabled="!connected || playing || summary.done" @click="stepOnce">单步</button>
        <button :disabled="!connected" @click="applyAndReset">重置</button>
        <span style="margin-left: auto; color: #94a3b8; font-size: 13px">
          {{ connected ? '已连接' : '连接中…' }}
        </span>
      </div>
      <SimCanvas :meta="meta" :frames="frames" :summary="summary" />
      <div class="status-bar">
        v_p={{ meta?.pursuer_vmax }} m/s · v_e={{ meta?.evader_vmax }} m/s ·
        dt={{ meta?.dt }}s · 轨迹点数 {{ frames.length }}
      </div>
    </div>
  </div>
  <div v-else style="padding: 24px">正在加载…</div>
</template>

<script setup>
import { onMounted, onUnmounted, reactive, ref } from 'vue'
import ControlPanel from './components/ControlPanel.vue'
import SimCanvas from './components/SimCanvas.vue'
import { createSession, defaultConfig, fetchOptions, wsUrl } from './api.js'

const options = ref({ methods: [], scenarios: [], layers: [], evader_policies: [], defaults: {} })
const config = reactive({})
const ready = ref(false)

const sessionId = ref(null)
const meta = ref(null)
const frames = ref([])
const summary = ref({ step: 0, captured: false, failed: false, done: false })
const playSpeed = ref(1)
const playing = ref(false)
const connected = ref(false)

let ws = null

function handleMessage(msg) {
  if (msg.type === 'meta') {
    meta.value = msg.meta
    if (msg.summary) summary.value = msg.summary
    frames.value = []
    return
  }
  if (msg.type === 'frame' || msg.type === 'done') {
    if (msg.frame) frames.value = [...frames.value, msg.frame]
    if (msg.summary) summary.value = msg.summary
    if (msg.type === 'done') playing.value = false
  }
}

function connectWs() {
  if (!sessionId.value) return
  if (ws) {
    ws.close()
    ws = null
  }
  connected.value = false
  ws = new WebSocket(wsUrl(sessionId.value))
  ws.onopen = () => {
    connected.value = true
  }
  ws.onmessage = (ev) => {
    handleMessage(JSON.parse(ev.data))
  }
  ws.onclose = () => {
    connected.value = false
    playing.value = false
  }
  ws.onerror = () => {
    connected.value = false
  }
}

function send(action, extra = {}) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return
  ws.send(JSON.stringify({ action, ...extra }))
}

function play() {
  playing.value = true
  send('play', { speed: playSpeed.value })
}

function pause() {
  playing.value = false
  send('pause')
}

function stepOnce() {
  send('step')
}

function applyAndReset() {
  playing.value = false
  const payload = { ...config, remove_layers: [...(config.remove_layers || [])] }
  send('configure', { config: payload })
}

async function bootstrap() {
  options.value = await fetchOptions()
  Object.assign(config, defaultConfig(options.value))
  const created = await createSession(config)
  sessionId.value = created.session_id
  meta.value = created.meta
  summary.value = created.summary
  frames.value = []
  ready.value = true
  connectWs()
}

onMounted(() => {
  bootstrap().catch((err) => {
    alert(err.message || String(err))
  })
})

onUnmounted(() => {
  if (ws) ws.close()
})
</script>
