const API_BASE = ''

export async function fetchOptions() {
  const res = await fetch(`${API_BASE}/api/options`)
  if (!res.ok) throw new Error('无法加载配置选项')
  return res.json()
}

export async function createSession(config) {
  const res = await fetch(`${API_BASE}/api/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!res.ok) throw new Error('创建会话失败')
  return res.json()
}

export function wsUrl(sessionId) {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host = window.location.host
  return `${proto}://${host}/api/ws/${sessionId}`
}

export function defaultConfig(options) {
  return { ...(options?.defaults || {}) }
}
