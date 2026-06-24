<template>
  <aside class="panel">
    <h2 style="margin: 0 0 12px; font-size: 18px">FCEM 实时可视化</h2>

    <div class="field">
      <label>追捕算法</label>
      <select v-model="local.method">
        <option v-for="m in options.methods" :key="m" :value="m">{{ m }}</option>
      </select>
    </div>

    <div class="field">
      <label>场景类型</label>
      <select v-model="local.scenario">
        <option v-for="s in options.scenarios" :key="s.id" :value="s.id">{{ s.label }}</option>
      </select>
    </div>

    <div class="field" v-if="local.scenario === 'random_obstacles'">
      <label>障碍物数量 ({{ local.obstacle_count }})</label>
      <input type="range" min="0" max="24" step="1" v-model.number="local.obstacle_count" />
    </div>

    <div class="field">
      <label>场景边长 (m)</label>
      <input type="number" min="20" max="120" step="5" v-model.number="local.world_size" />
    </div>

    <div class="section-title">动力学</div>

    <div class="field">
      <label>追捕者 vmax (m/s)</label>
      <input type="number" min="0.5" max="20" step="0.5" v-model.number="local.pursuer_vmax" />
    </div>
    <div class="field">
      <label>逃逸者 vmax (m/s)</label>
      <input type="number" min="0.5" max="30" step="0.5" v-model.number="local.evader_vmax" />
    </div>
    <div class="field">
      <label>速度比 v_e / v_p = {{ speedRatio }}</label>
    </div>
    <div class="field">
      <label>追捕者 amax</label>
      <input type="number" min="0.5" max="10" step="0.1" v-model.number="local.pursuer_amax" />
    </div>
    <div class="field">
      <label>逃逸者 amax</label>
      <input type="number" min="0.5" max="10" step="0.1" v-model.number="local.evader_amax" />
    </div>

    <div class="field">
      <label>逃逸策略</label>
      <select v-model="local.evader_policy">
        <option v-for="p in options.evader_policies" :key="p.id" :value="p.id">{{ p.label }}</option>
      </select>
    </div>

    <div class="section-title" v-if="local.method === 'fcem'">FCEM 层消融</div>
    <div class="layer-grid" v-if="local.method === 'fcem'">
      <label class="layer-item" v-for="layer in options.layers" :key="layer.id">
        <input
          type="checkbox"
          :value="layer.id"
          v-model="local.remove_layers"
        />
        <span>
          <strong>{{ layer.id }}</strong> ({{ layer.experiment }})
          <small>{{ layer.description }}</small>
        </span>
      </label>
    </div>
    <p v-else style="font-size: 12px; color: #64748b">层消融仅对 FCEM 生效</p>

    <div class="field">
      <label>随机种子</label>
      <input type="number" v-model.number="local.seed" />
    </div>

    <div class="field">
      <label>播放速度 ×{{ playSpeed.toFixed(1) }}</label>
      <input type="range" min="0.25" max="8" step="0.25" v-model.number="playSpeed" />
    </div>

    <button
      class="primary"
      style="width: 100%; margin-top: 8px; padding: 10px; border: none; border-radius: 6px"
      @click="$emit('apply')"
    >
      应用并重置
    </button>
  </aside>
</template>

<script setup>
import { computed, reactive, watch } from 'vue'

const props = defineProps({
  options: { type: Object, required: true },
  modelValue: { type: Object, required: true },
  playSpeed: { type: Number, default: 1 },
})

const emit = defineEmits(['update:modelValue', 'update:playSpeed', 'apply'])

const local = reactive({ ...props.modelValue })
const playSpeed = computed({
  get: () => props.playSpeed,
  set: (v) => emit('update:playSpeed', v),
})

watch(
  () => props.modelValue,
  (v) => Object.assign(local, v),
  { deep: true },
)

watch(
  local,
  (v) => emit('update:modelValue', { ...v, remove_layers: [...(v.remove_layers || [])] }),
  { deep: true },
)

const speedRatio = computed(() => {
  const vp = local.pursuer_vmax || 1
  return (local.evader_vmax / vp).toFixed(2)
})
</script>
