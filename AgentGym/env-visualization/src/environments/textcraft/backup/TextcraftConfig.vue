<template>
  <div class="textcraft-config">
    <div class="config-section">
      <h4>⚙️ TextCraft Configuration</h4>
      <p class="config-description">Customize your crafting adventure!</p>
      
      <div class="input-group">
        <label for="goal">🎯 Goal (Optional)</label>
        <textarea
          id="goal"
          v-model="config.goal"
          placeholder="e.g., Craft a wooden sword and shield for combat"
          class="config-input"
          rows="3"
          @input="validateAndEmit"
        ></textarea>
        <div class="input-hint">
          💡 Describe what you want to achieve in this environment
        </div>
      </div>
      
      <div class="input-group">
        <label for="commands">📋 Custom Commands (Optional)</label>
        <textarea
          id="commands"
          v-model="config.commands"
          placeholder="e.g., Special crafting recipes or custom game rules"
          class="config-input"
          rows="4"
          @input="validateAndEmit"
        ></textarea>
        <div class="input-hint">
          🔧 Define custom commands or rules for this session
        </div>
      </div>
      
      <div class="quick-presets">
        <h5>🚀 Quick Presets</h5>
        <div class="preset-buttons">
          <button 
            v-for="preset in presets" 
            :key="preset.name"
            @click="applyPreset(preset)"
            class="preset-btn"
          >
            {{ preset.icon }} {{ preset.name }}
          </button>
        </div>
      </div>
      
      <div class="config-status">
        <div class="status-indicator" :class="{ 'ready': isConfigValid }">
          {{ isConfigValid ? '✅ Ready to create!' : '⏳ Configure your environment...' }}
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive, onMounted } from 'vue'

export default {
  name: 'TextCraftConfig',
  emits: ['config-ready'],
  setup(props, { emit }) {
    const config = reactive({
      goal: '',
      commands: ''
    })

    const isConfigValid = ref(true) // TextCraft can work without specific config

    const presets = ref([
      {
        name: 'Builder',
        icon: '🏗️',
        goal: 'Build a complete house with walls, roof, and furniture',
        commands: 'Focus on construction and architectural elements'
      },
      {
        name: 'Crafter',
        icon: '⚒️',
        goal: 'Master all crafting recipes and create advanced tools',
        commands: 'Emphasize crafting chains and tool progression'
      },
      {
        name: 'Explorer',
        icon: '🗺️',
        goal: 'Explore the world and gather diverse resources',
        commands: 'Enable exploration mechanics and resource discovery'
      },
      {
        name: 'Survivor',
        icon: '🛡️',
        goal: 'Survive harsh conditions and defend against threats',
        commands: 'Include survival elements and defensive crafting'
      }
    ])

    const validateAndEmit = () => {
      // TextCraft doesn't require specific configuration, always valid
      isConfigValid.value = true
      emitConfig()
    }

    const emitConfig = () => {
      emit('config-ready', {
        goal: config.goal.trim() || null,
        commands: config.commands.trim() || null
      })
    }

    const applyPreset = (preset) => {
      config.goal = preset.goal
      config.commands = preset.commands
      validateAndEmit()
      
      // Visual feedback
      const button = event.target
      button.style.transform = 'scale(0.95)'
      setTimeout(() => {
        button.style.transform = ''
      }, 150)
    }

    const getConfig = () => {
      return {
        goal: config.goal.trim() || null,
        commands: config.commands.trim() || null
      }
    }

    onMounted(() => {
      emitConfig()
    })

    // Expose getConfig method for parent component
    return {
      config,
      isConfigValid,
      presets,
      validateAndEmit,
      applyPreset,
      getConfig
    }
  }
}
</script>

<style scoped>
.textcraft-config {
  margin: 1.5rem 0;
}

.config-section {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border: 2px solid #dee2e6;
  border-radius: 16px;
  padding: 2rem;
  position: relative;
  overflow: hidden;
}

.config-section::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #4CAF50, #45a049);
}

.config-section h4 {
  color: #2c3e50;
  margin: 0 0 0.5rem 0;
  font-size: 1.3rem;
  font-weight: 700;
}

.config-description {
  color: #6c757d;
  margin-bottom: 2rem;
  font-style: italic;
}

.input-group {
  margin-bottom: 2rem;
}

.input-group label {
  display: block;
  margin-bottom: 0.75rem;
  font-weight: 600;
  color: #495057;
  font-size: 1rem;
}

.config-input {
  width: 100%;
  padding: 1rem;
  border: 2px solid #e9ecef;
  border-radius: 12px;
  font-size: 0.95rem;
  font-family: inherit;
  transition: all 0.3s ease;
  background: white;
  resize: vertical;
  min-height: 60px;
}

.config-input:focus {
  outline: none;
  border-color: #4CAF50;
  box-shadow: 0 0 0 4px rgba(76, 175, 80, 0.1);
  transform: translateY(-1px);
}

.config-input::placeholder {
  color: #adb5bd;
  font-style: italic;
}

.input-hint {
  margin-top: 0.5rem;
  font-size: 0.85rem;
  color: #6c757d;
  padding-left: 0.5rem;
  border-left: 3px solid #4CAF50;
}

.quick-presets {
  margin-bottom: 2rem;
  padding: 1.5rem;
  background: rgba(255, 255, 255, 0.7);
  border-radius: 12px;
  border: 1px solid #e9ecef;
}

.quick-presets h5 {
  margin: 0 0 1rem 0;
  color: #495057;
  font-size: 1.1rem;
  font-weight: 600;
}

.preset-buttons {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.75rem;
}

.preset-btn {
  background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
  border: 2px solid #dee2e6;
  border-radius: 10px;
  padding: 0.75rem 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  font-weight: 500;
  color: #495057;
}

.preset-btn:hover {
  background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
  color: white;
  border-color: #4CAF50;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
}

.config-status {
  text-align: center;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 10px;
  border: 1px solid #e9ecef;
}

.status-indicator {
  font-size: 1.1rem;
  font-weight: 600;
  color: #6c757d;
  transition: all 0.3s ease;
}

.status-indicator.ready {
  color: #28a745;
  animation: readyPulse 2s ease-in-out infinite;
}

@keyframes readyPulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

/* 响应式设计 */
@media (max-width: 768px) {
  .config-section {
    padding: 1.5rem;
  }
  
  .preset-buttons {
    grid-template-columns: 1fr;
  }
  
  .config-input {
    padding: 0.75rem;
  }
}
</style>