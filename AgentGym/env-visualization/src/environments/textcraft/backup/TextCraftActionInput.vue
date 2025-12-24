<template>
  <div class="textcraft-action-input">
    <div class="input-section">
      <div class="input-group">
        <div class="input-container">
          <input
            v-model="currentAction"
            @keyup.enter="sendAction"
            @input="onInputChange"
            @focus="showSuggestions = true"
            placeholder="Enter your crafting action... (e.g., inventory, get 1 wood, craft 1 stick using 1 wood)"
            :disabled="!environmentId || loading"
            class="action-input-field"
          />
          
          <!-- TextCraft特定建议下拉框 -->
          <div v-if="showSuggestions && filteredSuggestions.length > 0" class="suggestions-dropdown">
            <div class="suggestions-header">
              <span>💡 Suggested Actions</span>
            </div>
            <div 
              v-for="suggestion in filteredSuggestions"
              :key="suggestion.text"
              class="suggestion-item"
              @click="selectSuggestion(suggestion.text)"
            >
              <span class="suggestion-icon">{{ suggestion.icon }}</span>
              <div class="suggestion-content">
                <div class="suggestion-text">{{ suggestion.text }}</div>
                <div class="suggestion-description">{{ suggestion.description }}</div>
              </div>
            </div>
          </div>
        </div>
        
        <button 
          @click="sendAction"
          :disabled="!currentAction.trim() || !environmentId || loading"
          class="send-button"
        >
          {{ loading ? '⏳ Sending...' : '🚀 Send' }}
        </button>
      </div>
      
      <!-- TextCraft建议动作 -->
      <div v-if="suggestedAction" class="suggested-action">
        <span class="suggestion-label">🤖 AI Suggests:</span>
        <span class="suggestion-text">{{ suggestedAction }}</span>
        <button @click="useSuggestedAction" class="use-suggestion-btn">
          ✨ Use This
        </button>
      </div>

      <!-- 快速动作按钮 -->
      <div class="quick-actions">
        <button 
          v-for="quickAction in quickActions"
          :key="quickAction.text"
          @click="selectSuggestion(quickAction.text)"
          class="quick-action-btn"
          :title="quickAction.description"
        >
          {{ quickAction.icon }} {{ quickAction.text }}
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue'
import TextCraftClient from '../client/TextCraftClient.js'

export default {
  name: 'TextCraftActionInput',
  props: {
    environmentId: {
      type: [String, Number],
      default: null
    },
    suggestedAction: {
      type: String,
      default: ''
    },
    currentEnvironmentState: {
      type: Object,
      default: null
    },
    availableCommands: {
      type: Array,
      default: () => []
    }
  },
  emits: ['action-sent', 'response-received'],
  setup(props, { emit }) {
    const currentAction = ref('')
    const loading = ref(false)
    const showSuggestions = ref(false)
    const client = new TextCraftClient()
    
    // TextCraft特定的建议动作
    const textcraftSuggestions = ref([
      { 
        text: 'inventory', 
        icon: '🎒', 
        description: 'Check your current inventory',
        category: 'basic'
      },
      { 
        text: 'look around', 
        icon: '👀', 
        description: 'Examine your surroundings',
        category: 'basic'
      },
      { 
        text: 'get 1 wood', 
        icon: '🪵', 
        description: 'Collect wood from the environment',
        category: 'gather'
      },
      { 
        text: 'get 1 stone', 
        icon: '🪨', 
        description: 'Collect stone from the environment',
        category: 'gather'
      },
      { 
        text: 'craft 1 stick using 1 wood', 
        icon: '🪄', 
        description: 'Craft a basic stick',
        category: 'craft'
      },
      { 
        text: 'craft 1 wooden_pickaxe using 3 wood, 2 stick', 
        icon: '⛏️', 
        description: 'Craft a wooden pickaxe',
        category: 'craft'
      },
      { 
        text: 'craft 1 wooden_axe using 3 wood, 2 stick', 
        icon: '🪓', 
        description: 'Craft a wooden axe',
        category: 'craft'
      },
      { 
        text: 'craft 1 wooden_sword using 2 wood, 1 stick', 
        icon: '⚔️', 
        description: 'Craft a wooden sword',
        category: 'craft'
      }
    ])

    // 快速动作按钮
    const quickActions = computed(() => [
      { text: 'inventory', icon: '🎒', description: 'Check inventory' },
      { text: 'look around', icon: '👀', description: 'Look around' },
      { text: 'get 1 wood', icon: '🪵', description: 'Get wood' },
      { text: 'get 1 stone', icon: '🪨', description: 'Get stone' }
    ])

    // 从可用命令中提取建议
    const commandSuggestions = computed(() => {
      if (!props.availableCommands || props.availableCommands.length === 0) {
        return []
      }
      
      return props.availableCommands.map(command => ({
        text: command,
        icon: '⚒️',
        description: 'Available crafting command',
        category: 'available'
      }))
    })

    // 过滤后的建议
    const filteredSuggestions = computed(() => {
      const query = currentAction.value.toLowerCase().trim()
      const allSuggestions = [...textcraftSuggestions.value, ...commandSuggestions.value]
      
      if (!query) {
        // 按类别排序：basic -> gather -> craft -> available
        const categoryOrder = { basic: 0, gather: 1, craft: 2, available: 3 }
        return allSuggestions
          .sort((a, b) => (categoryOrder[a.category] || 4) - (categoryOrder[b.category] || 4))
          .slice(0, 10)
      }
      
      return allSuggestions
        .filter(s => s.text.toLowerCase().includes(query) || s.description.toLowerCase().includes(query))
        .slice(0, 8)
    })
    
    // 发送动作
    const sendAction = async () => {
      if (!currentAction.value.trim() || !props.environmentId) {
        return
      }
      
      const action = currentAction.value.trim()
      loading.value = true
      
      try {
        emit('action-sent', action)
        
        const result = await client.step(props.environmentId, action)
        
        if (result.success) {
          emit('response-received', result.data)
        } else {
          emit('response-received', { 
            error: result.error || 'Unknown error',
            observation: `Error: ${result.error || 'Action failed'}`
          })
        }
        
        currentAction.value = ''
        showSuggestions.value = false
        
      } catch (error) {
        console.error('Action execution failed:', error)
        emit('response-received', { 
          error: error.message,
          observation: `Error: ${error.message}`
        })
      } finally {
        loading.value = false
      }
    }
    
    // 处理输入变化
    const onInputChange = () => {
      showSuggestions.value = true
    }
    
    // 选择建议
    const selectSuggestion = (suggestion) => {
      currentAction.value = suggestion
      showSuggestions.value = false
    }
    
    // 使用AI建议的动作
    const useSuggestedAction = () => {
      currentAction.value = props.suggestedAction
      showSuggestions.value = false
    }
    
    // 点击外部时隐藏建议
    const handleClickOutside = (event) => {
      if (!event.target.closest('.textcraft-action-input')) {
        showSuggestions.value = false
      }
    }
    
    // 添加全局点击监听器
    document.addEventListener('click', handleClickOutside)
    
    // 清理监听器
    const cleanup = () => {
      document.removeEventListener('click', handleClickOutside)
    }
    
    return {
      currentAction,
      loading,
      showSuggestions,
      filteredSuggestions,
      quickActions,
      sendAction,
      onInputChange,
      selectSuggestion,
      useSuggestedAction,
      cleanup
    }
  },
  beforeUnmount() {
    this.cleanup()
  }
}
</script>

<style scoped>
.textcraft-action-input {
  background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
  border-top: 2px solid #e2e8f0;
  padding: 1.5rem;
}

.input-group {
  display: flex;
  gap: 1rem;
  align-items: flex-start;
  margin-bottom: 1rem;
}

.input-container {
  flex: 1;
  position: relative;
}

.action-input-field {
  width: 100%;
  padding: 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  font-size: 1rem;
  transition: all 0.3s ease;
  background: white;
  color: #2d3748;
  font-family: inherit;
}

.action-input-field:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
  transform: translateY(-1px);
}

.action-input-field:disabled {
  background: #f7fafc;
  color: #a0aec0;
  cursor: not-allowed;
}

.action-input-field::placeholder {
  color: #a0aec0;
  font-style: italic;
}

.suggestions-dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: white;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
  z-index: 1000;
  max-height: 320px;
  overflow-y: auto;
  backdrop-filter: blur(10px);
}

.suggestions-header {
  padding: 1rem;
  border-bottom: 1px solid #f1f5f9;
  font-weight: 600;
  color: #4a5568;
  background: #f8fafc;
  font-size: 0.9rem;
}

.suggestion-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  cursor: pointer;
  border-bottom: 1px solid #f1f5f9;
  transition: all 0.2s ease;
}

.suggestion-item:hover {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.suggestion-item:last-child {
  border-bottom: none;
}

.suggestion-icon {
  font-size: 1.2rem;
  min-width: 24px;
}

.suggestion-content {
  flex: 1;
}

.suggestion-text {
  font-weight: 500;
  margin-bottom: 0.25rem;
}

.suggestion-description {
  font-size: 0.8rem;
  opacity: 0.8;
}

.send-button {
  padding: 1rem 2rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  font-weight: 600;
  font-size: 1rem;
  transition: all 0.3s ease;
  white-space: nowrap;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.send-button:hover:not(:disabled) {
  background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.send-button:disabled {
  background: #a0aec0;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.suggested-action {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
  padding: 1rem;
  background: linear-gradient(135deg, #fef5e7 0%, #fed7aa 100%);
  border: 2px solid #fbbf24;
  border-radius: 12px;
  color: #92400e;
}

.suggestion-label {
  font-weight: 700;
  white-space: nowrap;
}

.suggestion-text {
  flex: 1;
  font-style: italic;
  font-weight: 500;
}

.use-suggestion-btn {
  padding: 0.5rem 1rem;
  background: #fbbf24;
  color: #92400e;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 600;
  transition: all 0.3s ease;
  white-space: nowrap;
}

.use-suggestion-btn:hover {
  background: #f59e0b;
  transform: translateY(-1px);
}

.quick-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.quick-action-btn {
  padding: 0.75rem 1rem;
  background: white;
  border: 2px solid #e2e8f0;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 0.9rem;
  font-weight: 500;
  color: #4a5568;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.quick-action-btn:hover {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-color: #667eea;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .textcraft-action-input {
    padding: 1rem;
  }
  
  .input-group {
    flex-direction: column;
  }
  
  .send-button {
    width: 100%;
  }
  
  .suggested-action {
    flex-direction: column;
    text-align: center;
    gap: 0.75rem;
  }
  
  .suggestion-label,
  .suggestion-text {
    white-space: normal;
  }
  
  .quick-actions {
    justify-content: center;
  }
  
  .quick-action-btn {
    flex: 1;
    min-width: 120px;
  }
}
</style>