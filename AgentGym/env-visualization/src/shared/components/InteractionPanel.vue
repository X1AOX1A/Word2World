<template>
  <div class="interaction-panel">
    <div class="panel-header">
      <h3>💬 Interaction History</h3>
      <div class="header-controls">
        <label class="verbose-toggle">
          <input 
            type="checkbox" 
            v-model="verboseLogging"
            @change="onVerboseLoggingChange"
          />
          <span class="checkmark"></span>
          Verbose
        </label>
        <button @click="clearHistory" class="clear-btn" :disabled="history.length === 0">
          Clear
        </button>
      </div>
    </div>
    
    <div class="history-container" ref="historyContainer">
      <div v-if="history.length === 0" class="empty-history">
        <p>No interactions yet</p>
        <small>Send an action to start interacting with the environment</small>
      </div>
      
      <div v-for="(entry, index) in history" :key="index" :class="['history-entry', { 'completion-entry': entry.isCompletion }]">
        <div class="entry-header">
          <span :class="['entry-type', entry.type, { 'completion-type': entry.isCompletion }]">
            {{ entry.type === 'action' ? '➤' : '⇐' }}
            {{ entry.type === 'action' ? 'Action' : entry.isCompletion ? 'Completion' : 'Response' }}
          </span>
          <span class="entry-time">{{ formatTime(entry.timestamp) }}</span>
        </div>
        <div class="entry-content">{{ entry.content }}</div>
      </div>
    </div>
    
    <div class="input-section">
      <ActionInput 
        :environment-id="environmentId"
        :disabled="!environmentId"
        :suggested-action="suggestedAction"
        :environment-state="effectiveEnvironmentState"
        @action-sent="onActionSent"
        @response-received="onResponseReceived"
      />
    </div>
  </div>
</template>

<script>
import { ref, nextTick, watch, computed } from 'vue'
import ActionInput from './ActionInput.vue'
import environmentManager from '../services/environmentManager.js'

export default {
  name: 'InteractionPanel',
  components: {
    ActionInput
  },
  props: {
    environmentId: {
      type: [Number, String],
      default: null
    },
    environmentType: {
      type: String,
      default: 'textcraft'
    },
    suggestedAction: {
      type: String,
      default: ''
    },
    currentEnvironmentState: {
      type: Object,
      default: null
    }
  },
  emits: ['action-sent', 'response-received'],
  setup(props, { emit }) {
    const history = ref([])
    const historyContainer = ref(null)
    const currentEnvironmentState = ref(null)
    const verboseLogging = ref(false) // 默认值改为false，避免api依赖

    // Get environment client (create new instance each time)
    const environmentClient = computed(() => {
      return environmentManager.getClient(props.environmentType)
    })
    // Watch for environment type changes
    watch(() => props.environmentType, (newType) => {
      if (newType) {
        console.log(`InteractionPanel: Environment type changed to ${newType}`)
        // Environment client会自动更新，不需要手动设置
      }
    }, { immediate: true })

    // Computed property to merge external and internal environment state
    const effectiveEnvironmentState = computed(() => {
      return props.currentEnvironmentState || currentEnvironmentState.value
    })

    const onActionSent = (action) => {
      history.value.push({
        type: 'action',
        content: action,
        timestamp: new Date()
      })
      scrollToBottom()
      
      // Emit to parent component
      emit('action-sent', action)
    }

    const onResponseReceived = (response) => {
      // Handle different response formats
      let content = ''
      let isCompletion = false
      
      if (typeof response === 'string') {
        content = response
      } else if (response && response.observation) {
        content = response.observation
        isCompletion = response.isCompletion || response.done === true
        // Store the full environment state for AI suggestions
        currentEnvironmentState.value = response
      } else {
        content = JSON.stringify(response, null, 2)
      }

      history.value.push({
        type: 'response',
        content: content,
        timestamp: new Date(),
        isCompletion: isCompletion
      })
      scrollToBottom()
      
      // Emit to parent component
      emit('response-received', response)
    }

    const clearHistory = () => {
      history.value = []
      currentEnvironmentState.value = null
    }

    const formatTime = (timestamp) => {
      return timestamp.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      })
    }

    const scrollToBottom = () => {
      nextTick(() => {
        if (historyContainer.value) {
          historyContainer.value.scrollTop = historyContainer.value.scrollHeight
        }
      })
    }

    // Watch for environment ID changes and clear history
    watch(() => props.environmentId, (newId, oldId) => {
      if (newId !== oldId) {
        clearHistory()
      }
    })

    // Expose methods for parent component
    const addInteraction = (type, content, isCompleted = false) => {
      console.log(`📝 Adding interaction: ${type}`, content, 'completed:', isCompleted)
      
      // 处理响应内容，确保只显示有用信息
      let displayContent = content
      
      if (type === 'response') {
        // 如果content是对象，提取observation
        if (typeof content === 'object' && content !== null) {
          if (content.result && content.result.data && content.result.data.observation) {
            displayContent = content.result.data.observation
          } else if (content.observation) {
            displayContent = content.observation
          } else if (content.result && typeof content.result === 'string') {
            displayContent = content.result
          } else {
            displayContent = JSON.stringify(content)
          }
        } else if (typeof content === 'string') {
          displayContent = content
        }
        
        // 如果是完成状态，添加特殊样式标记
        if (isCompleted) {
          displayContent = `🎉 ${displayContent}\n\n✅ Auto run completed successfully!`
        }
      }
      
      const interaction = {
        id: Date.now() + Math.random(),
        type,
        content: displayContent,
        timestamp: new Date(),
        isAuto: type === 'action' && (content.includes('[Auto]') || content.includes('[Auto-AI]') || content.includes('[Auto-Fallback]')),
        isCompleted: isCompleted
      }
      
      history.value.push(interaction)
      
      // 保持历史记录在合理范围内
      if (history.value.length > 131072) {
        history.value = history.value.slice(-50)
      }
      
      // 滚动到底部
      nextTick(() => {
        scrollToBottom()
      })
      
      console.log(`✅ Interaction added. Total: ${history.value.length}`)
    }

    const onVerboseLoggingChange = () => {
      // Verbose logging change - 可以在这里添加日志逻辑
      console.log('Verbose logging changed to:', verboseLogging.value)
    }

    return {
      history,
      historyContainer,
      currentEnvironmentState,
      effectiveEnvironmentState,
      verboseLogging,
      onActionSent,
      onResponseReceived,
      clearHistory,
      formatTime,
      addInteraction,
      onVerboseLoggingChange
    }
  }
}
</script>

<style scoped>
.interaction-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: white;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background: #f8f9fa;
  border-bottom: 1px solid #dee2e6;
}

.panel-header h3 {
  margin: 0;
  color: #2c3e50;
  font-size: 1.1rem;
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.verbose-toggle {
  display: flex;
  align-items: center;
  font-size: 0.8rem;
  color: #6c757d;
  cursor: pointer;
  user-select: none;
}

.verbose-toggle input[type="checkbox"] {
  margin-right: 0.5rem;
  cursor: pointer;
}

.clear-btn {
  background: #6c757d;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 0.25rem 0.5rem;
  cursor: pointer;
  font-size: 0.8rem;
  transition: background 0.3s;
}

.clear-btn:hover {
  background: #5a6268;
}

.clear-btn:disabled {
  background: #dee2e6;
  cursor: not-allowed;
}

.history-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
}

.empty-history {
  text-align: center;
  color: #6c757d;
  padding: 2rem;
}

.empty-history small {
  display: block;
  margin-top: 0.5rem;
  font-size: 0.9rem;
}

.history-entry {
  margin-bottom: 1rem;
  padding: 0.75rem;
  border-radius: 6px;
  background: #f8f9fa;
  border-left: 3px solid #dee2e6;
}

.history-entry:last-child {
  margin-bottom: 0;
}

.entry-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  font-size: 0.8rem;
}

.entry-type {
  font-weight: 600;
  padding: 0.2rem 0.5rem;
  border-radius: 12px;
  font-size: 0.75rem;
}

.entry-type.action {
  background: #e3f2fd;
  color: #1976d2;
}

.entry-type.response {
  background: #e8f5e8;
  color: #388e3c;
}

.entry-type.completion-type {
  background: #fff3e0;
  color: #f57c00;
  font-weight: bold;
}

.history-entry:has(.entry-type.action) {
  border-left-color: #2196f3;
}

.history-entry:has(.entry-type.response) {
  border-left-color: #4caf50;
}

.completion-entry {
  border-left-color: #ff9800 !important;
  background: #fffdf7;
  box-shadow: 0 2px 4px rgba(255, 152, 0, 0.1);
}

.entry-time {
  color: #6c757d;
  font-size: 0.7rem;
}

.entry-content {
  white-space: pre-wrap;
  word-break: break-word;
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
  line-height: 1.4;
  color: #2c3e50;
}

.input-section {
  border-top: 1px solid #dee2e6;
  padding: 1rem;
  background: white;
}

/* Scrollbar styling */
.history-container::-webkit-scrollbar {
  width: 6px;
}

.history-container::-webkit-scrollbar-track {
  background: #f1f1f1;
}

.history-container::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

.history-container::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}
</style>
