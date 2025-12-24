<template>
  <div class="action-input">
    <div class="action-input-header">
      <h3>Agent Interaction</h3>
      <div class="action-controls">
        <button 
          @click="clearHistory" 
          class="clear-btn"
          title="Clear interaction history"
        >
          <span class="btn-icon">🧹</span>
        </button>
        <button 
          @click="toggleSettingsPanel" 
          class="settings-btn"
          title="Agent settings"
        >
          <span class="btn-icon">⚙️</span>
        </button>
      </div>
    </div>

    <!-- Interaction History -->
    <div class="interaction-history" ref="historyContainer">
      <div 
        v-for="(item, index) in interactionHistory" 
        :key="index" 
        :class="['history-item', item.type]"
      >
        <div class="history-item-header">
          <span class="history-icon">{{ item.type === 'action' ? '🤖' : '🌍' }}</span>
          <span class="history-label">{{ item.type === 'action' ? 'Agent' : 'Environment' }}</span>
          <span class="history-time">{{ formatTime(item.timestamp) }}</span>
        </div>
        <div class="history-content" v-html="formatContent(item.content)"></div>
      </div>

      <div v-if="interactionHistory.length === 0" class="empty-history">
        <div class="empty-icon">💬</div>
        <p>No interactions yet. Type an action below to get started.</p>
      </div>
    </div>

    <!-- Action Form -->
    <div class="action-form">
      <textarea 
        v-model="actionInput" 
        class="action-textarea" 
        placeholder="Enter an action or command..."
        ref="actionTextarea"
        @keydown.enter.ctrl="submitAction"
        :disabled="loading"
      ></textarea>

      <div class="action-form-controls">
        <button 
          v-if="suggestedAction"
          @click="useAutoAction"
          class="auto-action-btn"
          :disabled="loading"
        >
          <span class="btn-icon">💡</span> Use Suggestion
        </button>
        <button 
          @click="submitAction" 
          class="submit-btn"
          :disabled="!actionInput.trim() || loading"
        >
          {{ loading ? 'Sending...' : 'Send' }}
        </button>
      </div>
    </div>

    <!-- Settings Panel -->
    <div v-if="showSettingsPanel" class="settings-panel">
      <div class="settings-header">
        <h4>Agent Settings</h4>
        <button @click="toggleSettingsPanel" class="close-settings">✕</button>
      </div>
      <div class="settings-content">
        <div class="setting-item">
          <label class="setting-label">
            Model
          </label>
          <select v-model="agentModel" class="setting-control">
            <option value="gpt-4">GPT-4</option>
            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
            <option value="claude-3">Claude 3</option>
          </select>
        </div>
        <div class="setting-item">
          <label class="setting-label">
            Temperature
          </label>
          <input 
            type="range" 
            v-model="agentTemperature" 
            min="0" 
            max="1" 
            step="0.1" 
            class="setting-slider"
          />
          <span class="setting-value">{{ agentTemperature }}</span>
        </div>
        <div class="setting-item">
          <label class="setting-label">
            <input type="checkbox" v-model="autoScroll" class="setting-checkbox" />
            Auto-scroll to bottom
          </label>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, watch, nextTick, onMounted, computed } from 'vue'
import { useRoute } from 'vue-router'

export default {
  name: 'ActionInput',
  props: {
    environmentId: {
      type: [String, Number],
      default: null
    },
    environmentType: {
      type: String,
      default: ''
    },
    suggestedAction: {
      type: String,
      default: ''
    },
    currentState: {
      type: Object,
      default: () => null
    }
  },
  emits: ['action-sent', 'response-received'],
  setup(props, { emit, expose }) {
    // Input and state
    const actionInput = ref('')
    const loading = ref(false)
    const interactionHistory = ref([])
    const historyContainer = ref(null)
    const actionTextarea = ref(null)
    
    // Settings
    const showSettingsPanel = ref(false)
    const agentModel = ref('gpt-4')
    const agentTemperature = ref(0.7)
    const autoScroll = ref(true)
    
    // Deduplication tracking
    const lastStateContent = ref('') // Track the last observation content
    const lastResetTimestamp = ref(0)
    const lastActionTimestamp = ref(0) // Track when actions are sent
    const isProcessingReset = ref(false)
    const ignoreStateUpdatesUntil = ref(0) // Timestamp to ignore updates until
    const ignoreAll = ref(false) // Flag to temporarily ignore all state updates
    
    // Watch for environment ID changes
    watch(() => props.environmentId, (newId, oldId) => {
      if (newId) {
        if (newId !== oldId) {
          // Clear history when environment changes
          interactionHistory.value = []
          lastStateContent.value = ''
        }
      }
    })
    
    // Add an item to interaction history
    const addToHistory = (type, content, forceAdd = false) => {
      console.log(`🎯 ActionInput addToHistory called:`, {
        type,
        contentType: typeof content,
        contentPreview: content?.substring?.(0, 50) || content,
        forceAdd
      });
      // 确保 content 是字符串
      let processedContent = content;
      if (typeof content !== 'string') {
        console.log('⚠️ ActionInput: Converting non-string content to string');
        try {
          processedContent = JSON.stringify(content, null, 2);
        } catch (e) {
          processedContent = String(content);
        }
        console.log('✅ ActionInput: Content converted:', processedContent.substring(0, 50) + '...');
      }
      if (processedContent.length > 600 && type === 'response') {
            console.warn('⚠️ ActionInput: observation too long, truncating');
            processedContent = processedContent.substring(0, 600) + '...';
      }
          
      // 检查是否与最近的消息重复，如果重复且不强制添加，则跳过
      if (!forceAdd && interactionHistory.value.length > 0) {
        const lastItem = interactionHistory.value[interactionHistory.value.length - 1];
        if (lastItem.type === type && lastItem.content === processedContent) {
          console.log('🔄 Skipping duplicate history item:', processedContent?.substring(0, 30));
          return;
        }
      }
      
      // 记录消息到历史记录
      const timestamp = new Date().toISOString();
      console.log(`📝 Adding to history: type=${type}, processedContent=${processedContent?.substring(0, 30)}..., forceAdd=${forceAdd}`);
      
      interactionHistory.value.push({
        type,
        content: processedContent,
        timestamp
      });
      
      console.log('✅ ActionInput: Added to history successfully, total items:', interactionHistory.value.length);
      
      // Scroll to bottom on next tick
      if (autoScroll.value) {
        nextTick(() => {
          if (historyContainer.value) {
            historyContainer.value.scrollTop = historyContainer.value.scrollHeight;
          }
        });
      }
    }
    
    // Clear interaction history
    const clearHistory = () => {
      interactionHistory.value = []
      lastStateContent.value = '' // Reset the content reference
    }
    
    // Handle reset event specifically
    const handleReset = (resetMessage = null) => {
      const now = Date.now()
      
      // Prevent duplicate reset messages (throttle to once every 500ms)
      if (now - lastResetTimestamp.value < 500) {
        console.log('🛑 ActionInput: Ignoring duplicate reset event')
        return
      }
      
      // Set processing flag and ignore period
      isProcessingReset.value = true
      lastResetTimestamp.value = now
      ignoreStateUpdatesUntil.value = now + 1000 // Ignore updates for 1 second after reset
      
      try {
        // Clear history first
        clearHistory()
        
        // Add reset message if provided
        if (resetMessage) {
          addToHistory('response', resetMessage)
        } else {
          addToHistory('response', 'Env reset.')
        }
      } finally {
        // Reset processing flag after a delay
        setTimeout(() => {
          isProcessingReset.value = false
        }, 500)
      }
    }
    
    // Toggle settings panel
    const toggleSettingsPanel = () => {
      showSettingsPanel.value = !showSettingsPanel.value
    }
    
    // Submit an action
    const submitAction = async () => {
      if (!actionInput.value.trim() || loading.value) return
      if (!props.environmentId) {
        alert('No environment active. Please create or select an environment first.')
        return
      }
      
      const action = actionInput.value.trim()
      loading.value = true
      
      try {
        // Record timestamp of action submission - important for deduplication
        lastActionTimestamp.value = Date.now()
        console.log('🔵 Action sent at:', new Date(lastActionTimestamp.value).toLocaleTimeString(), 
                   'Environment type:', props.environmentType)
        
        // Add action to history
        addToHistory('action', action)
        
        // Emit action event
        emit('action-sent', action)
        
        // Clear input
        actionInput.value = ''
        
        // Focus textarea
        nextTick(() => {
          actionTextarea.value?.focus()
        })
        
        // For SciWorld, set a brief ignore period to avoid duplicates
        if (props.environmentType === 'sciworld') {
          ignoreStateUpdatesUntil.value = Date.now() + 200
        }
      } catch (error) {
        console.error('Error sending action:', error)
        addToHistory('response', `Error: ${error.message}`)
      } finally {
        loading.value = false
      }
    }
    
    // Use suggested action
    const useAutoAction = () => {
      if (!props.suggestedAction) return
      
      actionInput.value = props.suggestedAction
      nextTick(() => {
        actionTextarea.value?.focus()
      })
    }
    
    // Set focus on mount
    onMounted(() => {
      nextTick(() => {
        actionTextarea.value?.focus()
      })
    })
    
    // Expose methods for external components
    expose({
      addToHistory,
      clearHistory,
      handleReset  // Expose new reset handler
    })
    
    // Special handling for SciWorld environment
    const handleSciworldStateUpdate = (newState, observationContent, now) => {
      // For SciWorld we want to show ALL responses from the environment
      
      // If this is a direct action response coming from a step
      const actionJustSent = (now - lastActionTimestamp.value < 3000);
      
      // 检查是否来自isActionResponse标记（Auto Run模式）
      const isActionResponse = newState.isActionResponse === true;
      
      // 对于动作响应或强制显示的消息，总是显示
      if (actionJustSent || isActionResponse || newState.forceDisplay) {
        console.log('✅ ActionInput: Showing SciWorld action response');
        lastStateContent.value = observationContent;
        
        // 强制添加到历史记录，不管是否重复
        addToHistory('response', newState.observation, true);
        return true;
      }
      
      // Still check for exact duplicates - if we've seen this exact message before, skip it
      if (observationContent === lastStateContent.value) {
        console.log('🛑 ActionInput: Ignoring identical SciWorld content');
        return false;
      }

      // For all other cases, show the response
      lastStateContent.value = observationContent;
      addToHistory('response', newState.observation);
      
      return true; // Message was handled
    }
    
    // Watch for environment state changes with improved deduplication
    watch(
      () => props.currentState,
      (newState, oldState) => {
        if (!newState) return;
        
        // Special handling for SciWorld Auto Run responses
        if (newState.isActionResponse) {
          console.log('👁️ ActionInput: Processing SciWorld Action Response');
          const observationContent = typeof newState.observation === 'string' 
            ? newState.observation.trim() 
            : JSON.stringify(newState.observation);
          
          // 处理SciWorld的动作响应
          handleSciworldStateUpdate(newState, observationContent, Date.now());
          return;
        }
        
        if (!newState.observation) return;
        
        // Skip processing if we're in reset mode
        if (isProcessingReset.value) {
          console.log('🛑 ActionInput: Skipping state update during reset')
          return
        }
        
        // Skip if we're in the ignore period
        const now = Date.now()
        if (now < ignoreStateUpdatesUntil.value) {
          console.log('🛑 ActionInput: Ignoring state update during cooldown period')
          return
        }
        
        // Normalize observation content for comparison
        let observationContent = ''
        if (typeof newState.observation === 'string') {
          observationContent = newState.observation.trim()
        } else if (newState.observation) {
          observationContent = JSON.stringify(newState.observation)
        }
        
        // Special handling for SciWorld environment
        if (props.environmentType === 'sciworld') {
          // Use our enhanced SciWorld handler
          handleSciworldStateUpdate(newState, observationContent, now)
          return
        }
        
        // For other environments, use standard handling
        const actionJustSent = (now - lastActionTimestamp.value < 2000)
        if (actionJustSent) {
          console.log('👍 ActionInput: Processing response to recent action')
          lastStateContent.value = observationContent
          
          // 确保传递给addToHistory的是字符串
          const observationToAdd = typeof newState.observation === 'string' 
            ? newState.observation 
            : JSON.stringify(newState.observation, null, 2);
          console.log('📝 ActionInput: Adding observation to history (action response):', typeof observationToAdd);
          addToHistory('response', observationToAdd)
          return
        }
        
        // For non-action updates, we still need deduplication logic
        const sameContent = observationContent === lastStateContent.value
        
        if (ignoreAll.value) {
          console.log('🛑 ActionInput: Ignoring state update during processing')
          return
        }
        
        if (sameContent && (oldState === null || oldState.done === newState.done)) {
          console.log('🛑 ActionInput: Ignoring duplicate state content')
          return
        }
        
        // Set the temporary ignore flag
        ignoreAll.value = true
        
        try {
          // Update the last content reference
          lastStateContent.value = observationContent
          
          // Add environment response to history - 确保是字符串
          let observationToAdd = typeof newState.observation === 'string' 
            ? newState.observation 
            : JSON.stringify(newState.observation, null, 2);
          console.log('📝 ActionInput: Adding observation to history (state update):', typeof observationToAdd);
          // if (typeof observationToAdd !== 'string') {
          //   try {
          //     observationToAdd = JSON.stringify(observationToAdd, null, 2);
          //   } catch (e) {
          //     observationToAdd = String(observationToAdd);
          //   }
          // }
          
          addToHistory('response', observationToAdd)
        } finally {
          // Reset ignore flag after a short delay
          setTimeout(() => {
            ignoreAll.value = false
          }, 100)
        }
      },
      { deep: true }
    )
    
    // Simple hash function for message deduplication
    const hashMessage = (message) => {
      if (!message) return 0
      let hash = 0
      for (let i = 0; i < message.length; i++) {
        const char = message.charCodeAt(i)
        hash = ((hash << 5) - hash) + char
        hash = hash & hash // Convert to 32bit integer
      }
      return hash
    }
    
    // Format timestamps
    const formatTime = (timestamp) => {
      if (!timestamp) return ''
      
      const date = new Date(timestamp)
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
    
    // Format content with line breaks
    const formatContent = (content) => {
      console.log(`🎨 ActionInput formatContent called:`, {
        contentType: typeof content,
        contentPreview: content?.substring?.(0, 50) || content
      });
      
      if (!content) return ''
      
      // Ensure content is a string
      let contentStr = ''
      
      if (typeof content === 'string') {
        contentStr = content
        console.log('✅ formatContent: Using string content directly');
      } else if (typeof content === 'object' && content !== null) {
        console.log('⚠️ formatContent: Received object content, processing...');
        // Special handling for object responses
        if (content.observation) {
          // If the object has an observation property, use that
          contentStr = typeof content.observation === 'string' 
            ? content.observation 
            : JSON.stringify(content.observation, null, 2)
          console.log('✅ formatContent: Extracted observation field');
        } else {
          // Otherwise, format the entire object
          try {
            contentStr = JSON.stringify(content, null, 2)
            console.log('✅ formatContent: Converted object to JSON');
          } catch (e) {
            contentStr = String(content)
            console.log('⚠️ formatContent: Fallback to String conversion');
          }
        }
      } else {
        contentStr = String(content)
        console.log('⚠️ formatContent: Using String conversion');
      }
      
      console.log(`🎨 formatContent result preview:`, contentStr.substring(0, 50) + '...');
      
      return contentStr
        .replace(/\n/g, '<br>')
        .replace(/```([^`]+)```/g, '<pre class="code-block">$1</pre>')
    }
    
    return {
      actionInput,
      loading,
      interactionHistory,
      historyContainer,
      actionTextarea,
      showSettingsPanel,
      agentModel,
      agentTemperature,
      autoScroll,
      formatTime,
      formatContent,
      clearHistory,
      toggleSettingsPanel,
      submitAction,
      useAutoAction,
      addToHistory,
      handleReset
    }
  }
}
</script>

<style scoped>
.action-input {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

.action-input-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.8);
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

.action-input-header h3 {
  margin: 0;
  font-size: 1.1rem;
  color: #333;
}

.action-controls {
  display: flex;
  gap: 0.5rem;
}

.clear-btn, .settings-btn {
  background: none;
  border: none;
  color: #666;
  cursor: pointer;
  padding: 0.25rem;
  border-radius: 4px;
  transition: all 0.2s;
}

.clear-btn:hover, .settings-btn:hover {
  background: rgba(0, 0, 0, 0.05);
  color: #333;
}

.btn-icon {
  font-size: 1.1rem;
}

.interaction-history {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.history-item {
  padding: 0.75rem;
  border-radius: 8px;
  background: white;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
  max-width: 100%;
  overflow-wrap: break-word;
}

.history-item.action {
  border-left: 3px solid #4CAF50;
}

.history-item.response {
  border-left: 3px solid #2196F3;
}

.history-item-header {
  display: flex;
  align-items: center;
  margin-bottom: 0.5rem;
  font-size: 0.85rem;
}

.history-icon {
  margin-right: 0.5rem;
}

.history-label {
  font-weight: 600;
  color: #333;
}

.history-time {
  margin-left: auto;
  color: #999;
  font-size: 0.75rem;
}

.history-content {
  white-space: pre-wrap;
  line-height: 1.5;
  font-size: 0.9rem;
}

.empty-history {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #999;
  text-align: center;
  padding: 2rem;
}

.empty-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  opacity: 0.5;
}

.action-form {
  padding: 1rem;
  background: white;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
}

.action-textarea {
  width: 100%;
  min-height: 80px;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 8px;
  resize: vertical;
  font-family: inherit;
  font-size: 0.9rem;
  margin-bottom: 0.75rem;
}

.action-textarea:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
}

.action-form-controls {
  display: flex;
  justify-content: flex-end;
  gap: 0.75rem;
}

.auto-action-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: #e3f2fd;
  color: #1976d2;
  border: none;
  padding: 0.6rem 1rem;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.auto-action-btn:hover:not(:disabled) {
  background: #bbdefb;
}

.submit-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 0.6rem 1.5rem;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.submit-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.submit-btn:disabled, .auto-action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.settings-panel {
  position: absolute;
  top: 4rem;
  right: 1rem;
  background: white;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  width: 300px;
  z-index: 100;
  animation: fadeIn 0.2s ease-out;
}

.settings-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid #eee;
}

.settings-header h4 {
  margin: 0;
  color: #333;
}

.close-settings {
  background: none;
  border: none;
  font-size: 1.1rem;
  color: #999;
  cursor: pointer;
  padding: 0;
}

.settings-content {
  padding: 1rem;
}

.setting-item {
  margin-bottom: 1rem;
}

.setting-label {
  display: block;
  margin-bottom: 0.5rem;
  color: #666;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.setting-control, .setting-slider {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.setting-value {
  display: inline-block;
  margin-left: 0.5rem;
  color: #666;
}

.setting-checkbox {
  margin-right: 0.5rem;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

:deep(.code-block) {
  background: #f1f3f5;
  padding: 0.75rem;
  border-radius: 4px;
  font-family: monospace;
  margin: 0.5rem 0;
  overflow-x: auto;
  white-space: pre;
}

@media (max-width: 768px) {
  .action-form-controls {
    flex-direction: column;
  }
  
  .auto-action-btn, .submit-btn {
    width: 100%;
  }
  
  .settings-panel {
    width: calc(100% - 2rem);
    right: 1rem;
  }
}
</style>
