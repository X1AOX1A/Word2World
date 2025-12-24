<template>
  <div class="environment-view">
    <!-- Environment selection button -->
    <div class="env-selector-button">
      <router-link to="/" class="back-home">← Back to Home</router-link>
    </div>
    
    <div class="split-container">
      <!-- Environment Viewer Panel -->
      <div class="env-panel">
        <EnvViewer 
          :environment-type="envName"
          :environment-id="environmentId"
          :last-auto-action="lastAutoAction"
          :last-auto-response="lastAutoResponse"
          @environment-created="onEnvironmentCreated"
          @environment-reset="onEnvironmentReset"
          @state-updated="onStateUpdated"
          @suggest-action="onSuggestAction"
          @auto-action-sent="onAutoActionSent"
          @auto-response-received="onAutoResponseReceived"
          @action-completed="onStateUpdated"
          @observation-updated="onObservationUpdated"
          ref="envViewer"
        />
      </div>
      
      <!-- Agent Interaction Panel -->
      <div class="agent-panel">
        <ActionInput 
          :environment-id="environmentId"
          :environment-type="envName"
          :current-state="currentState"
          :suggested-action="suggestedAction"
          @action-sent="onActionSent"
          @response-received="onResponseReceived"
          ref="actionInput"
        />
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, computed } from 'vue'
import { useRoute } from 'vue-router'
import EnvViewer from '../shared/components/EnvViewer.vue'
import ActionInput from '../shared/components/ActionInput.vue'

export default {
  name: 'EnvironmentView',
  components: {
    EnvViewer,
    ActionInput
  },
  setup() {
    const route = useRoute()
    const envName = computed(() => route.params.envName)
    const environmentId = ref(null)
    const currentState = ref(null)
    const suggestedAction = ref('')
    const envViewer = ref(null)
    const actionInput = ref(null)
    
    // Track last state update to prevent duplicates
    const lastStateUpdate = ref(null)
    const stateUpdateTimeout = ref(null)
    const lastActionTime = ref(0) // Track when actions are sent
    
    // State for passing auto-action data to viewers
    const lastAutoAction = ref(null)
    const lastAutoResponse = ref(null)
    
    // Environment Creation Event Handler
    const onEnvironmentCreated = (id) => {
      console.log('Environment created:', id)
      environmentId.value = id
    }
    
    // Environment Reset Event Handler
    const onEnvironmentReset = (data) => {
      console.log('Environment reset', data)
      
      // 使用新的专用重置处理函数，而不是直接清除历史记录
      if (actionInput.value && actionInput.value.handleReset) {
        // 准备重置消息
        let resetMessage = 'Env Reseted'
        
        // 如果有新目标信息，添加到消息中
        if (data && data.reason === 'goal_changed' && data.newGoal) {
          resetMessage += `New Goal: ${data.newGoal}`
        } else if (data && data.result && data.result.data) {
          // 尝试从结果中提取目标
          const resultData = data.result.data;
          let goalText = '';
          
          // 处理不同的数据格式
          if (typeof resultData === 'string') {
            const goalMatch = resultData.match(/Your goal:\s*(.+?)(?:\n|$)/i) || 
                              resultData.match(/Goal:\s*(.+?)(?:\n|$)/i);
            if (goalMatch && goalMatch[1]) {
              goalText = goalMatch[1].trim();
            }
          } else if (resultData.observation) {
            // 如果观察结果是字符串，尝试从中提取目标
            const obsStr = typeof resultData.observation === 'string' 
              ? resultData.observation 
              : JSON.stringify(resultData.observation);
              
            const goalMatch = obsStr.match(/Your goal:\s*(.+?)(?:\n|$)/i) || 
                              obsStr.match(/Goal:\s*(.+?)(?:\n|$)/i);
            if (goalMatch && goalMatch[1]) {
              goalText = goalMatch[1].trim();
            }
          }
          
          if (goalText) {
            resetMessage += `New Goal: ${goalText}`
          }
        }
        // const parsed = JSON.parse(data.result.data)
        console.log("=======================")
        console.log(data.result.data)
        if (data.result.data.task_description) {
          resetMessage = data.result.data.task_description
        }
        else if (data.result.data.observation.raw) {
          resetMessage = data.result.data.observation.raw
        }
        else if (data.result.data.observation) {
          resetMessage = data.result.data.observation
        }
        else{
          resetMessage = data.result.data
        }
        console.log('🧹 Using handleReset with message:', resetMessage)
        actionInput.value.handleReset(resetMessage)
        
        // After reset, clear state update tracking to prevent duplicates
        lastStateUpdate.value = null
        
        // Ignore any state updates for a short period after reset
        if (stateUpdateTimeout.value) {
          clearTimeout(stateUpdateTimeout.value)
        }
        stateUpdateTimeout.value = setTimeout(() => {
          stateUpdateTimeout.value = null
        }, 500)
      }
    }
    
    // State Update Event Handler
    const onStateUpdated = (state) => {
      // Ignore updates during the timeout period after reset
      if (stateUpdateTimeout.value) {
        console.log('State update ignored - in reset timeout period')
        return
      }
      
      // Track the time of updates to help identify action responses
      const now = Date.now()
      
      // Check for duplicate state updates by comparing observation content
      const stateStr = typeof state?.observation === 'string' 
        ? state.observation 
        : JSON.stringify(state?.observation || '');
      
      const lastStateStr = typeof lastStateUpdate.value?.observation === 'string' 
        ? lastStateUpdate.value.observation 
        : JSON.stringify(lastStateUpdate.value?.observation || '');
      
      // Only treat as a duplicate if:
      // 1. There was a previous state
      // 2. The content is exactly the same
      // 3. The done status hasn't changed
      // 4. It's not immediately after an action (within 2 seconds)
      const timeSinceLastAction = now - lastActionTime.value
      const isPossibleActionResponse = timeSinceLastAction < 2000
      
      const isDuplicate = lastStateUpdate.value && 
                          stateStr === lastStateStr &&
                          lastStateUpdate.value.done === state.done &&
                          !isPossibleActionResponse;
      
      // If state is the same as last one AND not an action response, ignore the update
      if (isDuplicate) {
        console.log('Duplicate state update detected, ignoring')
        return
      }
      
      if (isPossibleActionResponse) {
        console.log('Processing likely action response')
      } else {
        console.log('State updated with new content')
      }
      
      lastStateUpdate.value = state
      currentState.value = state
    }
    
    // Suggested Action Event Handler
    const onSuggestAction = (action) => {
      console.log('Action suggested:', action)
      suggestedAction.value = action
    }
    
    // Action Sent Event Handler
    const onActionSent = async (action) => {
      console.log('Action sent:', action)
      
      // Record action time for deduplication logic
      lastActionTime.value = Date.now()
      
      // If envViewer is available, use its handleUserAction function
      if (envViewer.value && typeof envViewer.value.handleUserAction === 'function') {
        try {
          // Process the action using the same mechanism as auto actions
          await envViewer.value.handleUserAction(action)
        } catch (error) {
          console.error('Error handling user action:', error)
        }
      } else {
        // Fall back to textcraft-specific behavior
        if (envName.value === 'textcraft' && envViewer.value) {
          envViewer.value.addAgentAction && envViewer.value.addAgentAction(action)
        }
      }
    }
    
    // Response Received Event Handler
    const onResponseReceived = (response) => {
      console.log('Response received:', response)
    }
    
    // Auto Action Sent Event Handler
    const onAutoActionSent = (data) => {
      console.log('Auto action sent:', data)
      
      // Store the action data for the viewer
      lastAutoAction.value = {
        ...data,
        timestamp: Date.now()
      }
      
      // Add agent action to ActionInput history
      if (actionInput.value) {
        if (typeof data === 'object' && data.action) {
          actionInput.value.addToHistory('action', `[Auto] ${data.action}`)
        } else if (typeof data === 'string') {
          actionInput.value.addToHistory('action', `[Auto] ${data}`)
        }
      }
    }
    
    // Auto Response Received Event Handler
    const onAutoResponseReceived = (data) => {
      console.log('Auto response received:', data);
      console.log('📥 EnvironmentView: Processing auto response with content:', 
        typeof data.observation === 'string' ? data.observation?.substring(0, 100) : 'non-string data');
      
      // Store the response data for the viewer
      lastAutoResponse.value = {
        ...data,
        timestamp: Date.now()
      }
      
      // Add auto response to ActionInput history
      if (actionInput.value) {
        // Record this as a recent action response for deduplication logic
        lastActionTime.value = Date.now()
        
        // Process the response based on its type
        if (typeof data === 'string') {
          console.log('📥 Adding string response to history');
          actionInput.value.addToHistory('response', data)
        } else if (typeof data === 'object' && data !== null) {
          if (data.observation) {
            // If there's an observation field, use that
            console.log('📥 Adding observation field to history');
            
            // 确保总是添加到历史记录中，不管是否和上一条消息相同
            const forceAdd = data.timestamp || data.forceDisplay ? true : false;
            actionInput.value.addToHistory('response', data.observation, forceAdd);
            
            // Also update the current state for ActionInput
            currentState.value = data;
          } else {
            // Otherwise, format the whole object
            try {
              console.log('📥 Adding formatted JSON object to history');
              const formatted = JSON.stringify(data, null, 2)
              actionInput.value.addToHistory('response', formatted)
            } catch (e) {
              console.log('📥 Error formatting response:', e);
              actionInput.value.addToHistory('response', 'Received response (cannot display)')
            }
          }
        }
        console.log('📥 Added response to ActionInput history');
      } else {
        console.warn('⚠️ ActionInput reference not available, cannot add response to history');
      }
    }
    
    // Observation Updated Event Handler - for WebArena
    const onObservationUpdated = (data) => {
      console.log('📄 EnvironmentView: Observation updated:', data);
      console.log('📄 Observation type:', typeof data.observation);
      console.log('📄 Observation content preview:', data.observation?.substring?.(0, 100) || data.observation);
      
      // Add observation to ActionInput history as response
      if (actionInput.value && data.observation) {
        // 确保传递的是字符串
        const observationText = typeof data.observation === 'string' 
          ? data.observation 
          : JSON.stringify(data.observation, null, 2);
          
        console.log('📄 Adding to ActionInput history:', observationText.substring(0, 50) + '...');
        actionInput.value.addToHistory('response', observationText, true);
        
        // Update current state
        currentState.value = {
          observation: observationText,
          timestamp: data.timestamp,
          environmentId: data.environmentId
        };
        
        console.log('📄 Successfully added observation to ActionInput history');
      } else {
        console.warn('⚠️ ActionInput not available or no observation data:', {
          hasActionInput: !!actionInput.value,
          hasObservation: !!data.observation
        });
      }
    }
    
    onMounted(() => {
      console.log('EnvironmentView mounted, environment:', envName.value)
      
      // 设置全局环境类型，供AI Agent使用
      window.currentEnvironmentType = envName.value;
      console.log('设置全局环境类型:', window.currentEnvironmentType);
    })
    
    return {
      envName,
      environmentId,
      currentState,
      suggestedAction,
      lastAutoAction,
      lastAutoResponse,
      onEnvironmentCreated,
      onEnvironmentReset,
      onStateUpdated,
      onSuggestAction,
      onActionSent,
      onResponseReceived,
      onAutoActionSent,
      onAutoResponseReceived,
      onObservationUpdated,
      envViewer,
      actionInput
    }
  }
}
</script>

<style scoped>
.environment-view {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

.split-container {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.env-panel {
  flex: 7;
  overflow-y: auto;
  border-right: 1px solid rgba(0, 0, 0, 0.1);
}

.agent-panel {
  flex: 3;
  overflow-y: auto;
  background: rgba(255, 255, 255, 0.8);
}

.env-selector-button {
  position: fixed;
  top: 10px;
  left: 10px;
  z-index: 100;
}

.back-home {
  display: inline-block;
  background: rgba(255, 255, 255, 0.9);
  color: #764ba2;
  padding: 0.5rem 1rem;
  border-radius: 30px;
  font-weight: 600;
  text-decoration: none;
  transition: all 0.3s;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  font-size: 0.9rem;
}

.back-home:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

@media (max-width: 1024px) {
  .split-container {
    flex-direction: column;
  }
  
  .env-panel, .agent-panel {
    flex: none;
    height: 50vh;
    border-right: none;
  }
  
  .env-panel {
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  }
}
</style> 