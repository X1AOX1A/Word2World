<template>
  <div class="textcraft-page">
    <!-- 如果用户喜欢当前的卡通风格，可以直接使用完整系统 -->
    <TextCraftCompleteSystem v-if="useCompleteSystem" />
    
    <!-- 或者使用简化版本，更接近原始设计 -->
    <div v-else class="textcraft-simplified">
      <!-- 头部导航 -->
      <div class="page-header">
        <div class="header-content">
          <div class="header-left">
            <h1 class="page-title">⚒️ TextCraft Environment</h1>
            <p class="page-subtitle">AI-powered crafting and building adventure</p>
          </div>
          <div class="header-right">
            <div v-if="environmentId" class="env-status">
              <span class="status-dot connected"></span>
              <span>Environment {{ environmentId }} Active</span>
            </div>
            <button 
              @click="useCompleteSystem = true"
              class="btn btn-upgrade"
              title="Switch to enhanced interface"
            >
              ✨ Enhanced View
            </button>
          </div>
        </div>
      </div>

      <!-- 主要内容区域 -->
      <div class="main-content">
        <!-- 环境创建界面 -->
        <div v-if="!environmentId" class="environment-creation">
          <div class="creation-container">
            <div class="creation-header">
              <h2>🎯 Create Your TextCraft World</h2>
              <p>Configure your crafting adventure and let the AI guide you!</p>
            </div>
            
            <div class="creation-form">
              <!-- 目标设置 -->
              <div class="form-section">
                <label class="form-label">
                  <span class="label-icon">🎯</span>
                  <span>Crafting Goal (Optional)</span>
                </label>
                <textarea
                  v-model="config.goal"
                  placeholder="What would you like to achieve? e.g., 'Craft a wooden pickaxe and mine some stone'"
                  rows="3"
                  class="form-textarea"
                ></textarea>
              </div>

              <!-- 自定义命令 -->
              <div class="form-section">
                <label class="form-label">
                  <span class="label-icon">📋</span>
                  <span>Custom Commands (Optional)</span>
                </label>
                <textarea
                  v-model="config.commands"
                  placeholder="Any special crafting rules or available materials..."
                  rows="2"
                  class="form-textarea"
                ></textarea>
              </div>

              <!-- 快速预设 -->
              <div class="form-section">
                <label class="form-label">
                  <span class="label-icon">⚡</span>
                  <span>Quick Start Presets</span>
                </label>
                <div class="preset-grid">
                  <button 
                    v-for="preset in quickPresets"
                    :key="preset.name"
                    @click="applyPreset(preset)"
                    class="preset-card"
                  >
                    <div class="preset-icon">{{ preset.icon }}</div>
                    <div class="preset-info">
                      <div class="preset-name">{{ preset.name }}</div>
                      <div class="preset-desc">{{ preset.description }}</div>
                    </div>
                  </button>
                </div>
              </div>

              <!-- 创建按钮 -->
              <div class="form-actions">
                <button 
                  @click="createEnvironment"
                  :disabled="creating"
                  class="btn btn-create-large"
                >
                  {{ creating ? '🔄 Creating World...' : '🚀 Start Crafting Adventure' }}
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- 游戏界面 -->
        <div v-else class="game-area">
          <SplitPane :default-split="70">
            <template #left>
              <!-- 左侧：环境和库存 -->
              <div class="left-panel">
                <!-- 当前状态卡片 -->
                <div class="status-card">
                  <div class="status-header">
                    <h3>🎮 Current Status</h3>
                    <div class="status-controls">
                      <button @click="refreshState" :disabled="loading" class="control-btn">
                        {{ loading ? '⏳' : '🔄' }}
                      </button>
                      <button @click="resetEnvironment" class="control-btn">
                        🔄
                      </button>
                      <button 
                        @click="toggleAutoRun"
                        :class="['control-btn', 'auto-btn', isAutoRunning ? 'stop' : 'start']"
                      >
                        {{ isAutoRunning ? '⏹️' : '▶️' }}
                      </button>
                    </div>
                  </div>
                  
                  <div class="status-info">
                    <div class="info-row">
                      <span class="info-label">🎯 Goal:</span>
                      <span class="info-value">{{ currentGoal || 'Explore and craft freely' }}</span>
                    </div>
                    <div class="info-row">
                      <span class="info-label">🤖 AI Mode:</span>
                      <span class="info-value" :class="{ 'active': isAutoRunning }">
                        {{ isAutoRunning ? 'Auto-Playing' : 'Manual Control' }}
                      </span>
                    </div>
                  </div>
                </div>

                <!-- 库存展示 -->
                <div class="inventory-section">
                  <TextCraftInventoryViewer
                    :environment-id="environmentId"
                    :current-state="currentState"
                    @suggest-action="onSuggestAction"
                    @item-selected="onItemSelected"
                  />
                </div>

                <!-- 观察窗口 -->
                <div class="observation-section">
                  <div class="section-header">
                    <h4>👁️ Current Observation</h4>
                    <button @click="copyObservation" class="copy-btn" title="Copy observation">
                      📋
                    </button>
                  </div>
                  <div class="observation-box">
                    <pre v-if="currentObservation" class="observation-text">{{ currentObservation }}</pre>
                    <div v-else class="no-observation">
                      <span>🤔 No observation available</span>
                    </div>
                  </div>
                </div>
              </div>
            </template>
            
            <template #right>
              <!-- 右侧：交互和历史 -->
              <div class="right-panel">
                <!-- 交互历史 -->
                <div class="history-section">
                  <div class="section-header">
                    <h4>💬 Interaction History</h4>
                    <button @click="clearHistory" class="clear-btn">🗑️</button>
                  </div>
                  
                  <div class="history-container">
                    <div v-if="interactionHistory.length === 0" class="no-history">
                      <div class="no-history-icon">☀️</div>
                      <div class="no-history-text">
                        <h5>Ready to Start!</h5>
                        <p>Send an action below to begin your crafting journey</p>
                      </div>
                    </div>
                    
                    <div v-else class="history-list">
                      <div 
                        v-for="(interaction, index) in recentHistory"
                        :key="index"
                        class="history-item"
                        :class="{ 'auto': interaction.isAuto }"
                      >
                        <div class="history-header">
                          <span class="history-type">
                            {{ interaction.isAuto ? '🤖 AI' : '👤 You' }}
                          </span>
                          <span class="history-time">{{ formatTime(interaction.timestamp) }}</span>
                        </div>
                        <div class="history-action">
                          <strong>Action:</strong> {{ interaction.action }}
                        </div>
                        <div class="history-response">
                          <strong>Result:</strong> {{ interaction.response }}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <!-- 动作输入 -->
                <div class="input-section">
                  <TextCraftActionInput
                    :environment-id="environmentId"
                    :suggested-action="suggestedAction"
                    :available-commands="availableCommands"
                    @action-sent="onActionSent"
                    @response-received="onResponseReceived"
                  />
                </div>
              </div>
            </template>
          </SplitPane>
        </div>
      </div>

      <!-- 自动运行状态指示 -->
      <div v-if="isAutoRunning" class="auto-run-toast">
        <div class="toast-content">
          <div class="toast-spinner"></div>
          <div class="toast-text">
            <div class="toast-title">🤖 AI is playing...</div>
            <div class="toast-subtitle">Round {{ autoRunRound }} • {{ totalActions }} actions</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive, computed, onMounted } from 'vue'
import TextCraftCompleteSystem from './TextCraftCompleteSystem.vue'
import TextCraftInventoryViewer from './TextCraftInventoryViewer.vue'
import TextCraftActionInput from './TextCraftActionInput.vue'
import SplitPane from './SplitPane.vue'
import TextCraftClient from '../environments/textcraft/client/TextCraftClient.js'
import aiAgent from '../../../shared/services/aiAgent.js'

export default {
  name: 'TextCraftPage',
  components: {
    TextCraftCompleteSystem,
    TextCraftInventoryViewer,
    TextCraftActionInput,
    SplitPane
  },
  setup() {
    // 界面模式切换
    const useCompleteSystem = ref(false)
    
    // 核心状态
    const environmentId = ref(null)
    const loading = ref(false)
    const creating = ref(false)
    const isAutoRunning = ref(false)
    const autoRunRound = ref(0)
    const totalActions = ref(0)
    
    // 环境状态
    const currentState = ref({})
    const currentObservation = ref('')
    const currentGoal = ref('')
    const availableCommands = ref([])
    const suggestedAction = ref('')
    const selectedItem = ref(null)
    
    // 交互历史
    const interactionHistory = ref([])
    
    // 创建配置
    const config = reactive({
      goal: '',
      commands: ''
    })
    
    // 客户端
    const client = new TextCraftClient()
    
    // 快速预设
    const quickPresets = ref([
      {
        name: 'Beginner Crafter',
        icon: '🔰',
        description: 'Learn basic crafting',
        goal: 'Craft your first wooden tools',
        commands: ''
      },
      {
        name: 'Builder',
        icon: '🏗️',
        description: 'Construction focus',
        goal: 'Build a shelter with walls and roof',
        commands: 'Focus on building materials'
      },
      {
        name: 'Tool Master',
        icon: '⚒️',
        description: 'Advanced crafting',
        goal: 'Create a full set of tools and weapons',
        commands: 'Emphasize tool progression'
      },
      {
        name: 'Free Play',
        icon: '🎲',
        description: 'Open exploration',
        goal: '',
        commands: ''
      }
    ])

    // 计算属性
    const recentHistory = computed(() => {
      return interactionHistory.value.slice(-10).reverse()
    })

    // 方法
    const applyPreset = (preset) => {
      config.goal = preset.goal
      config.commands = preset.commands
    }

    const createEnvironment = async () => {
      creating.value = true
      try {
        const result = await client.createEnvironment({
          goal: config.goal.trim() || null,
          commands: config.commands.trim() || null
        })
        
        if (result.success) {
          environmentId.value = result.data.id
          currentGoal.value = config.goal || 'Explore and craft freely'
          await refreshState()
          
          // 初始化AI
          await aiAgent.initializeConversation(
            environmentId.value.toString(),
            'textcraft',
            result.data.observation
          )
        }
      } catch (error) {
        console.error('Failed to create environment:', error)
      } finally {
        creating.value = false
      }
    }

    const refreshState = async () => {
      if (!environmentId.value) return
      
      loading.value = true
      try {
        const result = await client.getObservation(environmentId.value)
        if (result.success) {
          currentObservation.value = result.data
          currentState.value = { observation: result.data }
          
          // 解析可用命令
          availableCommands.value = client.getAvailableCommands(result.data)
          
          // 解析目标
          const goal = client.getCurrentGoal(result.data)
          if (goal) currentGoal.value = goal
        }
      } catch (error) {
        console.error('Failed to refresh state:', error)
      } finally {
        loading.value = false
      }
    }

    const resetEnvironment = async () => {
      if (!environmentId.value) return
      
      try {
        await client.reset(environmentId.value, 0)
        interactionHistory.value = []
        totalActions.value = 0
        autoRunRound.value = 0
        await refreshState()
      } catch (error) {
        console.error('Failed to reset environment:', error)
      }
    }

    const toggleAutoRun = async () => {
      if (isAutoRunning.value) {
        stopAutoRun()
      } else {
        await startAutoRun()
      }
    }

    const startAutoRun = async () => {
      isAutoRunning.value = true
      autoRunRound.value = 0
      
      try {
        while (isAutoRunning.value) {
          autoRunRound.value++
          
          const actionResult = await aiAgent.generateNextAction(
            environmentId.value,
            currentObservation.value
          )
          
          if (actionResult.shouldStop) break
          
          const action = actionResult.action
          totalActions.value++
          
          // 添加到历史
          addInteraction(action, '', true)
          
          // 执行动作
          const stepResult = await client.step(environmentId.value, action)
          if (stepResult.success) {
            const response = stepResult.data.observation || stepResult.data
            
            // 更新最后一条历史记录
            if (interactionHistory.value.length > 0) {
              interactionHistory.value[interactionHistory.value.length - 1].response = response
            }
            
            // 更新AI状态
            aiAgent.updateEnvironmentState(
              environmentId.value,
              response,
              stepResult.data.reward || 0,
              stepResult.data.done || false
            )
            
            await refreshState()
            
            if (stepResult.data.done) break
          }
          
          await new Promise(resolve => setTimeout(resolve, 2000))
        }
      } catch (error) {
        console.error('Auto run error:', error)
      }
      
      isAutoRunning.value = false
    }

    const stopAutoRun = () => {
      isAutoRunning.value = false
    }

    const onActionSent = (action) => {
      addInteraction(action, '', false)
      totalActions.value++
    }

    const onResponseReceived = (response) => {
      if (interactionHistory.value.length > 0) {
        interactionHistory.value[interactionHistory.value.length - 1].response = 
          response.observation || response.state || JSON.stringify(response)
      }
      refreshState()
    }

    const onSuggestAction = (action) => {
      suggestedAction.value = action
    }

    const onItemSelected = (item) => {
      selectedItem.value = item
    }

    const addInteraction = (action, response, isAuto = false) => {
      interactionHistory.value.push({
        action,
        response,
        isAuto,
        timestamp: new Date()
      })
      
      if (interactionHistory.value.length > 100) {
        interactionHistory.value.shift()
      }
    }

    const clearHistory = () => {
      interactionHistory.value = []
      client.clearHistory(environmentId.value)
    }

    const copyObservation = () => {
      if (currentObservation.value) {
        navigator.clipboard.writeText(currentObservation.value)
      }
    }

    const formatTime = (timestamp) => {
      return timestamp.toLocaleTimeString()
    }

    return {
      // 状态
      useCompleteSystem,
      environmentId,
      loading,
      creating,
      isAutoRunning,
      autoRunRound,
      totalActions,
      currentState,
      currentObservation,
      currentGoal,
      availableCommands,
      suggestedAction,
      selectedItem,
      interactionHistory,
      recentHistory,
      config,
      quickPresets,
      
      // 方法
      applyPreset,
      createEnvironment,
      refreshState,
      resetEnvironment,
      toggleAutoRun,
      onActionSent,
      onResponseReceived,
      onSuggestAction,
      onItemSelected,
      clearHistory,
      copyObservation,
      formatTime
    }
  }
}
</script>

<style scoped>
.textcraft-page {
  height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  font-family: 'Inter', 'Segoe UI', sans-serif;
}

.textcraft-simplified {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

/* 页面头部 */
.page-header {
  background: rgba(255, 255, 255, 0.95);
  border-bottom: 2px solid rgba(255, 255, 255, 0.3);
  backdrop-filter: blur(10px);
  padding: 1rem 2rem;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1400px;
  margin: 0 auto;
}

.page-title {
  margin: 0;
  font-size: 2rem;
  font-weight: 800;
  color: #2d3748;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.page-subtitle {
  margin: 0.25rem 0 0 0;
  color: #718096;
  font-size: 1rem;
}

.env-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #4a5568;
  font-weight: 500;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #48bb78;
  box-shadow: 0 0 8px rgba(72, 187, 120, 0.4);
}

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.btn-upgrade {
  background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
  color: white;
}

.btn-upgrade:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(251, 191, 36, 0.4);
}

/* 主内容区域 */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

/* 环境创建界面 */
.environment-creation {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
}

.creation-container {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 20px;
  padding: 3rem;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  max-width: 700px;
  width: 100%;
  backdrop-filter: blur(10px);
}

.creation-header {
  text-align: center;
  margin-bottom: 2.5rem;
}

.creation-header h2 {
  margin: 0 0 1rem 0;
  font-size: 2rem;
  color: #2d3748;
  font-weight: 800;
}

.creation-header p {
  margin: 0;
  color: #718096;
  font-size: 1.1rem;
  line-height: 1.6;
}

.form-section {
  margin-bottom: 2rem;
}

.form-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
  font-weight: 600;
  color: #4a5568;
  font-size: 1rem;
}

.label-icon {
  font-size: 1.2rem;
}

.form-textarea {
  width: 100%;
  padding: 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  font-family: inherit;
  font-size: 1rem;
  resize: vertical;
  transition: all 0.3s ease;
  background: white;
}

.form-textarea:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
  transform: translateY(-1px);
}

.preset-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
}

.preset-card {
  background: white;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  padding: 1.25rem;
  cursor: pointer;
  transition: all 0.3s ease;
  text-align: center;
}

.preset-card:hover {
  border-color: #667eea;
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
}

.preset-icon {
  font-size: 2rem;
  margin-bottom: 0.75rem;
}

.preset-name {
  font-weight: 600;
  color: #2d3748;
  margin-bottom: 0.25rem;
}

.preset-desc {
  font-size: 0.85rem;
  color: #718096;
}

.btn-create-large {
  width: 100%;
  background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
  color: white;
  padding: 1.25rem 2rem;
  font-size: 1.1rem;
  border-radius: 16px;
}

.btn-create-large:hover:not(:disabled) {
  background: linear-gradient(135deg, #38a169 0%, #2f855a 100%);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(72, 187, 120, 0.4);
}

.btn-create-large:disabled {
  background: #a0aec0;
  cursor: not-allowed;
  transform: none;
}

/* 游戏区域 */
.game-area {
  flex: 1;
  margin: 1rem;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

.left-panel, .right-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: 1.5rem;
  gap: 1.5rem;
}

.right-panel {
  background: #f8fafc;
}

/* 状态卡片 */
.status-card {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  border: 2px solid #e2e8f0;
}

.status-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.status-header h3 {
  margin: 0;
  color: #2d3748;
  font-weight: 700;
}

.status-controls {
  display: flex;
  gap: 0.5rem;
}

.control-btn {
  width: 36px;
  height: 36px;
  border: 2px solid #e2e8f0;
  background: white;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.control-btn:hover {
  border-color: #667eea;
  background: #667eea;
  color: white;
}

.control-btn.auto-btn.start {
  border-color: #48bb78;
}

.control-btn.auto-btn.start:hover {
  background: #48bb78;
}

.control-btn.auto-btn.stop {
  border-color: #f56565;
}

.control-btn.auto-btn.stop:hover {
  background: #f56565;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}

.info-label {
  font-weight: 600;
  color: #4a5568;
}

.info-value {
  color: #2d3748;
}

.info-value.active {
  color: #48bb78;
  font-weight: 600;
}

/* 库存和观察区域 */
.inventory-section {
  flex: 1;
  min-height: 300px;
}

.observation-section {
  background: white;
  border-radius: 12px;
  border: 2px solid #e2e8f0;
  overflow: hidden;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
}

.section-header h4 {
  margin: 0;
  color: #2d3748;
  font-weight: 700;
}

.copy-btn, .clear-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 1.1rem;
  color: #718096;
  transition: color 0.3s ease;
}

.copy-btn:hover, .clear-btn:hover {
  color: #667eea;
}

.observation-box {
  max-height: 200px;
  overflow-y: auto;
}

.observation-text {
  padding: 1.5rem;
  margin: 0;
  font-family: 'Fira Code', monospace;
  font-size: 0.9rem;
  line-height: 1.5;
  color: #2d3748;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.no-observation {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 150px;
  color: #a0aec0;
}

/* 交互历史 */
.history-section {
  flex: 1;
  background: white;
  border-radius: 12px;
  border: 2px solid #e2e8f0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.history-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
}

.no-history {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
  color: #a0aec0;
}

.no-history-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.no-history-text h5 {
  color: #4a5568;
  margin-bottom: 0.5rem;
}

.no-history-text p {
  margin: 0;
  color: #718096;
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.history-item {
  background: #f8fafc;
  border-radius: 8px;
  padding: 1rem;
  border-left: 4px solid #e2e8f0;
}

.history-item.auto {
  border-left-color: #667eea;
  background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.history-type {
  font-weight: 600;
  color: #4a5568;
}

.history-time {
  font-size: 0.8rem;
  color: #718096;
}

.history-action, .history-response {
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
  line-height: 1.4;
}

.history-action strong, .history-response strong {
  color: #2d3748;
}

/* 自动运行提示 */
.auto-run-toast {
  position: fixed;
  bottom: 2rem;
  right: 2rem;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 1.5rem;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(10px);
  border: 2px solid rgba(102, 126, 234, 0.3);
  z-index: 1000;
}

.toast-content {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.toast-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid #e2e8f0;
  border-top: 2px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.toast-title {
  font-weight: 600;
  color: #2d3748;
}

.toast-subtitle {
  font-size: 0.9rem;
  color: #718096;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* 响应式设计 */
@media (max-width: 768px) {
  .page-header {
    padding: 1rem;
  }
  
  .header-content {
    flex-direction: column;
    gap: 1rem;
  }
  
  .creation-container {
    padding: 2rem;
    margin: 1rem;
  }
  
  .preset-grid {
    grid-template-columns: 1fr;
  }
  
  .game-area {
    margin: 0.5rem;
  }
  
  .left-panel, .right-panel {
    padding: 1rem;
  }
  
  .auto-run-toast {
    bottom: 1rem;
    right: 1rem;
  }
}
</style>