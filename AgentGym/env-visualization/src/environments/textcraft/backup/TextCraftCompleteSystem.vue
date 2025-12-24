<template>
  <div class="textcraft-complete-system">
    <!-- 头部状态栏 -->
    <div class="status-header">
      <div class="environment-info">
        <div class="env-badge">
          <span class="env-icon">⚒️</span>
          <div class="env-details">
            <span class="env-name">TextCraft Environment</span>
            <span class="env-status" :class="{ 'connected': isConnected }">
              {{ isConnected ? '🟢 Connected' : '🔴 Disconnected' }}
            </span>
          </div>
        </div>
      </div>
      
      <div class="session-controls">
        <button 
          v-if="!environmentId" 
          @click="createEnvironment" 
          :disabled="creating"
          class="btn btn-create"
        >
          {{ creating ? '⏳ Creating...' : '✨ Create Environment' }}
        </button>
        
        <div v-if="environmentId" class="control-buttons">
          <button @click="refreshState" :disabled="loading" class="btn btn-refresh">
            {{ loading ? '⏳' : '🔄' }} Refresh
          </button>
          <button @click="resetEnvironment" :disabled="loading" class="btn btn-reset">
            🔄 Reset
          </button>
          <button 
            @click="toggleAutoRun" 
            :disabled="loading"
            :class="['btn', 'btn-auto', isAutoRunning ? 'btn-stop' : 'btn-start']"
          >
            {{ isAutoRunning ? '⏹️ Stop Auto' : '▶️ Start Auto' }}
          </button>
        </div>
      </div>
    </div>

    <!-- 主要内容区域 -->
    <div class="main-content">
      <!-- 环境创建配置 -->
      <div v-if="!environmentId && !loading" class="environment-setup">
        <div class="setup-card">
          <h3>🎯 TextCraft Environment Setup</h3>
          <p class="setup-description">Configure your crafting adventure!</p>
          
          <div class="config-form">
            <div class="form-group">
              <label>🎯 Goal (Optional)</label>
              <textarea
                v-model="setupConfig.goal"
                placeholder="e.g., Craft a wooden sword and shield"
                rows="2"
              ></textarea>
            </div>
            
            <div class="form-group">
              <label>📋 Custom Commands (Optional)</label>
              <textarea
                v-model="setupConfig.commands"
                placeholder="e.g., Special crafting recipes"
                rows="3"
              ></textarea>
            </div>
            
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
        </div>
      </div>

      <!-- 分割面板：环境视图 + 交互面板 -->
      <div v-if="environmentId" class="game-interface">
        <SplitPane :default-split="65">
          <template #left>
            <div class="environment-view">
              <!-- 环境信息卡片 -->
              <div class="env-info-card">
                <h4>🎮 Environment Status</h4>
                <div class="info-grid">
                  <div class="info-item">
                    <span class="info-label">🆔 ID:</span>
                    <span class="info-value">{{ environmentId }}</span>
                  </div>
                  <div class="info-item">
                    <span class="info-label">🎯 Goal:</span>
                    <span class="info-value">{{ currentGoal || 'Not specified' }}</span>
                  </div>
                  <div class="info-item">
                    <span class="info-label">🤖 Auto Running:</span>
                    <span class="info-value" :class="{ 'running': isAutoRunning }">
                      {{ isAutoRunning ? 'Yes' : 'No' }}
                    </span>
                  </div>
                </div>
              </div>

              <!-- 库存展示 -->
              <div class="inventory-display">
                <h4>🎒 Inventory</h4>
                <div v-if="inventory.length > 0" class="inventory-grid">
                  <div 
                    v-for="item in inventory" 
                    :key="item.itemName"
                    class="inventory-item"
                  >
                    <div class="item-icon">{{ getItemIcon(item.itemName) }}</div>
                    <div class="item-info">
                      <div class="item-name">{{ formatItemName(item.itemName) }}</div>
                      <div class="item-count">x{{ item.count }}</div>
                    </div>
                  </div>
                </div>
                <div v-else class="empty-inventory">
                  <span class="empty-icon">📦</span>
                  <span>Inventory is empty</span>
                </div>
              </div>

              <!-- 当前观察 -->
              <div class="observation-display">
                <h4>👁️ Current Observation</h4>
                <div class="observation-content">
                  <pre v-if="currentObservation" class="observation-text">{{ currentObservation }}</pre>
                  <div v-else class="no-observation">
                    <span>🤔 No observation available</span>
                  </div>
                </div>
              </div>

              <!-- 可用合成命令 -->
              <div v-if="availableCommands.length > 0" class="commands-display">
                <h4>📋 Available Crafting Commands</h4>
                <div class="commands-list">
                  <div 
                    v-for="(command, index) in availableCommands" 
                    :key="index"
                    class="command-item"
                  >
                    {{ command }}
                  </div>
                </div>
              </div>
            </div>
          </template>
          
          <template #right>
            <div class="interaction-panel">
              <!-- 交互历史 -->
              <div class="interaction-history">
                <h4>💬 Interaction History</h4>
                <div class="history-content">
                  <div v-if="interactionHistory.length === 0" class="no-interactions">
                    <span class="sun-icon">☀️</span>
                    <span>No interactions yet</span>
                    <p>Send an action to start interacting!</p>
                  </div>
                  <div v-else class="history-list">
                    <div 
                      v-for="(interaction, index) in interactionHistory" 
                      :key="index"
                      class="interaction-item"
                      :class="{ 'auto-interaction': interaction.isAuto }"
                    >
                      <div class="interaction-header">
                        <span class="interaction-type">
                          {{ interaction.isAuto ? '🤖 Auto' : '👤 Manual' }}
                        </span>
                        <span class="interaction-time">{{ formatTime(interaction.timestamp) }}</span>
                      </div>
                      <div class="interaction-action">
                        <strong>Action:</strong> {{ interaction.action }}
                      </div>
                      <div class="interaction-response">
                        <strong>Response:</strong> {{ interaction.response }}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <!-- 动作输入 -->
              <div class="action-input-section">
                <ActionInput
                  :environment-id="environmentId"
                  :environment-type="'textcraft'"
                  :suggested-action="suggestedAction"
                  :current-environment-state="currentState"
                  @action-sent="onActionSent"
                  @response-received="onResponseReceived"
                />
              </div>
            </div>
          </template>
        </SplitPane>
      </div>
    </div>

    <!-- 自动运行状态覆盖层 -->
    <div v-if="isAutoRunning" class="auto-run-overlay">
      <div class="auto-run-indicator">
        <div class="pulse-animation"></div>
        <span>🤖 AI Agent is thinking and acting...</span>
        <div class="auto-run-stats">
          <span>Round: {{ autoRunStats.round }}</span>
          <span>Actions: {{ autoRunStats.totalActions }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
import SplitPane from './SplitPane.vue'
import ActionInput from './ActionInput.vue'
import TextCraftClient from '../environments/textcraft/client/TextCraftClient.js'
import aiAgent from '../../../shared/services/aiAgent.js'

export default {
  name: 'TextCraftCompleteSystem',
  components: {
    SplitPane,
    ActionInput
  },
  setup() {
    // 核心状态
    const environmentId = ref(null)
    const isConnected = ref(false)
    const loading = ref(false)
    const creating = ref(false)
    const isAutoRunning = ref(false)
    
    // 环境状态
    const currentState = ref(null)
    const currentObservation = ref('')
    const currentGoal = ref('')
    const inventory = ref([])
    const availableCommands = ref([])
    const suggestedAction = ref('')
    
    // 交互历史
    const interactionHistory = ref([])
    
    // 自动运行统计
    const autoRunStats = reactive({
      round: 0,
      totalActions: 0
    })
    
    // 设置配置
    const setupConfig = reactive({
      goal: '',
      commands: ''
    })
    
    // TextCraft客户端
    const client = new TextCraftClient()
    
    // 预设配置
    const presets = ref([
      {
        name: 'Builder',
        icon: '🏗️',
        goal: 'Build a complete house with walls and roof',
        commands: 'Focus on construction materials and building blocks'
      },
      {
        name: 'Crafter',
        icon: '⚒️',
        goal: 'Master all basic crafting recipes',
        commands: 'Learn to craft tools, weapons, and useful items'
      },
      {
        name: 'Survivor',
        icon: '🛡️',
        goal: 'Create survival equipment and shelter',
        commands: 'Focus on survival essentials and protection'
      }
    ])

    // 测试连接
    const testConnection = async () => {
      try {
        await client.testConnection()
        isConnected.value = true
      } catch (error) {
        isConnected.value = false
        console.error('Connection failed:', error)
      }
    }

    // 创建环境
    const createEnvironment = async () => {
      creating.value = true
      try {
        const config = {
          goal: setupConfig.goal.trim() || null,
          commands: setupConfig.commands.trim() || null
        }
        
        const result = await client.createEnvironment(config)
        if (result.success) {
          environmentId.value = result.data.id
          currentGoal.value = config.goal || 'Complete crafting objectives'
          
          // 初始化AI对话
          await aiAgent.initializeConversation(
            environmentId.value.toString(), 
            'textcraft', 
            result.data.observation || ''
          )
          
          // 获取初始状态
          await refreshState()
        }
      } catch (error) {
        console.error('Failed to create environment:', error)
      } finally {
        creating.value = false
      }
    }

    // 刷新状态
    const refreshState = async () => {
      if (!environmentId.value) return
      
      loading.value = true
      try {
        const result = await client.getObservation(environmentId.value)
        if (result.success) {
          const observation = typeof result.data === 'string' ? result.data : result.data.observation
          currentObservation.value = observation
          currentState.value = result.data
          
          // 解析观察信息
          parseObservation(observation)
        }
      } catch (error) {
        console.error('Failed to refresh state:', error)
      } finally {
        loading.value = false
      }
    }

    // 解析观察信息
    const parseObservation = (observation) => {
      if (!observation) return

      // 解析库存
      const inventoryMatch = observation.match(/Inventory:\s*(.+?)(?=\n|$)/i)
      if (inventoryMatch) {
        const inventoryText = inventoryMatch[1].trim()
        if (inventoryText.includes("You are not carrying anything")) {
          inventory.value = []
        } else {
          const itemMatches = Array.from(inventoryText.matchAll(/\[([^\]]+)\]\s*\((\d+)\)/g))
          inventory.value = itemMatches.map(match => ({
            itemName: match[1].trim(),
            count: parseInt(match[2])
          }))
        }
      }

      // 解析可用命令
      const commandsMatch = observation.match(/Crafting commands:\s*([\s\S]*?)(?=Goal:|$)/i)
      if (commandsMatch) {
        const commandsText = commandsMatch[1].trim()
        availableCommands.value = commandsText.split('\n').filter(cmd => cmd.trim().startsWith('craft'))
      }

      // 解析目标
      const goalMatch = observation.match(/Goal:\s*(.+?)(?:\n|$)/i)
      if (goalMatch) {
        currentGoal.value = goalMatch[1].trim()
      }
    }

    // 重置环境
    const resetEnvironment = async () => {
      if (!environmentId.value) return
      
      loading.value = true
      try {
        const result = await client.reset(environmentId.value, 0)
        if (result.success) {
          interactionHistory.value = []
          await refreshState()
          
          // 重新初始化AI对话
          await aiAgent.initializeConversation(
            environmentId.value.toString(), 
            'textcraft', 
            result.data.observation || ''
          )
        }
      } catch (error) {
        console.error('Failed to reset environment:', error)
      } finally {
        loading.value = false
      }
    }

    // 切换自动运行
    const toggleAutoRun = async () => {
      if (isAutoRunning.value) {
        stopAutoRun()
      } else {
        await startAutoRun()
      }
    }

    // 开始自动运行
    const startAutoRun = async () => {
      if (!environmentId.value) return
      
      isAutoRunning.value = true
      autoRunStats.round = 0
      autoRunStats.totalActions = 0
      
      try {
        while (isAutoRunning.value) {
          autoRunStats.round++
          
          // 生成动作
          const actionResult = await aiAgent.generateNextAction(
            environmentId.value, 
            currentObservation.value
          )
          
          if (actionResult.shouldStop) {
            break
          }
          
          const action = actionResult.action
          autoRunStats.totalActions++
          
          // 添加到交互历史
          addInteraction(action, '', true)
          
          // 执行动作
          const stepResult = await client.step(environmentId.value, action)
          if (stepResult.success) {
            const response = stepResult.data.observation || stepResult.data
            
            // 更新交互历史的响应
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
            
            // 刷新环境状态
            await refreshState()
            
            // 检查是否完成
            if (stepResult.data.done) {
              break
            }
          }
          
          // 添加延迟
          await new Promise(resolve => setTimeout(resolve, 2000))
        }
      } catch (error) {
        console.error('Auto run error:', error)
      }
      
      isAutoRunning.value = false
    }

    // 停止自动运行
    const stopAutoRun = () => {
      isAutoRunning.value = false
    }

    // 应用预设
    const applyPreset = (preset) => {
      setupConfig.goal = preset.goal
      setupConfig.commands = preset.commands
    }

    // 手动动作处理
    const onActionSent = (action) => {
      addInteraction(action, '', false)
    }

    const onResponseReceived = (response) => {
      if (interactionHistory.value.length > 0) {
        interactionHistory.value[interactionHistory.value.length - 1].response = 
          response.observation || response.state || JSON.stringify(response)
      }
      refreshState()
    }

    // 添加交互记录
    const addInteraction = (action, response, isAuto = false) => {
      interactionHistory.value.push({
        action,
        response,
        isAuto,
        timestamp: new Date()
      })
      
      // 保持最近50条记录
      if (interactionHistory.value.length > 50) {
        interactionHistory.value.shift()
      }
    }

    // 工具函数
    const getItemIcon = (itemName) => {
      const icons = {
        'wood': '🪵', 'wooden_pickaxe': '⛏️', 'wooden_axe': '🪓',
        'stone': '🪨', 'iron': '⚙️', 'diamond': '💎',
        'stick': '🪄', 'plank': '📋', 'coal': '🔥'
      }
      for (const [key, icon] of Object.entries(icons)) {
        if (itemName.toLowerCase().includes(key)) {
          return icon
        }
      }
      return '📦'
    }

    const formatItemName = (itemName) => {
      return itemName.replace(/^minecraft:/, '')
                   .replace(/_/g, ' ')
                   .replace(/\b\w/g, l => l.toUpperCase())
    }

    const formatTime = (timestamp) => {
      return timestamp.toLocaleTimeString()
    }

    // 生命周期
    onMounted(() => {
      testConnection()
    })

    return {
      // 状态
      environmentId,
      isConnected,
      loading,
      creating,
      isAutoRunning,
      currentState,
      currentObservation,
      currentGoal,
      inventory,
      availableCommands,
      suggestedAction,
      interactionHistory,
      autoRunStats,
      setupConfig,
      presets,
      
      // 方法
      createEnvironment,
      refreshState,
      resetEnvironment,
      toggleAutoRun,
      applyPreset,
      onActionSent,
      onResponseReceived,
      getItemIcon,
      formatItemName,
      formatTime
    }
  }
}
</script>

<style scoped>
.textcraft-complete-system {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #2d3748;
  font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* 头部状态栏 */
.status-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background: rgba(255, 255, 255, 0.95);
  border-bottom: 2px solid rgba(255, 255, 255, 0.3);
  backdrop-filter: blur(10px);
}

.env-badge {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.env-icon {
  font-size: 2rem;
}

.env-details {
  display: flex;
  flex-direction: column;
}

.env-name {
  font-weight: 700;
  font-size: 1.2rem;
  color: #2d3748;
}

.env-status {
  font-size: 0.9rem;
  color: #718096;
}

.env-status.connected {
  color: #38a169;
}

.control-buttons {
  display: flex;
  gap: 0.75rem;
}

/* 按钮样式 */
.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.btn-create {
  background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
  color: white;
}

.btn-refresh {
  background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
  color: white;
}

.btn-reset {
  background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
  color: white;
}

.btn-start {
  background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
  color: white;
}

.btn-stop {
  background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
  color: white;
}

.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

/* 主内容区域 */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

/* 环境设置 */
.environment-setup {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
}

.setup-card {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 20px;
  padding: 2.5rem;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  max-width: 600px;
  width: 100%;
  backdrop-filter: blur(10px);
}

.setup-card h3 {
  text-align: center;
  margin-bottom: 1rem;
  font-size: 1.8rem;
  color: #2d3748;
}

.setup-description {
  text-align: center;
  color: #718096;
  margin-bottom: 2rem;
  font-size: 1.1rem;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 600;
  color: #4a5568;
}

.form-group textarea {
  width: 100%;
  padding: 0.75rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-family: inherit;
  resize: vertical;
  transition: border-color 0.3s;
}

.form-group textarea:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.preset-buttons {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-top: 1.5rem;
}

.preset-btn {
  padding: 1rem;
  background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-weight: 500;
}

.preset-btn:hover {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  transform: translateY(-2px);
}

/* 游戏界面 */
.game-interface {
  flex: 1;
  background: rgba(255, 255, 255, 0.95);
  margin: 1rem;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

.environment-view {
  padding: 1.5rem;
  height: 100%;
  overflow-y: auto;
  background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
}

.interaction-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: white;
}

/* 卡片样式 */
.env-info-card,
.inventory-display,
.observation-display,
.commands-display {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  border: 1px solid #e2e8f0;
}

.env-info-card h4,
.inventory-display h4,
.observation-display h4,
.commands-display h4 {
  margin: 0 0 1rem 0;
  color: #2d3748;
  font-size: 1.1rem;
  font-weight: 700;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.info-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem;
  background: #f8fafc;
  border-radius: 8px;
}

.info-label {
  font-weight: 600;
  color: #4a5568;
}

.info-value {
  color: #2d3748;
}

.info-value.running {
  color: #38a169;
  font-weight: 600;
}

/* 库存网格 */
.inventory-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 1rem;
}

.inventory-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 8px;
  border: 2px solid #e2e8f0;
  transition: all 0.3s ease;
}

.inventory-item:hover {
  border-color: #667eea;
  transform: translateY(-2px);
}

.item-icon {
  font-size: 2rem;
  margin-bottom: 0.5rem;
}

.item-name {
  font-weight: 600;
  color: #2d3748;
  text-align: center;
  font-size: 0.9rem;
}

.item-count {
  color: #718096;
  font-size: 0.8rem;
}

.empty-inventory {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem;
  color: #a0aec0;
}

.empty-icon {
  font-size: 3rem;
  margin-bottom: 0.5rem;
}

/* 观察显示 */
.observation-content {
  background: #f8fafc;
  border-radius: 8px;
  min-height: 150px;
}

.observation-text {
  padding: 1rem;
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

/* 命令列表 */
.commands-list {
  max-height: 200px;
  overflow-y: auto;
}

.command-item {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  font-family: 'Fira Code', monospace;
  font-size: 0.85rem;
  color: #2d3748;
  transition: all 0.2s ease;
}

.command-item:hover {
  background: #edf2f7;
  border-color: #cbd5e0;
}

/* 交互历史 */
.interaction-history {
  flex: 1;
  padding: 1.5rem;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.interaction-history h4 {
  margin: 0 0 1rem 0;
  color: #2d3748;
  font-weight: 700;
}

.history-content {
  flex: 1;
  overflow-y: auto;
}

.no-interactions {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #a0aec0;
  text-align: center;
}

.sun-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.interaction-item {
  background: #f8fafc;
  border-radius: 8px;
  padding: 1rem;
  border-left: 4px solid #e2e8f0;
}

.interaction-item.auto-interaction {
  border-left-color: #667eea;
  background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
}

.interaction-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.interaction-type {
  font-weight: 600;
  color: #4a5568;
}

.interaction-time {
  font-size: 0.8rem;
  color: #718096;
}

.interaction-action,
.interaction-response {
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
  line-height: 1.4;
}

.interaction-action strong,
.interaction-response strong {
  color: #2d3748;
}

/* 动作输入区域 */
.action-input-section {
  border-top: 1px solid #e2e8f0;
  background: white;
}

/* 自动运行覆盖层 */
.auto-run-overlay {
  position: fixed;
  bottom: 2rem;
  right: 2rem;
  background: rgba(255, 255, 255, 0.95);
  padding: 1.5rem 2rem;
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(10px);
  border: 2px solid rgba(102, 126, 234, 0.3);
}

.auto-run-indicator {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
  color: #2d3748;
}

.pulse-animation {
  width: 16px;
  height: 16px;
  background: #667eea;
  border-radius: 50%;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.5;
    transform: scale(1.2);
  }
}

.auto-run-stats {
  display: flex;
  gap: 1rem;
  font-size: 0.9rem;
  color: #718096;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .status-header {
    flex-direction: column;
    gap: 1rem;
    padding: 1rem;
  }
  
  .control-buttons {
    flex-wrap: wrap;
  }
  
  .setup-card {
    margin: 1rem;
    padding: 2rem;
  }
  
  .game-interface {
    margin: 0.5rem;
  }
  
  .info-grid {
    grid-template-columns: 1fr;
  }
  
  .inventory-grid {
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  }
}
</style>