<template>
  <div class="env-viewer">
    <!-- Connection Status -->
    <div class="connection-status">
      <span 
        class="status-indicator" 
        :class="connectionStatus.connected ? 'status-connected' : 'status-disconnected'"
      ></span>
      <span>{{ connectionStatus.connected ? 'Connected' : 'Disconnected' }}</span>
      <span class="env-type">{{ currentEnvironment.name || 'Unknown Environment' }}</span>
    </div>

    <!-- Debug Info -->
    <div v-if="debugMode" class="debug-info">
      <h4>🐛 Debug Info</h4>
      <div><strong>Environment ID:</strong> {{ environmentId }} ({{ typeof environmentId }})</div>
      <div><strong>Client Available:</strong> {{ !!currentEnvironmentClient }}</div>
      <div><strong>Creating:</strong> {{ creating }}</div>
      <div><strong>Loading:</strong> {{ loading }}</div>
      <div><strong>Can Create:</strong> {{ canCreate }}</div>
    </div>

    <!-- Error Display -->
    <div v-if="error" class="error-message">
      <strong>Error:</strong> {{ error }}
      <button @click="clearError" class="clear-error-btn">✕</button>
    </div>

    <!-- Environment Creation -->
    <div v-if="!environmentId && canCreate" class="creation-card">
      <h3>Create New Environment</h3>
      <p>{{ currentEnvironment.description }}</p>
      
      <!-- Debug Toggle -->
      <div class="debug-toggle">
        <label>
          <input type="checkbox" v-model="debugMode"> 
          🐛 Debug Mode
        </label>
      </div>
      
      <!-- Environment Configuration Component (if exists) -->
      <component 
        v-if="currentEnvironmentConfigComponent" 
        :is="currentEnvironmentConfigComponent"
        @config-ready="onConfigReady"
        @config-changed="(config) => envConfig = config"
      />
      
      <button 
        @click="createEnvironment" 
        :disabled="creating || !connectionStatus.connected"
        class="create-btn"
      >
        {{ creating ? '🚀 Creating...' : '✨ Create Environment' }}
      </button>
    </div>

    <!-- Environment Info -->
    <div v-if="environmentId" class="env-info-card">
      <div class="env-header">
        <h3>{{ currentEnvironment.name }} Environment</h3>
        <span class="env-id">ID: {{ environmentId }}</span>
      </div>
      
      <!-- State Summary -->
      <div v-if="envState" class="state-summary">
        <div class="state-item">
          <span class="state-label">Steps:</span>
          <span class="state-value">{{ envState.steps || 0 }}</span>
        </div>
        <div class="state-item">
          <span class="state-label">Reward:</span>
          <span class="state-value">{{ envState.reward || 0 }}</span>
        </div>
        <div class="state-item">
          <span class="state-label">Status:</span>
          <span class="state-value" :class="{ 'done': envState.done }">
            {{ envState.done ? 'Complete' : 'Active' }}
          </span>
        </div>
      </div>
    </div>

    <!-- Control Panel -->
    <div v-if="environmentId" class="control-panel">
      <div class="control-row">
        <button @click="refreshState" :disabled="loading" class="control-btn">
          {{ loading ? 'Refreshing...' : 'Refresh' }}
        </button>
        
        <button @click="showResetDialog" class="control-btn reset-btn">
          Reset
        </button>
        
        <button 
          @click="toggleAutoRun" 
          :class="['control-btn', 'auto-btn', { 'auto-active': isAutoRunning }]"
          :disabled="!connectionStatus.connected || !envState || creating"
        >
          {{ isAutoRunning ? 'Stop Auto' : 'Start Auto' }}
        </button>
        
        <!-- 显示执行状态 -->
        <div v-if="isAutoRunning && isExecutingAction" class="execution-status">
          <span class="execution-indicator">⚡</span>
          <span class="execution-text">Executing...</span>
        </div>
        
        <select v-model="autoRunDelay" class="delay-select" :disabled="isAutoRunning">
          <option value="1000">1s</option>
          <option value="2000">2s</option>
          <option value="3000">3s</option>
          <option value="5000">5s</option>
        </select>
      </div>
      
      <!-- Reset Dialog -->
      <div v-if="showResetOptions" class="dialog-overlay" @click.self="cancelReset">
        <div class="dialog-content">
          <div class="dialog-header">
            <h3>Reset Environment</h3>
            <button class="close-dialog-btn" @click="cancelReset">&times;</button>
          </div>
          
          <div class="dialog-body">
            <p>Are you sure you want to reset the environment?</p>
            
            <div class="reset-input-group">
              <label for="reset-data-idx">Data Index:</label>
              <input 
                type="number" 
                id="reset-data-idx" 
                v-model.number="resetDataIdx" 
                @input="validateResetDataIdx"
                :min="0"
              >
            </div>
            
            <p v-if="props.environmentType === 'textcraft'" class="reset-info">
              <span class="info-icon">ℹ️</span>
              <span>
                In TextCraft, data_idx determines which crafting goal to use.
                <br>
                The default value (0) will reset to the first crafting goal.
                <br>
                Higher values will select different crafting goals.
              </span>
            </p>
          </div>
          
          <div class="dialog-footer">
            <button class="btn-secondary" @click="cancelReset">Cancel</button>
            <button class="btn-primary" @click="confirmReset">Confirm Reset</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Environment Viewer Component -->
    <div v-if="environmentId && !loading" class="viewer-container">
      <div v-if="props.environmentType === 'textcraft'" class="env-specific-viewer">
        <TextCraftViewer 
          ref="textcraftViewerRef"
          :key="`textcraft-${environmentId}`"
          :environment-id="environmentId" 
          :client="currentEnvironmentClient"
          :env-state="envState"
          @action-executed="onActionExecuted"
          @goal-completed="onGoalCompleted"
          @state-updated="onStateUpdated"
          @environment-reset="onEnvironmentReset"
        />
      </div>
      <div v-else-if="props.environmentType === 'babyai'" class="env-specific-viewer">
        <BabyAIViewer 
          :key="`babyai-${environmentId}`"
          :environment-id="environmentId" 
          :client="currentEnvironmentClient"
          :env-state="envState"
          @action-executed="onActionExecuted"
          @goal-completed="onGoalCompleted"
          @state-updated="onStateUpdated"
        />
      </div>
      <div v-else-if="props.environmentType === 'sciworld'" class="env-specific-viewer">
        <SciWorldViewer 
          :key="`sciworld-${environmentId}`"
          :environment-id="environmentId" 
          :client="currentEnvironmentClient"
          :env-state="envState"
          @action-executed="onActionExecuted"
          @goal-completed="onGoalCompleted"
          @state-updated="onStateUpdated"
          @direct-response="onDirectResponse"
          @auto-action-sent="onAutoActionSent"
          @auto-response-received="onAutoResponseReceived"
        />
      </div>
      <div v-else-if="props.environmentType === 'webarena'" class="env-specific-viewer">
        <WebArenaViewer 
          :key="`webarena-${environmentId}`"
          :environment-id="environmentId" 
          :client="currentEnvironmentClient"
          :env-state="envState"
          @action-executed="onActionExecuted"
          @goal-completed="onGoalCompleted"
          @state-updated="onStateUpdated"
          @observation-updated="onObservationUpdated"
        />
      </div>
      <div v-else-if="props.environmentType === 'searchqa'" class="env-specific-viewer">
        <SearchQAViewer 
          :key="`searchqa-${environmentId}`"
          :environment-id="environmentId" 
          :client="currentEnvironmentClient"
          :env-state="envState"
          :last-auto-action="props.lastAutoAction"
          :last-auto-response="props.lastAutoResponse"
          @action-executed="onActionExecuted"
          @goal-completed="onGoalCompleted"
          @state-updated="onStateUpdated"
        />
      </div>
      <div v-else class="no-viewer-message">
        No viewer available for environment type: {{ props.environmentType }}
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="loading-state">
      <div class="loading-spinner"></div>
      <span>Loading environment...</span>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { getEnvironment } from '../../environments/index.js'
import environmentManager from '../services/environmentManager.js'
import aiAgent from '../services/aiAgent.js'

// Import environment viewers directly
import TextCraftViewer from '../../environments/textcraft/components/TextcraftViewer.vue'
import BabyAIViewer from '../../environments/babyai/components/BabyAIViewer.vue'
import SciWorldViewer from '../../environments/sciworld/components/SciWorldViewer.vue'
import WebArenaViewer from '../../environments/webarena/components/WebArenaViewer.vue'
import SearchQAViewer from '../../environments/searchqa/components/SearchQAViewer.vue'

export default {
  name: 'EnvViewer',
  components: {
    TextCraftViewer,
    BabyAIViewer,
    SciWorldViewer,
    WebArenaViewer,
    SearchQAViewer,
  },
  props: {
    environmentId: {
      type: [String, Number],
      default: null
    },
    environmentType: {
      type: String,
      default: 'textcraft'
    },
    lastAutoAction: {
      type: Object,
      default: null
    },
    lastAutoResponse: {
      type: Object,
      default: null
    }
  },
  emits: [
    'environment-created',
    'environment-reset',
    'state-updated',
    'suggest-action',
    'auto-action-sent',
    'auto-response-received',
    'observation-updated', 
    'sciworld-debug' 
  ],
  setup(props, { emit, expose }) {
    // State
    const loading = ref(false)
    const creating = ref(false)
    const error = ref('')
    const envState = ref(null)
    const showResetOptions = ref(false)
    const currentEnvironmentClient = ref(null)
    const isAutoRunning = ref(false)
    const autoRunTimer = ref(null)
    const autoRunDelay = ref(1500) 
    const lastActionTime = ref(null) 
    const lastAutoActionTime = ref(null) 
    const isExecutingAction = ref(false) 
    const connectionStatus = ref({ connected: false, message: 'Not connected' })
    const canCreate = ref(true)
    const envConfig = ref({})
    const debugMode = ref(false)
    const resetDataIdx = ref(0)
    const environmentConfig = ref(null)
    
    // Find the client for the current environment
    const getEnvironmentClient = () => {
      return props.client
    }

    // References to specific environment viewers
    const textcraftViewerRef = ref(null)

    // Flag to prevent duplicate client loading
    const isClientLoading = ref(false)
    const lastLoadedType = ref('')

    // Environment initialization and lifecycle management
    const environmentReady = ref(false);
    
    // Handle environment creation callback
    const onEnvironmentCreated = (id) => {
      console.log(`✅ EnvViewer: Environment ${id} created`);
      emit('environment-created', id);
    };
    
    // Handle reset confirmation
    const confirmReset = async () => {
      showResetOptions.value = false;
      
      if (!currentEnvironmentClient.value || !props.environmentId) return;
      
      // Prevent multiple resets in quick succession
      const now = Date.now();
      if (resetInProgress || (now - lastResetTime < 500)) {
        console.log('⚠️ Reset operation in progress or called too frequently, skipping');
        return;
      }
      
      resetInProgress = true;
      lastResetTime = now;
      loading.value = true;
      
      try {
        // Use the user-specified data_idx without restriction
        const finalResetIdx = resetDataIdx.value;
        
        console.log(`🔄 EnvViewer: Resetting environment ${props.environmentId} with data_idx=${finalResetIdx}`);
        
        const result = await currentEnvironmentClient.value.reset(
          props.environmentId, 
          finalResetIdx
        );
        
        if (result && result.success) {
          console.log('✅ Environment reset successfully');
          
          emit('environment-reset', {
            result,
            clearHistory: true 
          });
          
          const observation = await currentEnvironmentClient.value.getObservation(props.environmentId)
          let observationText = '';
          if (observation) {
            console.log('🔍 EnvViewer reset: Processing observation:', typeof observation, observation);
            
            if (typeof observation === 'string') {
              observationText = observation;
              console.log('✅ EnvViewer reset: Using string data directly');
            } else if (observation.observation) {
              observationText = typeof observation.observation === 'string'
                ? observation.observation
                : JSON.stringify(observation.observation, null, 2);
              console.log('✅ EnvViewer reset: Extracted observation field');
            } else {
              observationText = JSON.stringify(observation, null, 2);
              console.log('✅ EnvViewer reset: Converted data object to JSON');
            }
          }
          
          if (typeof observationText !== 'string') {
            console.warn('⚠️ EnvViewer reset: observationText is not string, converting:', typeof observationText);
            observationText = String(observationText);
          }
          
          console.log('📝 EnvViewer reset: Final observationText:', observationText.substring(0, 100) + '...');
          
          envState.value = {
            observation: observationText,
            steps: 0,
            reward: 0,
            done: false
          };
          
          console.log('🔄 Reset environment state:', envState.value);
          
          console.log('🧹 Clearing AI conversation history...');
          aiAgent.clearConversation(props.environmentId);
          
          console.log('🚀 Initializing new AI conversation...');
          aiAgent.initializeConversation(
            props.environmentId,
            props.environmentType,
            observationText
          );
          
          emit('state-updated', envState.value);
          
          console.log('✅ Environment reset and initialization complete');
        } else {
          throw new Error(result?.error || 'Reset failed');
        }
      } catch (err) {
        console.error('❌ Failed to reset environment:', err);
        error.value = err.message;
      } finally {
        loading.value = false;
        // Small delay before allowing another reset
        setTimeout(() => {
          resetInProgress = false;
        }, 500);
      }
    };
    
    // Handle component mount
    onMounted(async () => {
      console.log('🚀 EnvViewer mounted');
      console.log('📋 Props:', props);
      
      // Initial client load
      await loadClient(props.environmentType);
      
      // Track component ready state
      environmentReady.value = true;
    });

    // Get current environment configuration
    const currentEnvironment = computed(() => {
      return getEnvironment(props.environmentType) || getEnvironment('textcraft')
    })

    // Load client while preventing duplicate loading
    const loadClient = async (environmentType, force = false) => {
      // Prevent duplicate loading of the same client
      if (!force && environmentType === lastLoadedType.value && currentEnvironmentClient.value) {
        console.log(`✅ Client already loaded for: ${environmentType}`)
        return currentEnvironmentClient.value
      }

      if (isClientLoading.value) {
        console.log(`⏳ Client loading in progress for: ${environmentType}`)
        return null
      }

      isClientLoading.value = true
      error.value = ''

      try {
        console.log(`🔧 Loading client for: ${environmentType}`)
        const client = await environmentManager.getClient(environmentType)
        
        if (client) {
          currentEnvironmentClient.value = client
          lastLoadedType.value = environmentType
          console.log(`✅ Client loaded successfully: ${environmentType}`)
          
          // Test connection after loading client
          await testConnection()
        } else {
          throw new Error(`Failed to load client for ${environmentType}`)
        }

        return client
      } catch (err) {
        console.error(`❌ Failed to load client for ${environmentType}:`, err)
        error.value = `Failed to load client: ${err.message}`
        currentEnvironmentClient.value = null
        return null
      } finally {
        isClientLoading.value = false
      }
    }

    // Get current environment configuration component
    const currentEnvironmentConfigComponent = ref(null)

    // Test server connection
    const testConnection = async () => {
      if (!currentEnvironmentClient.value) {
        connectionStatus.value.connected = false
        return
      }
      
      try {
        const result = await currentEnvironmentClient.value.testConnection()
        connectionStatus.value.connected = result.success || false
        console.log(`🔗 Connection status: ${connectionStatus.value.connected}`)
      } catch (err) {
        connectionStatus.value.connected = false
        console.error(`❌ Connection test failed:`, err)
      }
    }

    // Clear error
    const clearError = () => {
      error.value = ''
    }

    // Configuration ready callback
    const onConfigReady = (configData) => {
      canCreate.value = true
      envConfig.value = configData
      console.log('Config ready:', configData)
    }

    // Create environment - fixed to handle promises correctly
    const createEnvironment = async () => {
      console.log('🚀 Create Environment button clicked')
      
      if (!currentEnvironmentClient.value) {
        error.value = 'No client available for environment creation'
        console.error('❌ No client available')
        return
      }

      creating.value = true
      error.value = ''

      try {
        console.log('🏗️ Creating environment with config:', envConfig.value)
        console.log('🔍 Using client:', currentEnvironmentClient.value.constructor.name)
        
        const config = envConfig.value || {}
        
        // Special handling for specific environments
        let result;
        if (props.environmentType === 'sciworld') {
          // For sciworld, we need to pass a data index for automatic reset
          const dataIdx = config.dataIdx !== undefined ? config.dataIdx : 0;
          console.log(`🔍 Creating sciworld environment with dataIdx=${dataIdx}`);
          result = await currentEnvironmentClient.value.createEnvironment(dataIdx);
        } else {
          // For other environments, use the standard approach
          result = await currentEnvironmentClient.value.createEnvironment(config);
        }
        
        console.log('📋 Full environment creation result:', result)
        
        if (!result) {
          throw new Error('Empty result received from createEnvironment')
        }
        
        if (!result.success) {
          throw new Error(result.error || 'Failed to create environment')
        }
        
        // Extract environment ID from result
        const data = result.data
        if (!data) {
          throw new Error('No data received in successful result')
        }
        
        const extractedId = data.id || data.environmentId || data.env_idx || result.environmentId
        if (!extractedId) {
          throw new Error('No environment ID found in result data')
        }
        
        const finalId = parseInt(extractedId)
        console.log('✅ Environment created with ID:', finalId)
        
        // Emit environment creation event
        emit('environment-created', finalId)
        
        // Get initial state
        await nextTick()
        await refreshState()
        
        // Initialize AI conversation if applicable
        if (envState.value?.observation) {
          window.currentEnvironmentType = props.environmentType;
          console.log('EnvViewer设置全局环境类型:', window.currentEnvironmentType);
          
          // Ensure the observation is properly formatted for AI Agent
          let observationForAI = envState.value.observation;
          if (typeof observationForAI !== 'string') {
            console.warn('⚠️ AI Agent: observation is not a string, converting...');
            observationForAI = JSON.stringify(observationForAI, null, 2);
          }
          
          console.log('🤖 AI Agent initialization observation type:', typeof observationForAI);
          console.log('🤖 AI Agent initialization observation preview:', observationForAI.substring(0, 200) + '...');
          
          aiAgent.initializeConversation(
            finalId.toString(),
            props.environmentType,
            observationForAI
          )
        }

        await testConnection()
        
        console.log('🎯 Environment fully initialized and ready for auto run')
      } catch (err) {
        console.error('❌ Failed to create environment:', err)
        error.value = err.message
      } finally {
        creating.value = false
      }
    }

    // Refresh state - fixed to handle promises correctly
    const refreshState = async () => {
      if (!props.environmentId || !currentEnvironmentClient.value) {
        console.warn('⚠️ Cannot refresh state: missing environmentId or client')
        return
      }
      
      // Prevent multiple refreshes in quick succession
      if (loading.value) {
        console.log('⏱️ Refresh already in progress, skipping');
        return;
      }

      loading.value = true
      error.value = ''
      
      try {
        console.log('🔄 Refreshing state for environment:', props.environmentId)
        
        // Make sure client exists and has getObservation method
        if (!currentEnvironmentClient.value.getObservation) {
          throw new Error('Client does not have getObservation method')
        }
        
        const result = await currentEnvironmentClient.value.getObservation(props.environmentId)
        console.log('📋 Raw observation result:', result)
        
        if (!result || !result.success) {
          throw new Error(result?.error || 'Failed to get observation')
        }
        
        // Extract observation data (ensure it's a string)
        let observationData = result.data
        if (typeof observationData !== 'string') {
          // Handle different data structures from different environments
          if (observationData && typeof observationData === 'object') {
            // For SearchQA and other environments that return structured data
            if (observationData.raw) {
              // SearchQA: use the raw observation text
              observationData = observationData.raw
            } else if (observationData.observation) {
              // Other environments: use the observation field
              observationData = typeof observationData.observation === 'string' 
                ? observationData.observation 
                : JSON.stringify(observationData.observation, null, 2)
            } else if (observationData.question) {
              // If no raw but has question, construct basic observation
              observationData = `Question: ${observationData.question}`
            } else {
              // Fallback to JSON formatting
              observationData = JSON.stringify(observationData, null, 2)
            }
          } else {
            // For primitive types
            observationData = String(observationData || '')
          }
        }
        
        console.log('🔍 Processed observation type:', typeof observationData)
        console.log('🔍 Processed observation preview:', observationData.substring(0, 100) + '...')
        
        // Update environment state
        envState.value = {
          observation: observationData,
          steps: envState.value?.steps || 0,
          reward: envState.value?.reward || 0,
          done: envState.value?.done || false
        }
        
        emit('state-updated', envState.value)
      } catch (err) {
        console.error('❌ Failed to refresh state:', err)
        error.value = err.message
      } finally {
        loading.value = false
      }
    }

    // Reset dialog functions
    const showResetDialog = () => {
      showResetOptions.value = true;
    }

    const cancelReset = () => {
      showResetOptions.value = false;
    }

    // Reset operation debounce control
    let resetInProgress = false;
    let lastResetTime = 0;

    // Auto-run control
    const toggleAutoRun = async () => {
      if (isAutoRunning.value) {
        stopAutoRun()
      } else {
        await startAutoRun()
      }
    }

    const startAutoRun = async () => {
      if (!currentEnvironmentClient.value || !props.environmentId) {
        console.warn('⚠️ Cannot start auto run: no client or environment')
        return
      }

      if (!envState.value) {
        console.warn('⚠️ Environment state not ready, attempting to refresh...')
        try {
          await refreshState()
          if (!envState.value) {
            console.error('❌ Failed to get environment state')
            return
          }
        } catch (error) {
          console.error('❌ Failed to refresh state:', error)
          return
        }
      }

      if (!connectionStatus.value.connected) {
        console.warn('⚠️ Cannot start auto run: not connected to server')
        return
      }

      console.log('✅ All pre-conditions met for auto run')
      console.log('🔍 Environment state preview:', {
        hasState: !!envState.value,
        stateType: typeof envState.value,
        hasObservation: !!(envState.value?.observation || envState.value)
      })

      isAutoRunning.value = true
      console.log('▶️ Starting auto run immediately')
      
      performAutoAction();
      
      autoRunTimer.value = setInterval(async () => {
        await performAutoAction()
      }, autoRunDelay.value)
    }

    const stopAutoRun = () => {
      if (autoRunTimer.value) {
        clearInterval(autoRunTimer.value)
        autoRunTimer.value = null
      }
      isAutoRunning.value = false
      isExecutingAction.value = false
      console.log('⏹️ Auto run stopped')
    }

    const performAutoAction = async () => {
      if (!currentEnvironmentClient.value || !props.environmentId || !envState.value) {
        console.warn('⚠️ Cannot perform auto action: missing requirements', {
          hasClient: !!currentEnvironmentClient.value,
          hasEnvId: !!props.environmentId,
          hasState: !!envState.value
        })
        return
      }

      // 检查连接状态
      if (!connectionStatus.value.connected) {
        console.warn('⚠️ Cannot perform auto action: not connected to server')
        return
      }

      // 防并发执行机制 - 如果正在执行action，则跳过
      if (isExecutingAction.value) {
        console.log('🔄 Auto action skipped - previous action still executing')
        return
      }

      // 标记开始执行
      isExecutingAction.value = true
      const now = Date.now()
      lastAutoActionTime.value = now

      console.log('🚀 Starting auto action execution...')

      try {
        // Get the current observation - handle different environment formats
        let currentObservation = '';
        
        if (props.environmentType === 'webarena') {
          // For WebArena, envState.value is the observation string directly
          currentObservation = typeof envState.value === 'string' 
            ? envState.value 
            : JSON.stringify(envState.value);
        } else {
          // For other environments, get from envState.value.observation
          let observationData = envState.value.observation;
          if (typeof observationData === 'string') {
            currentObservation = observationData;
          } else if (observationData && typeof observationData === 'object') {
            // Use the same logic as refreshState for consistent handling
            if (observationData.raw) {
              // SearchQA: use the raw observation text
              currentObservation = observationData.raw;
            } else if (observationData.observation) {
              // Other environments: use the observation field
              currentObservation = typeof observationData.observation === 'string' 
                ? observationData.observation 
                : JSON.stringify(observationData.observation, null, 2);
            } else if (observationData.question) {
              // If no raw but has question, construct basic observation
              currentObservation = `Question: ${observationData.question}`;
            } else {
              // Fallback to JSON formatting
              currentObservation = JSON.stringify(observationData, null, 2);
            }
          } else {
            // For primitive types
            currentObservation = String(observationData || '');
          }
        }
        
        console.log('🔍 performAutoAction currentObservation type:', typeof currentObservation);
        console.log('🔍 performAutoAction currentObservation preview:', currentObservation.substring(0, 100) + '...');
        
        // // 检查是否存在重复的"Your task is to..."内容，这是SciWorld环境中常见的重复问题
        // if (props.environmentType === 'sciworld' && 
        //     currentObservation.match(/Your task is to [^\.]+\./gi)?.length > 1) {
        //   console.warn('⚠️ Detected duplicate task descriptions in observation, clearing AI conversation')
        //   aiAgent.clearConversation(props.environmentId)
        //   aiAgent.initializeConversation(
        //     props.environmentId,
        //     props.environmentType,
        //     currentObservation, // 它会处理去重
        //     20 // 默认最大轮数
        //   )
        // }
        
        // Generate action
        const actionResult = await aiAgent.generateNextAction(
          props.environmentId, 
          currentObservation
        )

        if (actionResult.shouldStop) {
          console.log('🛑 Auto run stopped by AI:', actionResult.reason)
          stopAutoRun()
          return
        }

        if (!actionResult.action) {
          console.warn('⚠️ No action generated, continuing...')
          return
        }

        console.log('📤 Executing action:', actionResult.action)

        // Execute action
        console.log('🔄 Calling step method with:', {
          environmentId: props.environmentId,
          action: actionResult.action,
          client: !!currentEnvironmentClient.value
        })
        
        const stepResult = await currentEnvironmentClient.value.step(
          props.environmentId, 
          actionResult.action
        )
        
        console.log('📥 Step result received:', stepResult)

        if (stepResult.success) {
          console.log('✅ Auto action executed:', actionResult.action)
          
          // Extract observation text from the response
          const observationText = extractObservationText(stepResult.data)
          
          // Update state
          envState.value = {
            observation: observationText,
            steps: (envState.value.steps || 0) + 1,
            reward: stepResult.data.reward || 0,
            done: stepResult.data.done || false
          }

          // Update AI environment state
          aiAgent.updateEnvironmentState(
            props.environmentId,
            observationText, // Pass text, not object
            envState.value.reward,
            envState.value.done
          )

          emit('auto-action-sent', {
            action: actionResult.action,
            result: {
              observation: observationText,
              reward: stepResult.data.reward,
              done: stepResult.data.done
            }
          })
          
          // For SciWorld, let the component handle state updates to avoid duplicates
          if (props.environmentType !== 'sciworld') {
            emit('auto-response-received', {
              observation: observationText,
              reward: stepResult.data.reward,
              done: stepResult.data.done
            })
            emit('state-updated', envState.value)
          } else {
            console.log('⏩ SciWorld: Skipping duplicate state emissions from autoAction')
            
            // For debugging, explicitly emit these events to see if they're being handled
            console.log('🔍 Debug: SciWorld environment response:', observationText.substring(0, 50) + '...')
            
            // Special debug event for SciWorld
            emit('sciworld-debug', {
              observation: observationText,
              action: actionResult.action,
              timestamp: new Date().toISOString()
            })
          }

          // If task is complete, stop auto run
          if (envState.value.done) {
            console.log('🎯 Task completed, stopping auto run')
            stopAutoRun()
          }
        }
      } catch (err) {
        console.error('❌ Auto action failed:', err)
        // Don't stop auto run, just log error
      } finally {
        // 确保在任何情况下都清除执行标记
        isExecutingAction.value = false
        console.log('🏁 Auto action execution completed')
      }
    }

    // Helper function to extract observation text from response
    const extractObservationText = (data) => {
      if (!data) return ''
      
      if (typeof data === 'string') {
        return data
      }
      
      if (typeof data === 'object') {
        if (data.observation && typeof data.observation === 'string') {
          return data.observation
        }
        if (data.state && typeof data.state === 'string') {
          return data.state
        }
        
        // Fall back to stringifying the object
        return JSON.stringify(data, null, 2)
      }
      
      return String(data)
    }

    // Event handlers
    const onActionExecuted = (data) => {
      emit('auto-action-sent', data)
    }

    const onGoalCompleted = (data) => {
      console.log('🎯 Goal completed:', data)
      stopAutoRun()
    }

    const onEnvironmentReset = (data) => {
      emit('environment-reset', data)
    }

    const onStateUpdated = (data) => {
      // Update the local state
      envState.value = data
      
      // For SciWorld, handle special to avoid duplicate emissions
      if (props.environmentType === 'sciworld') {
        // We still need to emit events to parent components
        emit('state-updated', data)
      } else {
        // For other environments, normal propagation
        emit('state-updated', data)
      }
    }

    const onSuggestAction = (data) => {
      emit('suggest-action', data)
    }

    const onAutoActionSent = (data) => {
      emit('auto-action-sent', data)
    }

    const onAutoResponseReceived = (data) => {
      emit('auto-response-received', data)
    }

    const onObservationUpdated = (data) => {
      console.log('📄 EnvViewer: Observation updated from WebArena:', data)
      emit('observation-updated', data)
    }

    // Critical action result handler for SciWorld
    const onCriticalActionResult = (data) => {
      console.log('🚨 Critical action result received:', data.action);
      
      // Always emit this as an auto-response to ensure it shows up in the history
      emit('auto-response-received', {
        observation: data.observation,
        action: data.action,
        timestamp: data.timestamp
      });
    };

    // Direct response handler for SciWorld
    const onDirectResponse = (data) => {
      console.log('📝 SciWorld direct response received:', data.action);
      console.log('📝 Response observation content:', data.observation?.substring(0, 100));
      
      // Always emit this as an auto-response to ensure it shows up in the history
      emit('auto-response-received', {
        observation: data.observation,
        action: data.action,
        timestamp: data.timestamp || new Date().toISOString(),
        forceDisplay: data.forceDisplay || false
      });
      
      console.log('📝 Emitted auto-response-received event to EnvironmentView');
    };

    // Format date
    const formatDate = (date) => {
      return new Date(date).toLocaleString()
    }

    // Method to pass agent actions to TextcraftViewer
    const addAgentAction = (action) => {
      console.log('📢 Forwarding agent action to environment viewer:', action)
      
      // Get the textcraft component reference
      if (props.environmentType === 'textcraft' && textcraftViewerRef.value) {
        textcraftViewerRef.value.onAgentAction && textcraftViewerRef.value.onAgentAction(action)
      } else {
        console.warn('❌ Could not forward action, environment component not found')
      }
    }

    // Step action handler function
    const stepAction = async (action) => {
      console.log(`🎬 Stepping with action: ${action}`)
      
      if (!currentEnvironmentClient.value || !props.environmentId) {
        error.value = 'Environment not initialized'
        return null
      }
      
      error.value = ''
      loading.value = true
      
      try {
        // Call the step method on the environment client
        const result = await currentEnvironmentClient.value.step(props.environmentId, action)
        
        // Check for errors
        if (!result.success) {
          error.value = `Error executing action: ${result.error}`
          return null
        }
        
        // Update state
        const observationText = extractObservationText(result.data)
        envState.value = {
          observation: observationText,
          steps: (envState.value?.steps || 0) + 1,
          reward: result.data.reward || envState.value?.reward || 0,
          done: result.data.done || envState.value?.done || false
        }
        
        // For SciWorld, we'll let the SciWorldViewer component handle state updates
        // to avoid duplicates. For other environments, emit events here.
        if (props.environmentType !== 'sciworld') {
          // Emit state updated event
          emit('state-updated', envState.value)
          
          // Also emit auto-response-received to ensure it's captured in the interaction history
          emit('auto-response-received', envState.value)
        } else {
          // For SciWorld, just log that we're skipping duplicate emissions
          console.log('⏩ SciWorld: Skipping duplicate state emissions from EnvViewer')
        }
        
        return result.data
      } catch (e) {
        error.value = `Error: ${e.message}`
        console.error('Error stepping environment:', e)
        return null
      } finally {
        loading.value = false
      }
    }

    // Create a user-action handler that acts as a bridge to stepAction
    const handleUserAction = async (action) => {
      console.log('🎭 Handling user action:', action)
      
      // Use the same stepAction function to ensure consistency
      const result = await stepAction(action)
      
      if (result) {
        // No need to emit events here as stepAction already does this
        console.log('✅ User action processed successfully')
      } else {
        console.error('❌ Failed to process user action')
      }
      
      return result
    }

    // Watch for environment type changes
    watch(() => props.environmentType, async (newType, oldType) => {
      if (newType !== oldType && newType) {
        console.log(`🔄 Environment type changed from ${oldType} to ${newType}`);
        
        // Store environment type globally for other components to access
        window.currentEnvironmentType = newType;
        
        // Stop auto run
        stopAutoRun();
        
        // Reset state
        envState.value = null;
        error.value = '';
        
        // Reset data index to appropriate default for this environment
        if (newType === 'textcraft') {
          resetDataIdx.value = 0; // TextCraft is very sensitive to data_idx
        } else if (newType === 'babyai') {
          resetDataIdx.value = 0; // BabyAI default
        } else {
          resetDataIdx.value = 0; // Safe default for other environments
        }
        
        // Load new client
        await loadClient(newType, true); // Force reload
      }
    })

    // Watch for environment ID changes
    watch(() => props.environmentId, async (newId, oldId) => {
      if (newId !== oldId) {
        console.log(`🆔 Environment ID changed from ${oldId} to ${newId}`)
        if (newId && currentEnvironmentClient.value) {
          // We'll let the child component (SciWorldViewer) handle its own state initialization
          // This prevents duplicate refresh calls when components are mounted
          if (props.environmentType !== 'sciworld') {
            await nextTick()
            // Use a delayed refresh to avoid race conditions with mounting
            setTimeout(() => refreshState(), 200)
          } else {
            console.log('⏭️ EnvViewer: Skipping automatic refresh for SciWorld to prevent duplicates');
          }
        }
      }
    })

    onMounted(async () => {
      console.log('🚀 EnvViewer mounted')
      console.log('📋 Props:', props)
      
      // Store current environment type globally
      window.currentEnvironmentType = props.environmentType;
      
      // Initial load
      await loadClient(props.environmentType)
      
      // If environment ID exists, refresh state - but only for non-SciWorld environments
      // For SciWorld, let the SciWorldViewer component handle its own state refresh
      if (props.environmentId && currentEnvironmentClient.value) {
        await nextTick()
        // Skip for SciWorld to avoid duplicate API calls
        if (props.environmentType !== 'sciworld') {
          // Add a small delay to let child components initialize properly
          setTimeout(() => refreshState(), 200)
        } else {
          console.log('⏭️ EnvViewer: Skipping initial refresh for SciWorld component')
        }
      }
    })

    onUnmounted(() => {
      console.log('🧹 EnvViewer unmounted')
      stopAutoRun()
    })

    // Validate data index for reset
    const validateResetDataIdx = () => {
      // Ensure it's a number and not negative
      if (isNaN(resetDataIdx.value) || resetDataIdx.value === null) {
        resetDataIdx.value = 0;
      } else if (resetDataIdx.value < 0) {
        resetDataIdx.value = 0;
      }
    }

    return {
      props,
      loading,
      creating,
      error,
      envState,
      environmentConfig,
      connectionStatus,
      isAutoRunning,
      isExecutingAction,
      autoRunDelay,
      showResetOptions,
      resetDataIdx,
      canCreate,
      envConfig,
      debugMode,
      currentEnvironment,
      currentEnvironmentClient,
      currentEnvironmentConfigComponent,
      clearError,
      createEnvironment,
      refreshState,
      showResetDialog,
      cancelReset,
      confirmReset,
      toggleAutoRun,
      formatDate,
      onConfigReady,
      onActionExecuted,
      onGoalCompleted,
      onEnvironmentReset,
      onStateUpdated,
      onSuggestAction,
      onAutoActionSent,
      onAutoResponseReceived,
      onObservationUpdated,
      addAgentAction,
      textcraftViewerRef,
      validateResetDataIdx,
      stepAction,
      handleUserAction,
      onCriticalActionResult,
      onDirectResponse
    }
  }
}
</script>

<style scoped>
.env-viewer {
  padding: 1.5rem;
  height: 100%;
  overflow-y: auto;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

.debug-info {
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1rem;
  font-family: monospace;
  font-size: 0.9rem;
}

.debug-info h4 {
  margin: 0 0 0.5rem 0;
  color: #856404;
}

.debug-info div {
  margin-bottom: 0.25rem;
}

.debug-toggle {
  margin: 1rem 0;
  padding: 0.5rem;
  background: #f8f9fa;
  border-radius: 4px;
}

.debug-toggle label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
}

.connection-status {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  background: rgba(255, 255, 255, 0.9);
  padding: 1rem;
  border-radius: 12px;
  margin-bottom: 1.5rem;
  font-weight: 500;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  display: inline-block;
}

.status-connected {
  background: #4CAF50;
  box-shadow: 0 0 8px rgba(76, 175, 80, 0.4);
}

.status-disconnected {
  background: #f44336;
  box-shadow: 0 0 8px rgba(244, 67, 54, 0.4);
}

.env-type {
  margin-left: auto;
  font-size: 0.9rem;
  opacity: 0.8;
}

.error-message {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: linear-gradient(135deg, #ffebee, #ffcdd2);
  color: #c62828;
  border: 2px solid #ef5350;
  border-radius: 12px;
  padding: 1rem;
  margin-bottom: 1.5rem;
  font-weight: 500;
}

.clear-error-btn {
  background: none;
  border: none;
  color: #c62828;
  cursor: pointer;
  font-size: 1.2rem;
  padding: 0.25rem;
  border-radius: 4px;
}

.clear-error-btn:hover {
  background: rgba(198, 40, 40, 0.1);
}

.creation-card, .env-info-card, .control-panel {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 2rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}

.creation-card h3, .env-info-card h3 {
  margin: 0 0 1rem 0;
  color: #333;
}

.env-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.env-id {
  background: #e3f2fd;
  color: #1976d2;
  padding: 0.25rem 0.75rem;
  border-radius: 16px;
  font-size: 0.8rem;
  font-weight: 600;
}

.state-summary {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
}

.state-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.state-label {
  font-size: 0.8rem;
  color: #666;
  margin-bottom: 0.25rem;
}

.state-value {
  font-size: 1.2rem;
  font-weight: 600;
  color: #333;
}

.state-value.done {
  color: #4CAF50;
}

.create-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
  width: 100%;
  margin-top: 1rem;
}

.create-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
}

.create-btn:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
  transform: none;
}

.control-row {
  display: flex;
  gap: 1rem;
  align-items: center;
  flex-wrap: wrap;
}

.control-btn {
  padding: 0.6rem 1.2rem;
  border: 1px solid #ddd;
  border-radius: 8px;
  background: white;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s;
}

.control-btn:hover:not(:disabled) {
  background: #f8f9fa;
  border-color: #667eea;
  transform: translateY(-1px);
}

.reset-btn {
  border-color: #dc3545;
  color: #dc3545;
}

.reset-btn:hover:not(:disabled) {
  background: #dc3545;
  color: white;
}

.auto-btn {
  border-color: #28a745;
  color: #28a745;
}

.auto-btn.auto-active {
  background: #28a745;
  color: white;
}

.execution-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.8rem;
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 6px;
  font-size: 0.9rem;
  color: #856404;
}

.execution-indicator {
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.execution-text {
  font-weight: 500;
}

.delay-select {
  padding: 0.6rem;
  border: 1px solid #ddd;
  border-radius: 8px;
  background: white;
}

.dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.dialog-content {
  background: white;
  border-radius: 12px;
  width: 100%;
  max-width: 450px;
  overflow: hidden;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
  animation: dialogFadeIn 0.3s;
}

.dialog-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.25rem 1.5rem;
  border-bottom: 1px solid #e9ecef;
}

.dialog-header h3 {
  margin: 0;
  color: #333;
  font-weight: 600;
}

.close-dialog-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  line-height: 1;
  cursor: pointer;
  color: #666;
  padding: 0.25rem;
  border-radius: 4px;
}

.close-dialog-btn:hover {
  color: #333;
  background: #f0f0f0;
}

.dialog-body {
  padding: 1.5rem;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 1rem;
  padding: 1.25rem 1.5rem;
  border-top: 1px solid #e9ecef;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border: none;
  padding: 0.75rem 1.25rem;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
  transition: all 0.3s;
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
}

.btn-secondary {
  background: #f8f9fa;
  color: #333;
  border: 1px solid #ddd;
  padding: 0.75rem 1.25rem;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s;
}

.btn-secondary:hover {
  background: #e9ecef;
}

.reset-input-group {
  margin: 1.25rem 0;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.reset-input-group label {
  font-weight: 500;
  color: #555;
}

.reset-input-group input {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 1rem;
  outline: none;
}

.reset-input-group input:focus {
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
}

.reset-info {
  background: rgba(102, 126, 234, 0.1);
  padding: 0.75rem 1rem;
  border-radius: 8px;
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  font-size: 0.9rem;
  color: #555;
  line-height: 1.4;
}

.info-icon {
  flex-shrink: 0;
  margin-top: 0.1rem;
}

@keyframes dialogFadeIn {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.viewer-container {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}

.env-specific-viewer {
  padding: 1rem;
}

.no-viewer-message {
  padding: 2rem;
  text-align: center;
  color: #666;
  font-style: italic;
}

.loading-state {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 1rem;
  padding: 3rem;
  color: #667eea;
  font-weight: 600;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #e9ecef;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@media (max-width: 768px) {
  .env-viewer {
    padding: 1rem;
  }
  
  .state-summary {
    gap: 1rem;
  }
  
  .control-row {
    flex-direction: column;
    align-items: stretch;
  }
  
  .control-btn {
    width: 100%;
  }
}
</style>