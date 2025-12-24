<template>
  <div id="app">
    <router-view />
  </div>
</template>

<script>
import { ref, nextTick, onMounted } from 'vue'
import EnvViewer from './shared/components/EnvViewer.vue'
import EnvironmentSelector from './shared/components/EnvironmentSelector.vue'
import InteractionPanel from './shared/components/InteractionPanel.vue'
import SplitPane from './shared/components/SplitPane.vue'
import { getEnvironment, DEFAULT_ENVIRONMENT } from './environments/index.js'

export default {
  name: 'App',
  components: {
    EnvViewer,
    EnvironmentSelector,
    InteractionPanel,
    SplitPane
  },
  setup() {
    const environmentId = ref(null)
    const currentEnvironment = ref(null)
    const showEnvironmentSelector = ref(true)
    const isConnected = ref(false)
    
    const envViewer = ref(null)
    const interactionPanel = ref(null)
    
    const suggestedAction = ref('')
    const currentEnvironmentState = ref(null)

    onMounted(() => {
      currentEnvironment.value = getEnvironment(DEFAULT_ENVIRONMENT)
    })

    const toggleEnvironmentSelector = () => {
      showEnvironmentSelector.value = !showEnvironmentSelector.value
    }

    const onEnvironmentSelected = (envConfig) => {
      console.log('Environment selected:', envConfig)
      
      resetSessionState()
      
      currentEnvironment.value = envConfig
      showEnvironmentSelector.value = false
      
      nextTick(() => {
        clearInteractionHistory()
      })
    }

    const resetSessionState = () => {
      environmentId.value = null
      suggestedAction.value = ''
      currentEnvironmentState.value = null
      isConnected.value = false
    }

    const clearInteractionHistory = () => {
      if (interactionPanel.value?.clearHistory) {
        interactionPanel.value.clearHistory()
        console.log('Interaction history cleared')
      }
    }

    const onEnvironmentCreated = (id) => {
      console.log('🏗️ Environment creation event received:', id, typeof id)
      
      let numericId
      if (typeof id === 'object') {
        console.error('❌ Received object as environment ID:', id)
        if (id && id.id !== undefined) {
          numericId = parseInt(id.id)
        } else {
          console.error('❌ Cannot extract ID from object:', id)
          return
        }
      } else {
        numericId = parseInt(id)
      }
      
      if (isNaN(numericId)) {
        console.error('❌ Invalid environment ID:', id)
        return
      }
      
      console.log('✅ Setting environment ID to:', numericId)
      environmentId.value = numericId
      isConnected.value = true
      
      console.log('Environment created with ID:', numericId)
    }

    const onEnvironmentReset = (result) => {
      console.log('Environment reset:', result)
      
      nextTick(() => {
        clearInteractionHistory()
      })
    }

    const onStateUpdated = (state) => {
      console.log('State updated:', state)
      currentEnvironmentState.value = state
    }

    const onSuggestAction = (action) => {
      console.log('Action suggested:', action)
      suggestedAction.value = action
    }

    const onActionSent = (action) => {
      console.log('User action sent:', action)
    }

    const onResponseReceived = (response) => {
      console.log('Response received:', response)
      if (envViewer.value?.refreshState) {
        envViewer.value.refreshState()
      }
    }

    const onAutoActionSent = (action) => {
      console.log('Auto action sent:', action)
      if (interactionPanel.value?.addInteraction) {
        interactionPanel.value.addInteraction('action', `[Auto] ${action.action || action}`)
      }
    }

    const onAutoResponseReceived = (response) => {
      console.log('Auto response received:', response)
      
      const isCompletion = response?.result && 
        typeof response.result === 'string' && (
          response.result.includes('Auto run finished') || 
          response.result.includes('Goal completed') ||
          response.result.includes('Task Completed')
        )
      
      if (interactionPanel.value?.addInteraction) {
        interactionPanel.value.addInteraction('response', response, isCompletion)
      }
      
      if (envViewer.value?.refreshState) {
        envViewer.value.refreshState()
      }
    }

    return {
      environmentId,
      currentEnvironment,
      showEnvironmentSelector,
      isConnected,
      suggestedAction,
      currentEnvironmentState,
      
      envViewer,
      interactionPanel,
      
      toggleEnvironmentSelector,
      onEnvironmentSelected,
      onEnvironmentCreated,
      onEnvironmentReset,
      onStateUpdated,
      onSuggestAction,
      onActionSent,
      onResponseReceived,
      onAutoActionSent,
      onAutoResponseReceived
    }
  }
}
</script>

<style>
/* Global Styles */
:root {
  --primary-color: #667eea;
  --secondary-color: #764ba2;
  --success-color: #4CAF50;
  --warning-color: #ff9800;
  --danger-color: #f44336;
  --text-color: #333333;
  --background-color: #f5f7fa;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: var(--text-color);
  background-color: var(--background-color);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#app {
  min-height: 100vh;
}

a {
  text-decoration: none;
  color: var(--primary-color);
}

button {
  cursor: pointer;
}

button:disabled {
  cursor: not-allowed;
}

/* Utility Classes */
.text-center {
  text-align: center;
}

.mb-1 {
  margin-bottom: 0.5rem;
}

.mb-2 {
  margin-bottom: 1rem;
}

.mb-3 {
  margin-bottom: 1.5rem;
}

/* Responsive Design */
@media (max-width: 768px) {
  .hide-mobile {
    display: none;
  }
}

/* Transitions */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>