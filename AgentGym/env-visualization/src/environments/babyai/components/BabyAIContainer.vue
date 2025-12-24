<template>
  <div class="babyai-container">
    <!-- Content Grid -->
    <div class="content-grid">
      <div class="viewer-panel">
        <BabyAIViewer
          :environment-id="environmentId"
          :client="client"
          :env-state="currentState"
          @state-updated="onViewerStateUpdated"
          ref="viewer"
        />
      </div>
      
      <div class="action-panel">
        <BabyAIActionHandler
          :environment-id="environmentId"
          :client="client"
          :current-state="currentState"
          @action-completed="onActionCompleted"
          ref="actionHandler"
        />
      </div>
    </div>
    
    <!-- Controls Bar -->
    <div class="control-bar">
      <div class="control-left">
        <button 
          @click="resetEnvironment" 
          class="control-button reset"
          :disabled="loading || !environmentId"
        >
          🔄 Reset
        </button>
      </div>
      
      <div class="status-indicator" :class="{ active: isActive, completed: isDone }">
        {{ statusText }}
      </div>
    </div>
    
    <!-- Loading Overlay -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner"></div>
      <div>{{ loadingMessage }}</div>
    </div>
  </div>
</template>

<script>
import BabyAIViewer from './BabyAIViewer.vue';
import BabyAIActionHandler from './BabyAIActionHandler.vue';

export default {
  name: 'BabyAIContainer',
  components: {
    BabyAIViewer,
    BabyAIActionHandler
  },
  props: {
    environmentId: {
      type: [String, Number],
      default: null
    },
    client: {
      type: Object,
      default: null
    }
  },
  emits: ['action-completed', 'state-updated'],
  data() {
    return {
      currentState: null,
      loading: false,
      loadingMessage: 'Loading...',
      isActive: false,
      isDone: false,
      isResetting: false // Flag to prevent duplicate reset operations
    }
  },
  computed: {
    statusText() {
      if (this.isDone) return 'Completed';
      if (this.isActive) return 'Active';
      return 'Inactive';
    }
  },
  watch: {
    environmentId: {
      immediate: true,
      handler(newId) {
        if (newId) {
          // When refreshing the environment, always fetch the latest state
          // The environment should be already reset by the client's createEnvironment
          this.refreshEnvironment();
        }
      }
    }
  },
  methods: {
    refreshEnvironment() {
      // When refreshing the environment, always fetch the latest state
      this.fetchCurrentState();
    },
    
    async fetchCurrentState() {
      if (!this.client || !this.environmentId || this.isResetting) return;
      
      this.setLoading(true, 'Loading environment state...');
      
      try {
        const result = await this.client.getObservation(this.environmentId);
        
        if (result && result.success) {
          this.currentState = result.data;
          this.isActive = true;
          this.isDone = result.data.done || false;
        } else {
          console.error('Failed to fetch BabyAI state:', result?.error);
        }
      } catch (error) {
        console.error('Error fetching BabyAI state:', error);
      } finally {
        this.setLoading(false);
      }
    },
    
    async resetEnvironment() {
      if (!this.client || !this.environmentId || this.isResetting || this.loading) return;
      
      // Set resetting flag to prevent duplicate resets
      this.isResetting = true;
      this.setLoading(true, 'Resetting environment...');
      
      try {
        // Reset with default data index (0)
        const result = await this.client.reset(this.environmentId, 0);
        
        if (result && result.success) {
          console.log('Reset successful, updating state with:', result.data);
          
          // Update current state
          this.currentState = result.data;
          
          // Reset done state
          this.isDone = false;
          
          // Clear action handler state
          if (this.$refs.actionHandler) {
            this.$refs.actionHandler.clearHistory();
          }
          
          // Refresh the viewer to update with new state
          if (this.$refs.viewer) {
            this.$refs.viewer.refreshState();
            this.$refs.viewer.fetchRenderImage();
          }
          
          this.isActive = true;
          
          // Emit state update for global handling
          this.$emit('state-updated', result.data);
          
          console.log('BabyAI environment reset successfully');
        } else {
          console.error('Failed to reset BabyAI environment:', result?.error);
        }
      } catch (error) {
        console.error('Error resetting BabyAI environment:', error);
      } finally {
        this.setLoading(false);
        
        // Reset the flag after a short delay to prevent rapid clicking
        setTimeout(() => {
          this.isResetting = false;
        }, 500);
      }
    },
    
    onViewerStateUpdated(state) {
      // Forward state updates to parent components for global handling
      this.$emit('state-updated', state);
    },
    
    onActionCompleted(resultData) {
      // Update the current state and done status
      this.currentState = resultData;
      this.isDone = resultData.done || false;
      
      // If done, ensure viewer is updated (especially if the render image should be hidden)
      if (this.isDone && this.$refs.viewer) {
        this.$refs.viewer.isDone = true;
      }
      
      // Forward the action result to parent for global handling
      this.$emit('action-completed', resultData);
    },
    
    setLoading(isLoading, message = 'Loading...') {
      this.loading = isLoading;
      this.loadingMessage = message;
    }
  },
  mounted() {
    if (this.environmentId && this.client) {
      // When the component mounts, ensure the environment is properly initialized
      this.refreshEnvironment();
    }
  }
}
</script>

<style scoped>
.babyai-container {
  height: 100%;
  display: flex;
  flex-direction: column;
  position: relative;
}

.content-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  height: calc(100% - 60px); /* Adjust for control bar height */
  overflow: hidden;
}

.viewer-panel, .action-panel {
  background: #f9f9f9;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  height: 100%;
}

.control-bar {
  height: 60px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 1rem;
  background: white;
  border-radius: 8px;
  margin-top: 1rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.control-left, .control-right {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.control-button {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 4px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.control-button.reset {
  background: #ff9800;
  color: white;
}

.control-button.reset:hover {
  background: #f57c00;
}

.status-indicator {
  font-weight: 600;
  color: #f44336;
  padding: 0.25rem 0.75rem;
  border-radius: 4px;
  background: #ffebee;
}

.status-indicator.active {
  color: #4caf50;
  background: #e8f5e9;
}

.status-indicator.completed {
  color: #ff9800;
  background: #fff3e0;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.9);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 100;
  color: #2196F3;
  font-weight: 600;
}

.loading-spinner {
  width: 50px;
  height: 50px;
  border: 5px solid #e9ecef;
  border-top: 5px solid #2196F3;
  border-radius: 50%;
  margin-bottom: 1rem;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@media (max-width: 768px) {
  .content-grid {
    grid-template-columns: 1fr;
    grid-template-rows: 1fr 1fr;
  }
}
</style> 