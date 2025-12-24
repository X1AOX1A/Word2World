<template>
  <div class="action-handler">
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner"></div>
      <div>Processing action...</div>
    </div>
    
    <div class="action-input-container">
      <!-- Available Actions Section -->
      <div class="action-suggestions" v-if="availableActions.length > 0">
        <h4>Available Actions:</h4>
        <div class="action-buttons">
          <button 
            v-for="action in availableActions" 
            :key="action"
            @click="selectAction(action)"
            class="action-button"
            :disabled="isDone"
          >
            {{ action }}
          </button>
        </div>
      </div>
      
      <!-- Action Input Form -->
      <div class="action-form">
        <div class="input-group">
          <input 
            type="text" 
            v-model="actionInput"
            @keyup.enter="sendAction"
            placeholder="Enter an action..."
            class="action-input"
            :disabled="loading || isDone"
            ref="actionInputField"
          />
          <button 
            @click="sendAction" 
            class="send-button"
            :disabled="loading || !actionInput.trim() || isDone"
          >
            Send
          </button>
        </div>
      </div>
      
      <!-- Completed State Banner -->
      <div v-if="isDone" class="completion-banner">
        <div class="completion-icon">✅</div>
        <div>Task completed successfully!</div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'BabyAIActionHandler',
  props: {
    environmentId: {
      type: [String, Number],
      default: null
    },
    client: {
      type: Object,
      default: null
    },
    currentState: {
      type: Object,
      default: () => null
    }
  },
  data() {
    return {
      actionInput: '',
      loading: false,
      availableActions: [],
      isDone: false
    }
  },
  watch: {
    currentState: {
      immediate: true,
      deep: true,
      handler(newState) {
        if (newState) {
          // Update done state
          this.isDone = newState.done || false;
          
          // Extract available actions if not done
          if (!this.isDone && newState.observation) {
            this.extractAvailableActions(newState.observation);
          } else if (this.isDone) {
            // Clear available actions if done
            this.availableActions = [];
          }
        }
      }
    }
  },
  methods: {
    selectAction(action) {
      this.actionInput = action;
      this.$nextTick(() => {
        this.$refs.actionInputField.focus();
      });
    },
    
    extractAvailableActions(observation) {
      // Extract available actions from the observation
      if (typeof observation !== 'string') return;
      
      const actionMatch = observation.match(/Available actions:\s*\[(.*?)\]/i);
      if (actionMatch && actionMatch[1]) {
        // Parse the actions from the string format
        try {
          const actionsStr = actionMatch[1];
          const actions = actionsStr.split(',').map(action => {
            return action.trim().replace(/"/g, '');
          }).filter(action => action);
          
          this.availableActions = actions;
        } catch (error) {
          console.error('Error parsing available actions:', error);
          this.availableActions = [];
        }
      } else {
        this.availableActions = [];
      }
    },
    
    async sendAction() {
      if (!this.actionInput.trim() || !this.client || !this.environmentId || this.loading || this.isDone) {
        return;
      }
      
      const action = this.actionInput.trim();
      this.loading = true;
      
      try {
        // Send the action to the server
        const result = await this.client.step(this.environmentId, action);
        
        if (result && result.success) {
          // Update done state
          this.isDone = result.data.done || false;
          
          // Clear input
          this.actionInput = '';
          
          // Emit the updated state
          this.$emit('action-completed', result.data);
        } else {
          console.error('Action failed:', result?.error);
        }
      } catch (error) {
        console.error('Error sending action:', error);
      } finally {
        this.loading = false;
      }
    },
    
    clearHistory() {
      // Just clear input and reset state
      this.actionInput = '';
      this.isDone = false;
    },
    
    handleReset(resetMessage = null) {
      // Reset component state
      this.actionInput = '';
      this.isDone = false;
      
      // Indicate this was handled
      return true;
    }
  }
}
</script>

<style scoped>
.action-handler {
  height: 100%;
  display: flex;
  flex-direction: column;
  position: relative;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.8);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 10;
  color: #2196F3;
  font-weight: 600;
}

.loading-spinner {
  width: 30px;
  height: 30px;
  border: 3px solid #e9ecef;
  border-top: 3px solid #2196F3;
  border-radius: 50%;
  margin-bottom: 0.5rem;
  animation: spin 1s linear infinite;
}

.action-input-container {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.action-suggestions {
  background: white;
  border-radius: 8px;
  padding: 1rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  margin-bottom: 1rem;
}

.action-suggestions h4 {
  font-size: 1rem;
  margin-bottom: 0.75rem;
  color: #333;
}

.action-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.action-button {
  background: #e3f2fd;
  color: #0277bd;
  border: none;
  border-radius: 4px;
  padding: 0.5rem 0.75rem;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.action-button:hover:not([disabled]) {
  background: #bbdefb;
}

.action-button[disabled] {
  opacity: 0.6;
  cursor: not-allowed;
}

.action-form {
  margin-bottom: 1rem;
}

.input-group {
  display: flex;
}

.action-input {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-right: none;
  border-radius: 4px 0 0 4px;
  font-size: 1rem;
}

.action-input[disabled] {
  background-color: #f5f5f5;
  cursor: not-allowed;
}

.send-button {
  background: #2196F3;
  color: white;
  border: none;
  border-radius: 0 4px 4px 0;
  padding: 0 1.25rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s ease;
}

.send-button:hover:not([disabled]) {
  background: #1976D2;
}

.send-button:disabled {
  background: #90CAF9;
  cursor: not-allowed;
}

.completion-banner {
  background: #e8f5e9;
  color: #2e7d32;
  border-radius: 8px;
  padding: 1rem;
  text-align: center;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.completion-icon {
  font-size: 1.5rem;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style> 