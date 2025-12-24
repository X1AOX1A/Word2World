<template>
  <div class="searchqa-viewer">
    <div class="content-container">
      <!-- Status and Info Panel -->
      <div class="main-column">
        <!-- Question Panel -->
        <div class="panel question-panel">
          <h3 class="panel-title">
            <span class="panel-icon">❓</span> Question
          </h3>
          <div class="question-content">
            {{ currentQuestion || 'No question available' }}
          </div>
        </div>

        <!-- Status Panel -->
        <div class="panel status-panel">
          <h3 class="panel-title">
            <span class="panel-icon">📊</span> Status
          </h3>
          <div class="status-grid">
            <div class="status-item">
              <div class="status-label">Rounds</div>
              <div class="status-value" :class="{ 'warning': currentRound >= 4, 'danger': currentRound >= 5 }">
                {{ currentRound }} / 5
              </div>
            </div>
            <div class="status-item">
              <div class="status-label">State</div>
              <div class="status-value" :class="getStatusClass()">
                {{ getStatusText() }}
              </div>
            </div>
            <div class="status-item">
              <div class="status-label">Score</div>
              <div class="status-value" :class="{ 'success': totalReward > 0 }">
                {{ totalReward.toFixed(2) }}
              </div>
            </div>
          </div>
        </div>

        <!-- Search History Panel -->
        <div class="panel history-panel">
          <h3 class="panel-title">
            <span class="panel-icon">🔍</span> Search History
            <button @click="clearHistory" class="clear-history-btn" :disabled="searchHistory.length === 0">
              <span class="clear-icon">🗑️</span>
            </button>
          </h3>
          
          <div v-if="searchHistory.length > 0" class="history-container">
            <div v-for="(item, index) in searchHistory" :key="index" class="history-item" :class="item.type">
              <div class="history-header">
                <span class="history-type">{{ getHistoryTypeIcon(item.type) }} {{ item.type.toUpperCase() }}</span>
                <span class="history-time">{{ formatTime(item.timestamp) }}</span>
              </div>
              <div class="history-content">
                <div v-if="item.type === 'search'" class="search-query">
                  <strong>Query:</strong> {{ item.query }}
                </div>
                <div v-if="item.type === 'search' && item.results" class="search-results">
                  <strong>Results:</strong>
                  <div class="search-result" v-html="formatSearchResult(item.results)"></div>
                </div>
                <div v-if="item.type === 'answer'" class="answer-content">
                  <strong>Answer:</strong> {{ item.answer }}
                  <div class="answer-result" :class="{ 'correct': item.correct, 'incorrect': !item.correct }">
                    {{ item.correct ? '✅ Correct!' : '❌ Incorrect' }}
                  </div>
                </div>
                <div v-if="item.type === 'invalid'" class="invalid-action">
                  <strong>Invalid Action:</strong> {{ item.message }}
                </div>
              </div>
            </div>
          </div>
          
          <div v-else class="empty-history">
            <span class="empty-icon">📝</span>
            <span>No actions taken yet</span>
          </div>
        </div>

        <!-- Current Information Panel -->
        <div v-if="currentSearchResults" class="panel info-panel">
          <h3 class="panel-title">
            <span class="panel-icon">ℹ️</span> Current Information
          </h3>
          <div class="info-content" v-html="formatSearchResult(currentSearchResults)"></div>
        </div>
      </div>
    </div>
    
    <!-- Loading State -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner"></div>
      <div>Processing...</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'SearchQAViewer',
  props: {
    environmentId: {
      type: [String, Number],
      default: null
    },
    client: {
      type: Object,
      default: null
    },
    envState: {
      type: Object,
      default: () => null
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
    'action-executed',
    'question-answered',
    'state-updated',
    'environment-reset'
  ],
  data() {
    return {
      loading: false,
      currentQuestion: '',
      currentRound: 0,
      totalReward: 0,
      searchHistory: [],
      currentSearchResults: null,
      isCompleted: false,
      lastAction: null,
      
      // Lock mechanism to prevent multiple simultaneous operations
      isOperationLocked: false,
      operationLockTimeout: null,

      debug: true
    }
  },
  watch: {
    // Watch for environment ID changes
    environmentId(newId, oldId) {
      if (newId && newId !== oldId) {
        console.log(`📌 SearchQAViewer: Environment ID changed from ${oldId} to ${newId}`);
        this.resetState();
        this.initializeEnvironment();
      }
    },
    
    // Watch for external state updates
    envState: {
      handler(newState) {
        if (newState && newState.observation) {
          console.log('📌 SearchQAViewer: Received new state data from parent');
          
          // Check if this is a reset operation
          const isReset = newState.steps === 0;
          
          if (isReset) {
            console.log('🔄 SearchQAViewer: Detected environment reset');
            this.resetState();
          }
          
          // Process state data
          this.processStateData(newState);
        }
      },
      deep: true
    },
    lastAutoAction: {
      handler(newAction) {
        if (newAction && newAction.timestamp) {
          console.log('🎯 SearchQAViewer: Received auto-action via props:', newAction);
          this.handleAutoActionSent(newAction);
        }
      },
      deep: true
    },
    lastAutoResponse: {
      handler(newResponse) {
        if (newResponse && newResponse.timestamp) {
          console.log('🎯 SearchQAViewer: Received auto-response via props:', newResponse);
          this.handleAutoResponseReceived(newResponse);
        }
      },
      deep: true
    }
  },
  methods: {
    // Initialize environment on first load
    async initializeEnvironment() {
      if (!this.environmentId || !this.client) {
        console.log('⚠️ Cannot initialize: Missing environment ID or client');
        return;
      }
      
      if (this.loading || this.isOperationLocked) {
        console.log('⚠️ Skipping initialization - already loading or locked');
        return;
      }
      
      try {
        this.lockOperation('Initializing environment');
        this.loading = true;
        
        // Get the current observation to extract the question
        const stateResult = await this.client.getObservation(this.environmentId);
        if (stateResult && stateResult.success) {
          console.log('📋 Initial observation result:', stateResult);
          
          // Try to extract question from the response
          if (stateResult.data) {
            // First try the structured data approach
            this.processStateData(stateResult);
            
            // If still no question, try extracting from raw text
            if (!this.currentQuestion) {
              if (typeof stateResult.data === 'string') {
                this.extractQuestionFromText(stateResult.data);
              } else if (stateResult.data.raw) {
                this.extractQuestionFromText(stateResult.data.raw);
              } else if (stateResult.data.question) {
                this.currentQuestion = stateResult.data.question;
                console.log('📝 Question extracted from data.question:', this.currentQuestion);
              }
            }
          }
        }
      } catch (error) {
        console.error('❌ Error initializing SearchQA environment:', error);
      } finally {
        this.loading = false;
        this.unlockOperation();
      }
    },
    
    // Reset internal state when environment changes
    resetState() {
      this.loading = false;
      this.currentQuestion = '';
      this.currentRound = 0;
      this.totalReward = 0;
      this.searchHistory = [];
      this.currentSearchResults = null;
      this.isCompleted = false;
      this.lastAction = null;
      this.unlockOperation();
    },
    
    // Process state data to extract question, results, etc.
    processStateData(stateData) {
      if (!stateData || !stateData.data) return;
      
      const data = stateData.data;
      
      // Extract question from observation
      if (data.observation) {
        this.parseObservation(data.observation);
      }
      
      // Also try to extract question directly from raw observation text
      if (!this.currentQuestion && typeof data === 'string') {
        this.extractQuestionFromText(data);
      } else if (!this.currentQuestion && data.raw) {
        this.extractQuestionFromText(data.raw);
      }
      
      // Update reward
      if (data.reward !== undefined) {
        this.totalReward = data.reward;
      }
      
      // Check if completed
      if (data.done !== undefined) {
        this.isCompleted = data.done;
      }
      
      // Track rounds based on history length
      this.currentRound = this.searchHistory.length;
      
      console.log('📊 SearchQA State:', {
        question: this.currentQuestion,
        round: this.currentRound,
        reward: this.totalReward,
        completed: this.isCompleted
      });
    },
    
    // Extract question from text content
    extractQuestionFromText(text) {
      if (!text || typeof text !== 'string') return;
      
      // Look for "Question: ..." pattern at the end of the text
      const questionMatch = text.match(/Question:\s*(.+?)(?:\s*$)/s);
      if (questionMatch && questionMatch[1]) {
        this.currentQuestion = questionMatch[1].trim();
        console.log('📝 Question extracted from text:', this.currentQuestion);
        return;
      }
      
      // Alternative pattern after "Follow this process every time"
      const altQuestionMatch = text.match(/Follow this process every time\.\s*\n\s*Question:\s*(.+?)(?:\s*$)/s);
      if (altQuestionMatch && altQuestionMatch[1]) {
        this.currentQuestion = altQuestionMatch[1].trim();
        console.log('📝 Question extracted from alternative pattern:', this.currentQuestion);
        return;
      }
      
      console.log('⚠️ No question found in text:', text.substring(0, 200) + '...');
    },
    
    // Parse observation to extract different types of content
    parseObservation(observation) {
      if (!observation) return;
      
      let observationData = observation;
      
      // If observation is a string, first try to parse it as structured data
      if (typeof observation === 'string') {
        observationData = observation;
      } else if (observation.raw) {
        // If it's already processed observation data from client
        this.handleProcessedObservation(observation);
        return;
      }
      
      // Try to parse using client's processObservationData if available
      if (this.client && this.client.processObservationData) {
        const processedData = this.client.processObservationData(observationData);
        this.handleProcessedObservation(processedData);
      } else {
        // Fallback to manual parsing
        this.fallbackParseObservation(observationData);
      }
    },
    
    // Handle processed observation data from client
    handleProcessedObservation(processedData) {
      // Update question if available
      if (processedData.question && processedData.question !== this.currentQuestion) {
        this.currentQuestion = processedData.question;
        console.log('📝 Question updated:', this.currentQuestion);
      }
      
      // If no question found in processed data, try extracting from raw text
      if (!this.currentQuestion && processedData.raw) {
        this.extractQuestionFromText(processedData.raw);
      }
      
      // Get the last action from client if available
      let lastAction = null;
      if (this.client && this.client.getLastAction && this.environmentId) {
        lastAction = this.client.getLastAction(this.environmentId);
      }
      
      // Handle different observation types
      switch (processedData.type) {
        case 'question':
          // Initial question, reset state if needed
          this.currentSearchResults = null;
          break;
          
        case 'search_result':
          // New search results received
          this.currentSearchResults = processedData.searchResults;
          
          // Add to search history if we have a recent search action
          if (lastAction && lastAction.type === 'search') {
            this.addToHistory('search', {
              query: lastAction.content,
              results: processedData.searchResults
            });
          }
          break;
          
        case 'answer_feedback':
          // Answer feedback received
          if (lastAction && lastAction.type === 'answer') {
            this.addToHistory('answer', {
              answer: lastAction.content,
              correct: processedData.feedback === 'correct'
            });
            
            if (processedData.feedback === 'correct') {
              this.isCompleted = true;
            }
          }
          break;
          
        case 'error_feedback':
          // Invalid action feedback
          this.addToHistory('invalid', {
            message: processedData.raw,
            action: lastAction ? lastAction.raw : 'Unknown action'
          });
          break;
      }
    },
    
    // Fallback parsing for when client methods aren't available
    fallbackParseObservation(content) {
      if (typeof content === 'object') {
        content = JSON.stringify(content);
      }
      
      // Get the last action from fallback storage
      let lastAction = this.lastAction;
      
      // Extract question
      const questionMatch = content.match(/Question:\s*(.+?)(?:\s*$)/s);
      if (questionMatch && questionMatch[1].trim() !== this.currentQuestion) {
        this.currentQuestion = questionMatch[1].trim();
      }
      
      // Check for search results
      const infoMatch = content.match(/<information>(.*?)<\/information>/s);
      if (infoMatch) {
        this.currentSearchResults = infoMatch[1].trim();
        
        if (lastAction && lastAction.type === 'search') {
          this.addToHistory('search', {
            query: lastAction.content,
            results: this.currentSearchResults
          });
          this.lastAction = null;
        }
      }
      
      // Check for answer feedback
      if (content.includes('Congratulations! You have answered the question correctly')) {
        if (lastAction && lastAction.type === 'answer') {
          this.addToHistory('answer', {
            answer: lastAction.content,
            correct: true
          });
          this.lastAction = null;
        }
        this.isCompleted = true;
      } else if (content.includes('Sorry, your answer is incorrect')) {
        if (lastAction && lastAction.type === 'answer') {
          this.addToHistory('answer', {
            answer: lastAction.content,
            correct: false
          });
          this.lastAction = null;
        }
      }
      
      // Check for invalid action feedback
      if (content.includes('Your previous action is invalid')) {
        this.addToHistory('invalid', {
          message: content,
          action: lastAction ? lastAction.raw : 'Unknown action'
        });
        this.lastAction = null;
      }
    },
    
    // Add item to search history
    addToHistory(type, data) {
      const historyItem = {
        type,
        timestamp: new Date(),
        ...data
      };
      
      this.searchHistory.push(historyItem);
      this.currentRound = this.searchHistory.filter(item => item.type === 'search').length;
      
      // Keep history to a reasonable size
      if (this.searchHistory.length > 20) {
        this.searchHistory.shift();
      }
    },
    
    // Format search results for display
    formatSearchResult(result) {
      if (!result) return '';
      
      // Convert newlines to <br> and preserve formatting
      return result
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
    },
    
    // Get status class based on current state
    getStatusClass() {
      if (this.isCompleted) return 'success';
      if (this.currentRound >= 5) return 'danger';
      if (this.currentRound >= 4) return 'warning';
      return 'active';
    },
    
    // Get status text
    getStatusText() {
      if (this.isCompleted) return 'Completed';
      if (this.currentRound >= 5) return 'Max Rounds';
      if (this.currentRound === 0) return 'Ready';
      return 'Active';
    },
    
    // Get history type icon
    getHistoryTypeIcon(type) {
      switch (type) {
        case 'search': return '🔍';
        case 'answer': return '💭';
        case 'invalid': return '⚠️';
        default: return '📝';
      }
    },
    
    // Format timestamp
    formatTime(timestamp) {
      return new Date(timestamp).toLocaleTimeString();
    },
    
    // Clear search history
    clearHistory() {
      this.searchHistory = [];
      this.currentRound = 0;
      this.currentSearchResults = null;
    },
    
    // Lock operation to prevent conflicts
    lockOperation(reason) {
      console.log(`🔒 Locking operations: ${reason}`);
      this.isOperationLocked = true;
      
      // Set a timeout to automatically unlock after 30 seconds
      this.operationLockTimeout = setTimeout(() => {
        console.log('⏰ Auto-unlocking operations after timeout');
        this.unlockOperation();
      }, 30000);
    },
    
    // Unlock operation
    unlockOperation() {
      if (this.operationLockTimeout) {
        clearTimeout(this.operationLockTimeout);
        this.operationLockTimeout = null;
      }
      
      if (this.isOperationLocked) {
        console.log('🔓 Unlocking operations');
        this.isOperationLocked = false;
      }
    },
    
    // Handle auto-action sent event
    handleAutoActionSent(data) {
      console.log('🎯 SearchQAViewer: Received auto-action-sent:', data);
      
      if (data && data.action) {
        // Parse the action and store it for correlation with response
        const parsedAction = this.client?.parseAction?.(data.action) || this.parseActionFallback(data.action);
        this.lastAction = parsedAction;
        
        console.log('📝 SearchQAViewer: Tracked auto action:', parsedAction);
        
        // For search actions, we'll wait for the response to add to history
        // For answer actions, we might get immediate feedback
        if (parsedAction.type === 'answer') {
          // Add answer to history immediately, we'll update with result when response comes
          this.addToHistory('answer', {
            answer: parsedAction.content,
            correct: null // Will be updated when response arrives
          });
        }
      }
    },
    
    // Handle auto-response received event
    handleAutoResponseReceived(data) {
      console.log('🎯 SearchQAViewer: Received auto-response-received:', data);
      
      if (data && data.observation) {
        // Process the response observation
        if (this.client && this.client.processObservationData) {
          const processedData = this.client.processObservationData(data.observation);
          this.handleProcessedObservation(processedData);
        } else {
          this.fallbackParseObservation(data.observation);
        }
        
        // Update states
        if (data.reward !== undefined) {
          this.totalReward = data.reward;
        }
        if (data.done !== undefined) {
          this.isCompleted = data.done;
        }
      }
    },
    
    // Fallback action parsing
    parseActionFallback(action) {
      const searchMatch = action.match(/<search>(.*?)<\/search>/s);
      if (searchMatch) {
        return {
          type: 'search',
          content: searchMatch[1].trim(),
          raw: action
        };
      }
      
      const answerMatch = action.match(/<answer>(.*?)<\/answer>/s);
      if (answerMatch) {
        return {
          type: 'answer',
          content: answerMatch[1].trim(),
          raw: action
        };
      }
      
      const thinkMatch = action.match(/<think>(.*?)<\/think>/s);
      if (thinkMatch) {
        return {
          type: 'think',
          content: thinkMatch[1].trim(),
          raw: action
        };
      }
      
      return {
        type: 'invalid',
        content: action,
        raw: action
      };
    }
  },
  mounted() {
    console.log('📌 SearchQAViewer mounted, environment ID:', this.environmentId);
    if (this.environmentId) {
      this.initializeEnvironment();
    }
  },
  beforeUnmount() {
    console.log('📌 SearchQAViewer unmounting, environment ID:', this.environmentId);
    // Clean up any pending operations
    if (this.operationLockTimeout) {
      clearTimeout(this.operationLockTimeout);
    }
  }
}
</script>

<style scoped>
.searchqa-viewer {
  padding: 1.5rem;
  height: 100%;
  width: 100%;
  overflow-y: auto;
  position: relative;
  font-family: 'Nunito', 'Poppins', sans-serif;
  display: flex;
  flex-direction: column;
  background-color: #f8f9fa;
}

/* Content container */
.content-container {
  display: flex;
  gap: 1.5rem;
  flex: 1;
  min-height: 0;
}

.main-column {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  flex: 1;
}

/* Panel styles */
.panel {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  border: 1px solid #e9ecef;
}

.panel-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 0 0 1rem 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: #343a40;
  border-bottom: 2px solid #FF5722;
  padding-bottom: 0.5rem;
}

.panel-icon {
  margin-right: 0.5rem;
}

/* Question panel */
.question-panel {
  border-left: 4px solid #FF5722;
}

.question-content {
  font-size: 1.1rem;
  line-height: 1.6;
  color: #495057;
  background: #fff3e0;
  padding: 1rem;
  border-radius: 8px;
  border-left: 4px solid #FF5722;
  font-weight: 500;
}

/* Status panel */
.status-panel {
  border-left: 4px solid #28a745;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 1rem;
}

.status-item {
  text-align: center;
  padding: 0.75rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.status-label {
  font-size: 0.9rem;
  color: #6c757d;
  margin-bottom: 0.25rem;
}

.status-value {
  font-size: 1.2rem;
  font-weight: 600;
  color: #495057;
}

.status-value.warning {
  color: #ffc107;
}

.status-value.danger {
  color: #dc3545;
}

.status-value.success {
  color: #28a745;
}

.status-value.active {
  color: #007bff;
}

/* History panel */
.history-panel {
  border-left: 4px solid #007bff;
  flex: 1;
  min-height: 400px;
}

.clear-history-btn {
  background: none;
  border: none;
  color: #dc3545;
  cursor: pointer;
  padding: 0.25rem;
  border-radius: 4px;
  transition: background-color 0.2s;
}

.clear-history-btn:hover:not(:disabled) {
  background-color: #f8d7da;
}

.clear-history-btn:disabled {
  color: #6c757d;
  cursor: not-allowed;
}

.history-container {
  max-height: 500px;
  overflow-y: auto;
}

.history-item {
  margin-bottom: 1rem;
  padding: 1rem;
  border-radius: 8px;
  border-left: 4px solid #dee2e6;
}

.history-item.search {
  border-left-color: #007bff;
  background: #f8f9fa;
}

.history-item.answer {
  border-left-color: #28a745;
  background: #f0f8f0;
}

.history-item.invalid {
  border-left-color: #dc3545;
  background: #fdf2f2;
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.history-type {
  font-weight: 600;
  font-size: 0.9rem;
}

.history-time {
  font-size: 0.8rem;
  color: #6c757d;
}

.history-content {
  font-size: 0.9rem;
  line-height: 1.5;
}

.search-query, .answer-content, .invalid-action {
  margin-bottom: 0.5rem;
}

.search-result {
  background: white;
  padding: 0.75rem;
  border-radius: 6px;
  margin-top: 0.5rem;
  border: 1px solid #dee2e6;
  font-size: 0.85rem;
  line-height: 1.4;
}

.answer-result {
  margin-top: 0.5rem;
  padding: 0.5rem;
  border-radius: 6px;
  font-weight: 600;
}

.answer-result.correct {
  background: #d4edda;
  color: #155724;
}

.answer-result.incorrect {
  background: #f8d7da;
  color: #721c24;
}

.empty-history {
  text-align: center;
  color: #6c757d;
  padding: 2rem 0;
  font-style: italic;
}

.empty-icon {
  display: block;
  font-size: 2rem;
  margin-bottom: 0.5rem;
}

/* Info panel */
.info-panel {
  border-left: 4px solid #17a2b8;
  min-height: 200px;
}

.info-content {
  background: #e7f3ff;
  padding: 1rem;
  border-radius: 8px;
  border: 1px solid #b3d9ff;
  font-size: 0.9rem;
  line-height: 1.5;
}

/* Loading state */
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
  z-index: 10;
  color: #FF5722;
  font-weight: 600;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #e9ecef;
  border-top: 4px solid #FF5722;
  border-radius: 50%;
  margin-bottom: 1rem;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Responsive design */
@media (max-width: 768px) {
  .searchqa-viewer {
    padding: 1rem;
  }
  
  .content-container {
    flex-direction: column;
  }
  
  .status-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
