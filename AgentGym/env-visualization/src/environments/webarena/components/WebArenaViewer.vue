<template>
  <div class="webarena-viewer">
    <!-- Task Section -->
    <div class="task-section">
      <h3>Current Task</h3>
      <div class="task-text">{{ task || 'No task specified' }}</div>
    </div>
    
    <!-- Browser Screenshot -->
    <div class="browser-section">
      <h3>Browser View</h3>
      <div class="screenshot-container">
        <img v-if="screenshot" :src="screenshotSource" alt="WebArena Browser Screenshot" />
        <div v-else-if="isDone" class="completed-message">
          <div class="completion-icon">✅</div>
          <div class="completion-text">Task completed!</div>
        </div>
        <div v-else class="no-screenshot-message">
          <div class="loading-spinner"></div>
          <div>Loading browser view...</div>
        </div>
      </div>
      
      <!-- URL Display -->
      <div v-if="currentUrl" class="url-display">
        <span class="url-label">URL:</span>
        <span class="url-value">{{ currentUrl }}</span>
      </div>
    </div>
    
    <!-- Loading State -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner"></div>
      <div>Loading...</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'WebArenaViewer',
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
    }
  },
  data() {
    return {
      loading: false,
      currentState: null,
      screenshot: null,
      task: 'Navigate and interact with web pages',
      currentUrl: '',
      isDone: false,
      
      // Deduplication tracking similar to BabyAI
      lastRefreshId: null,
      lastScreenshotId: null,
      isRefreshing: false,
      isScreenshotting: false
    }
  },
  computed: {
    screenshotSource() {
      if (!this.screenshot) return '';
      
      // If it's already a complete data URL, use it directly
      if (typeof this.screenshot === 'string' && this.screenshot.startsWith('data:image/')) {
        return this.screenshot;
      }
      
      // If it's base64 data, create PNG data URL
      if (typeof this.screenshot === 'string' && /^[A-Za-z0-9+/=]+$/.test(this.screenshot)) {
        return `data:image/png;base64,${this.screenshot}`;
      }
      
      // For binary data arrays
      if (Array.isArray(this.screenshot) || this.screenshot instanceof Uint8Array) {
        try {
          const uint8Array = new Uint8Array(this.screenshot);
          const binary = Array.from(uint8Array)
            .map(byte => String.fromCharCode(byte))
            .join('');
          const base64String = btoa(binary);
          return `data:image/png;base64,${base64String}`;
        } catch (error) {
          console.error('Error converting binary data to base64:', error);
          return '';
        }
      }
      
      return '';
    }
  },
  watch: {
    environmentId: {
      immediate: true,
      handler(newId, oldId) {
        if (newId && newId !== oldId) {
          console.log('WebArenaViewer: environmentId changed to', newId)
          this.lastRefreshId = null
          this.lastScreenshotId = null
          this.refreshState()
          this.fetchScreenshot()
        }
      }
    },
    envState: {
      deep: true,
      handler(newState, oldState) {
        if (newState) {
          console.log('WebArenaViewer: envState updated', newState)
          
          const stateChanged = oldState === null || 
                               JSON.stringify(oldState) !== JSON.stringify(newState);
          
          if (stateChanged) {
            this.currentState = newState
            this.isDone = newState.terminated || newState.done || false
            this.extractTaskInfo(newState)
            
            this.$emit('state-updated', newState)
            
            if (!this.isDone && !this.isScreenshotting) {
              this.fetchScreenshot()
            }
          }
        }
      }
    }
  },
  methods: {
    async refreshState() {
      if (!this.client || !this.environmentId || this.isRefreshing) return
      
      this.isRefreshing = true
      this.loading = true
      console.log('WebArenaViewer: refreshing state for environment', this.environmentId)
      
      try {
        const result = await this.client.getObservation(this.environmentId)
        
        if (result && result.success) {
          let observationData = result.data
          
          // Handle promise
          if (observationData && typeof observationData.then === 'function') {
            observationData = await observationData
          }
          
          console.log('WebArenaViewer: got observation data', observationData)
          console.log('WebArenaViewer: observation data type:', typeof observationData)
          
          // Handle both string and object observation data
          let dataToProcess = observationData;
          
          // If observationData is a string containing the full observation text
          if (typeof observationData === 'string') {
            // Wrap string data in an object structure for consistent processing
            dataToProcess = {
              observation: observationData,
              initialObservation: observationData
            };
            console.log('WebArenaViewer: wrapped string observation data for processing')
          } else if (typeof observationData === 'object' && observationData !== null) {
            // If it's an object, ensure we have the observation field accessible
            if (!dataToProcess.initialObservation && dataToProcess.observation) {
              dataToProcess.initialObservation = dataToProcess.observation;
            }
          }
          
          console.log('WebArenaViewer: processing data for task extraction:', dataToProcess)
          
          this.extractTaskInfo(dataToProcess)
          this.isDone = observationData.terminated || observationData.done || false
          this.currentState = observationData
          this.lastRefreshId = this.environmentId
          
          this.$emit('state-updated', this.currentState)
        }
      } catch (error) {
        console.error('Error refreshing WebArena state:', error)
      } finally {
        this.loading = false
        this.isRefreshing = false
      }
    },
    
    extractTaskInfo(data) {
      if (!data) return
      
      // Try to extract task/objective from various fields
      if (typeof data === 'object') {
        // First priority: extract from initialObservation field
        if (data.initialObservation && typeof data.initialObservation === 'string') {
          const objectiveMatch = data.initialObservation.match(/OBJECTIVE:\s*([^\n]+)/i)
          if (objectiveMatch) {
            this.task = objectiveMatch[1].trim()
            console.log('WebArenaViewer: Extracted task from initialObservation:', this.task)
          }
          
          // Also try to extract URL from initialObservation
          const urlMatch = data.initialObservation.match(/URL:\s*([^\n]+)/i)
          if (urlMatch) {
            this.currentUrl = urlMatch[1].trim()
            console.log('WebArenaViewer: Extracted URL from initialObservation:', this.currentUrl)
          }
        }
        // Second priority: direct fields
        else if (data.task) {
          this.task = data.task
        } else if (data.objective) {
          this.task = data.objective
        } else if (data.intent) {
          this.task = data.intent
        }
        // Third priority: extract from observation text
        else if (data.observation && typeof data.observation === 'string') {
          // Try to extract OBJECTIVE from observation text (similar format)
          const objectiveMatch = data.observation.match(/OBJECTIVE:\s*([^\n]+)/i)
          if (objectiveMatch) {
            this.task = objectiveMatch[1].trim()
          } else {
            // Fallback: try other patterns
            const intentMatch = data.observation.match(/(?:intent|objective|task):\s*([^\n]+)/i)
            if (intentMatch) {
              this.task = intentMatch[1].trim()
            }
          }
          
          // Try to extract URL from observation text
          const urlMatch = data.observation.match(/URL:\s*([^\n]+)/i)
          if (urlMatch) {
            this.currentUrl = urlMatch[1].trim()
          }
        }
        
        // Extract URL if not already found and available in other fields
        if (!this.currentUrl) {
          if (data.url) {
            this.currentUrl = data.url
          } else if (data.metadata && data.metadata.url) {
            this.currentUrl = data.metadata.url
          }
        }
      }
      // Handle string data (could be the full observation text)
      else if (typeof data === 'string') {
        // Try to extract OBJECTIVE from string data
        const objectiveMatch = data.match(/OBJECTIVE:\s*([^\n]+)/i)
        if (objectiveMatch) {
          this.task = objectiveMatch[1].trim()
          console.log('WebArenaViewer: Extracted task from string data:', this.task)
        }
        
        // Try to extract URL from string data
        const urlMatch = data.match(/URL:\s*([^\n]+)/i)
        if (urlMatch) {
          this.currentUrl = urlMatch[1].trim()
          console.log('WebArenaViewer: Extracted URL from string data:', this.currentUrl)
        }
      }
    },
    
    async fetchScreenshot() {
      if (!this.client || !this.environmentId || this.isDone || this.isScreenshotting) return
      
      this.isScreenshotting = true
      console.log('WebArenaViewer: fetching screenshot for environment', this.environmentId)
      
      try {
        const result = await this.client.getScreenshot(this.environmentId)
        
        if (result && result.success && result.data) {
          this.screenshot = result.data
          this.lastScreenshotId = this.environmentId
          console.log('WebArenaViewer: screenshot updated successfully')
        } else {
          console.log('WebArenaViewer: no screenshot available')
          this.screenshot = null
        }
      } catch (error) {
        console.error('Error fetching WebArena screenshot:', error)
        this.screenshot = null
      } finally {
        this.isScreenshotting = false
      }
    }
  },
  mounted() {
    console.log('WebArenaViewer mounted, environment ID:', this.environmentId)
    
    setTimeout(() => {
      if (this.environmentId && this.client) {
        this.refreshState()
        this.fetchScreenshot()
      }
    }, 50)
  }
}
</script>

<style scoped>
.webarena-viewer {
  padding: 1.5rem;
  height: 100%;
  overflow-y: auto;
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.task-section {
  background: white;
  border-radius: 8px;
  padding: 1.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.task-text {
  background: #e3f2fd;
  color: #0277bd;
  font-weight: 1200;
  padding: 1rem;
  border-radius: 4px;
  border-left: 4px solid #2196F3;
  font-size: 1.7rem;
}

.browser-section {
  background: white;
  border-radius: 8px;
  padding: 1.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  flex: 1;
}

.screenshot-container {
  background: #f5f5f5;
  border-radius: 8px;
  padding: 1rem;
  text-align: center;
  min-height: 600px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.screenshot-container img {
  max-width: 100%;
  max-height: 900px;
  height: auto;
  border-radius: 4px;
  border: 1px solid #e0e0e0;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.no-screenshot-message {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #666;
  padding: 2rem;
}

.completed-message {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
}

.completion-icon {
  font-size: 4rem;
  color: #4CAF50;
  margin-bottom: 1rem;
}

.completion-text {
  font-size: 1.5rem;
  font-weight: 600;
  color: #4CAF50;
}

.url-display {
  background: #e8eaf6;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  margin-top: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
}

.url-label {
  font-weight: 600;
  color: #555;
}

.url-value {
  color: #3F51B5;
  word-break: break-all;
  font-family: monospace;
  flex-grow: 1;
  background: white;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  border-left: 3px solid #3F51B5;
}

.task-section h3, .browser-section h3 {
  font-size: 2rem;
  margin-bottom: 0.5rem;
  color: #333;
  border-bottom: 2px solid #2196F3;
  padding-bottom: 0.5rem;
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
  width: 40px;
  height: 40px;
  border: 4px solid #e9ecef;
  border-top: 4px solid #2196F3;
  border-radius: 50%;
  margin-bottom: 1rem;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
