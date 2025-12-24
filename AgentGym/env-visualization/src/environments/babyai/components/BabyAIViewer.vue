<template>
  <div class="babyai-viewer">
    <!-- Mission -->
    <div class="mission-section">
      <h3>Goal</h3>
      <div class="mission-text">{{ mission }}</div>
    </div>
    
    <!-- Render Image -->
    <div class="image-container">
      <img v-if="renderData && renderData.image" :src="renderData.image" alt="BabyAI Environment" />
      <div v-else-if="isDone" class="completed-message">
        <div class="completion-icon">✅</div>
        <div class="completion-text">Task completed!</div>
      </div>
      <div v-else class="no-image-message">
        <div class="loading-spinner"></div>
        <div>Loading environment...</div>
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
  name: 'BabyAIViewer',
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
      renderData: null,
      mission: 'No mission specified',
      isDone: false,
      
      // Deduplication tracking
      lastRefreshId: null,       // Track the last environment ID refreshed
      lastRenderImageId: null,   // Track the last environment ID rendered
      renderStateVersion: 0,     // Counter to track state changes requiring new renders
      isRefreshing: false,       // Flag to prevent concurrent refreshes
      isRendering: false,        // Flag to prevent concurrent renders
      refreshDebounceTimer: null // Timer for debouncing refresh calls
    }
  },
  watch: {
    environmentId: {
      immediate: true,
      handler(newId, oldId) {
        if (newId && newId !== oldId) {
          console.log('BabyAIViewer: environmentId changed to', newId)
          this.lastRefreshId = null
          this.lastRenderImageId = null
          this.renderStateVersion = 0
          this.refreshState()
          this.fetchRenderImage()
        }
      }
    },
    envState: {
      deep: true,
      handler(newState, oldState) {
        if (newState) {
          console.log('BabyAIViewer: envState updated', newState)
          
          const stateChanged = oldState === null || 
                               JSON.stringify(oldState) !== JSON.stringify(newState);
          
          // Only update if state actually changed
          if (stateChanged) {
            this.currentState = newState
            this.isDone = newState.done || false
            this.extractMission(newState.observation)
            
            // Increment render version to force a new render
            this.renderStateVersion++
            
            // Re-emit state update to ensure it propagates
            if (newState.observation) {
              console.log('BabyAIViewer: re-emitting state update')
              this.$emit('state-updated', newState)
            }
            
            // Don't try to render if the environment is done
            if (!this.isDone && !this.isRendering) {
              this.fetchRenderImage()
            }
          } else {
            console.log('BabyAIViewer: state unchanged, skipping update')
          }
        }
      }
    }
  },
  methods: {
    debouncedRefreshState() {
      // Cancel any existing timer
      if (this.refreshDebounceTimer) {
        clearTimeout(this.refreshDebounceTimer)
      }
      
      // Set a new timer
      this.refreshDebounceTimer = setTimeout(() => {
        this.refreshState()
      }, 100)
    },
    
    async refreshState() {
      if (!this.client || !this.environmentId) return
      
      // Skip if already refreshing or same environment ID
      if (this.isRefreshing) {
        console.log('BabyAIViewer: Skipping refreshState - already refreshing')
        return
      }
      
      this.isRefreshing = true
      this.loading = true
      console.log('BabyAIViewer: refreshing state for environment', this.environmentId)
      
      try {
        const result = await this.client.getObservation(this.environmentId)
        
        if (result && result.success) {
          // Process observation data
          let observationData = result.data
          
          // Handle promise
          if (observationData && typeof observationData.then === 'function') {
            observationData = await observationData
          }
          
          console.log('BabyAIViewer: got observation data', observationData)
          
          // Extract mission if possible
          this.extractMission(observationData)
          
          // Update done state
          this.isDone = observationData.done || false
          
          // Update state
          this.currentState = observationData
          
          // Remember this environment ID
          this.lastRefreshId = this.environmentId
          
          // Emit state updated event
          this.$emit('state-updated', this.currentState)
        }
      } catch (error) {
        console.error('Error refreshing BabyAI state:', error)
      } finally {
        this.loading = false
        this.isRefreshing = false
      }
    },
    
    extractMission(observation) {
      // Try to extract mission from observation
      if (typeof observation === 'string') {
        // Check for standard BabyAI format with mission
        const missionMatch = observation.match(/Your goal:\s*(.+?)(?:\n|$)/i)
        if (missionMatch) {
          this.mission = missionMatch[1].trim()
          console.log('BabyAIViewer: extracted mission from observation:', this.mission)
          return
        }
      } else if (typeof observation === 'object' && observation !== null) {
        // If render data has mission, use that
        if (this.renderData && this.renderData.mission) {
          this.mission = this.renderData.mission
          console.log('BabyAIViewer: using mission from render data:', this.mission)
          return
        }
      }
    },
    
    async fetchRenderImage() {
      if (!this.client || !this.environmentId || this.isDone) return
      
      // Skip if already rendering, but DON'T skip based on ID anymore
      // We want to render after each state change
      if (this.isRendering) {
        console.log('BabyAIViewer: Skipping fetchRenderImage - already rendering')
        return
      }
      
      this.isRendering = true
      console.log('BabyAIViewer: fetching render image for environment', this.environmentId, 'version', this.renderStateVersion)
      
      try {
        // Check if client has render method
        if (typeof this.client.render === 'function') {
          const result = await this.client.render(this.environmentId)
          
          console.log('BabyAIViewer: render result received', result)
          
          if (result && result.success && result.data) {
            // Process the render data based on its structure
            const responseData = result.data
            console.log('BabyAIViewer: render data type:', typeof responseData)
            
            // Check if we need to adapt the response format
            if (typeof responseData === 'string' && responseData.startsWith('data:image')) {
              // Direct image string
              this.renderData = { image: responseData }
              console.log('BabyAIViewer: received direct image string')
            } else if (typeof responseData === 'object' && responseData !== null) {
              // Standard object response
              this.renderData = responseData
              console.log('BabyAIViewer: received object response with keys:', Object.keys(responseData))
            } else {
              console.error('BabyAIViewer: unsupported render data format:', responseData)
              this.renderData = null
            }
            
            // Remember this environment ID
            this.lastRenderImageId = this.environmentId
          } else {
            console.error('Failed to render BabyAI image:', result?.error)
            this.renderData = null
          }
        }
      } catch (error) {
        console.error('Error fetching BabyAI render image:', error)
        this.renderData = null
      } finally {
        this.isRendering = false
      }
    }
  },
  mounted() {
    console.log('BabyAIViewer mounted, environment ID:', this.environmentId)
    
    // Delay the initial call to prevent race conditions with multiple instances
    setTimeout(() => {
      if (this.environmentId && this.client) {
        this.refreshState()
        this.fetchRenderImage()
      }
    }, 50)
  }
}
</script>

<style scoped>
.babyai-viewer {
  padding: 0.5rem;
  height: 100%;
  overflow: hidden;
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.mission-section {
  background: white;
  border-radius: 6px;
  padding: 0.6rem;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
  flex-shrink: 0;
}

.mission-text {
  background: #e3f2fd;
  color: #0277bd;
  font-weight: 600;
  padding: 0.5rem;
  border-radius: 4px;
  border-left: 3px solid #2196F3;
  font-size: 1.7rem;
  line-height: 1.2;
}

.image-container {
  background: white;
  border-radius: 6px;
  padding: 0.3rem;
  text-align: center;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
  overflow: hidden;
  min-height: 0;
}

.image-container img {
  width: 60%;
  height: 60%;
  max-width: 100%;
  max-height: 100%;
  border-radius: 4px;
  object-fit: contain;
  display: block;
}

.no-image-message {
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

.mission-section h3 {
  font-size: 2rem;
  margin-bottom: 0.3rem;
  color: #333;
  border-bottom: 2px solid #2196F3;
  padding-bottom: 0.3rem;
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
