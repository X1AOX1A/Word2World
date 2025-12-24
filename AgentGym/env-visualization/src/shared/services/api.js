/**
 * Multi-Environment API Client
 * 
 * This module provides an abstraction layer for communicating with multiple
 * AgentEnv server types (TextCraft, BabyAI, SciWorld, WebArena, SearchQA).
 */

import aiAgent from './aiAgent.js'
import environmentManager from './environmentManager.js'

// Configuration for different environments
const ENV_CONFIG = {
  textcraft: {
    baseUrl: '/api/textcraft',  // Use proxy path to avoid CORS issues
    endpoints: {
      hello: '/',
      create: '/create',
      step: '/step',
      reset: '/reset',
      observation: '/observation'
    }
  },
  babyai: {
    baseUrl: '/api/babyai',
    endpoints: {
      hello: '/',
      create: '/create',
      step: '/step',
      reset: '/reset',
      observation: '/observation',
      render: '/render'
    }
  },
  sciworld: {
    baseUrl: '/api/sciworld',
    endpoints: {
      hello: '/',
      create: '/create',
      step: '/step',
      reset: '/reset',
      observation: '/observation',
      goals: '/goals',
      detail: '/detail',
      objectTree: '/object_tree',
      inventory: '/inventory',
      taskDescription: '/task_description',
      possibleActions: '/possible_actions',
      possibleObjects: '/possible_objects',
      actionHint: '/action_hint'
    }
  },
  webarena: {
    baseUrl: '/api/webarena',
    endpoints: {
      hello: '/',
      create: '/create',
      step: '/step',
      reset: '/reset',
      observation: '/observation',
      observationMetadata: '/observation_metadata'
    }
  },
  searchqa: {
    baseUrl: '/api/searchqa',
    endpoints: {
      hello: '/',
      create: '/create',
      step: '/step',
      reset: '/reset',
      observation: '/observation'
    }
  }
}

class MultiEnvironmentAPI {
  constructor() {
    this.currentEnvType = 'textcraft' // Default environment
    this.environmentConfigs = new Map() // Store environment configs (commands, goals)
    this.autoRunning = new Map() // Track auto-running environments
    this.environmentInventories = new Map() // Store inventory state for each environment
    this.interactionHistories = new Map() // Store interaction history for each environment
    this.verboseLogging = false // Enable/disable verbose logging for auto inventory queries
  }

  /**
   * Set the current environment type
   */
  setEnvironmentType(envType) {
    if (!ENV_CONFIG[envType]) {
      throw new Error(`Unknown environment type: ${envType}`)
    }
    this.currentEnvType = envType
    console.log(`Switched API to environment: ${envType}`)
  }

  /**
   * Get current environment configuration
   */
  getCurrentConfig() {
    return ENV_CONFIG[this.currentEnvType]
  }

  /**
   * Get base URL for current environment
   */
  get baseUrl() {
    return this.getCurrentConfig().baseUrl
  }

  /**
   * Get endpoints for current environment
   */
  get endpoints() {
    return this.getCurrentConfig().endpoints
  }

  /**
   * Make a HTTP request to the API
   * @param {string} endpoint - The endpoint path
   * @param {object} options - Request options (method, body, etc.)
   */
  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`
    
    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutMs = options.timeout || 120000; // 120 second default timeout for WebArena compatibility
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    
    const config = {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: controller.signal,
      ...options
    }

    try {
      console.log(`🔗 Making request to: ${url}`, config.method || 'GET');
      const response = await fetch(url, config)
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const contentType = response.headers.get('content-type')
      if (contentType && contentType.includes('application/json')) {
        return await response.json()
      } else {
        return await response.text()
      }
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error.name === 'AbortError') {
        const timeoutError = new Error(`Request timeout after ${timeoutMs / 1000} seconds`);
        timeoutError.name = 'TimeoutError';
        console.error(`⏰ Request timeout for ${url}:`, timeoutError);
        throw timeoutError;
      }
      
      console.error('API request failed:', error)
      throw error
    }
  }

  /**
   * Test connection to the server
   */
  async hello() {
    return this.request(this.endpoints.hello)
  }

  /**
   * Create a new environment with proper three-step initialization
   * @param {string} commands - Initial commands (optional)
   * @param {string} goal - Goal description (optional)
   */
  async createEnvironment(commands = '', goal = '') {
    let body
    
    // Different body format for different environment types
    if (this.environmentType === 'babyai') {
      body = {
        task_id: Math.floor(Math.random() * 4639), // Random task for BabyAI
        seed: 42
      }
    } else {
      // TextCraft format
      body = {
        minecraft_dir: "agentenv_textcraft",
        commands: commands && commands.trim() ? commands.trim() : null,
        goal: goal && goal.trim() ? goal.trim() : null
      }
    }
    
    console.log(`🔨 Creating ${this.environmentType} environment with body:`, body)
    
    // Step 1: Create environment
    const createResponse = await this.request(this.endpoints.create, {
      method: 'POST',
      body: JSON.stringify(body)
    })

    // Check for errors in response
    if (createResponse.error) {
      throw new Error(`Server error: ${createResponse.error}`)
    }

    if (createResponse.id === undefined) {
      throw new Error('No environment ID returned from create')
    }

    const envId = createResponse.id
    console.log(`✅ Environment created with ID: ${envId}`)

    // Step 2: Reset environment (for proper initialization)
    console.log(`🔄 Resetting environment ${envId}`)
    try {
      // Use the proper reset method that handles environment-specific parameters
      await this.reset(envId, 0)  // Use data_idx = 0 as default
      console.log(`✅ Environment ${envId} reset successfully`)
    } catch (resetError) {
      console.warn(`⚠️ Reset failed, continuing anyway:`, resetError)
    }

    // Step 3: Get initial observation
    console.log(`👁️ Getting initial observation for environment ${envId}`)
    let observation
    try {
      observation = await this.getObservation(envId)
      console.log(`✅ Initial observation retrieved:`, observation.substring(0, 100) + '...')
    } catch (obsError) {
      console.warn(`⚠️ Failed to get initial observation:`, obsError)
      observation = createResponse.observation || "Environment created successfully"
    }

    // Store environment configuration for display and auto-run
    const parsedInfo = observation ? this.parseObservationInfo(observation) : { commands: null, goal: null }
    
    this.environmentConfigs.set(envId, {
      commands: parsedInfo.commands || (commands && commands.trim() ? commands.trim() : null),
      goal: parsedInfo.goal || (goal && goal.trim() ? goal.trim() : null),
      created: new Date().toISOString()
    })

    // Return response in expected format
    const result = {
      id: envId,
      observation: observation,
      reward: createResponse.reward || 0,
      done: createResponse.done || false
    }

    console.log(`🎉 Environment ${envId} fully initialized:`, result)
    return result
  }

  /**
   * Execute a step/action in the environment
   * @param {string} envId - Environment ID
   * @param {string} action - Action to execute
   * @param {boolean} autoQueryInventory - Whether to automatically query inventory after this action
   */
  async step(envId, action, autoQueryInventory = true) {
    // Clean the action similar to how Agent does it
    let cleanedAction = action
    
    // Remove </s> suffix if present
    if (cleanedAction.endsWith("</s>")) {
      cleanedAction = cleanedAction.slice(0, -5)
    }
    
    // Extract action after "Action:" if present
    const actionParts = cleanedAction.split("Action:")
    if (actionParts.length > 1) {
      cleanedAction = actionParts[1].trim()
    } else {
      cleanedAction = actionParts[0].trim()
    }

    const body = {
      id: parseInt(envId),
      action: cleanedAction
    }
    
    console.log(`🚀 Executing step for env ${envId}, action: "${cleanedAction}"`);
    
    // Use longer timeout for WebArena compatibility (3 minutes)
    const response = await this.request(this.endpoints.step, {
      method: 'POST',
      body: JSON.stringify(body),
      timeout: 180000 // 3 minutes timeout for step operations
    })

    // Check for errors in response
    if (response.error) {
      console.error('Step error:', response.error)
      throw new Error(`Server error: ${response.error}`)
    }

    // Log the full response for debugging with better formatting
    console.log(`📝 Step response for action "${cleanedAction}":`)
    console.log(`   - done: ${response.done}`)
    console.log(`   - state: ${response.state ? response.state.substring(0, 50) + '...' : 'null'}`)
    console.log(`   - observation: ${response.observation ? response.observation.substring(0, 50) + '...' : 'null'}`)
    console.log(`   - reward: ${response.reward}`)
    console.log(`   - info: ${response.info ? JSON.stringify(response.info) : 'null'}`)
    console.log(`   - all keys: ${Object.keys(response).join(', ')}`)
    
    // Log completion detection for manual actions
    if (response.done === true) {
      console.log(`🏆 Manual action "${cleanedAction}" completed the task!`)
    }

    // Update inventory based on the action and response
    if (response.observation) {
      this.updateInventory(parseInt(envId), cleanedAction, response.observation)
    }

    // Add to interaction history for manual commands (not for silent inventory queries)
    if (autoQueryInventory) {
      this.addToInteractionHistory(envId, cleanedAction, response.observation || response.state || JSON.stringify(response))
    }

    // Auto-query inventory after non-inventory actions (if enabled)
    if (autoQueryInventory && cleanedAction.toLowerCase().trim() !== 'inventory') {
      console.log(`🎒 Auto-querying inventory after action: ${cleanedAction}`)
      
      // Query inventory without adding to interaction history, but log it
      try {
        const inventoryResponse = await this.queryInventorySilent(envId)
        console.log(`🎒 Inventory query completed, found ${Array.isArray(inventoryResponse) ? inventoryResponse.length : 0} items`)
        
        // If verbose logging is enabled, also add to interaction history
        if (this.verboseLogging) {
          this.addToInteractionHistory(envId, '[Auto] inventory', `Found ${Array.isArray(inventoryResponse) ? inventoryResponse.length : 0} items in inventory`)
        }
      } catch (error) {
        console.warn('⚠️ Auto inventory query failed:', error)
        // Don't throw - this shouldn't break the main action
      }
    }

    return response
  }

  /**
   * Query current inventory by sending an inventory command
   * @param {string} envId - Environment ID
   * @returns {Promise<Array>} Current inventory items
   */
  async queryInventory(envId) {
    console.log(`🎒 Querying inventory for environment ${envId}`)
    
    try {
      // Send inventory command to get current state (disable auto-query to avoid recursion)
      const response = await this.step(envId, 'inventory', false)
      
      // The inventory should be updated by the step call above
      // Return the parsed inventory
      return this.getInventory(parseInt(envId))
    } catch (error) {
      console.error('❌ Failed to query inventory:', error)
      return []
    }
  }

  /**
   * Query inventory silently (for auto-refresh, not added to interaction history)
   * @param {string} envId - Environment ID
   * @returns {Promise<Array>} Current inventory items
   */
  async queryInventorySilent(envId) {
    console.log(`🎒 Silent inventory query for environment ${envId}`)
    
    try {
      // Send inventory command directly to API without going through step method
      const body = {
        id: parseInt(envId),
        action: 'inventory'
      }
      
      const response = await this.request(this.endpoints.step, {
        method: 'POST',
        body: JSON.stringify(body)
      })

      // Check for errors in response
      if (response.error) {
        throw new Error(`Server error: ${response.error}`)
      }

      // Update inventory based on the response
      if (response.observation) {
        this.updateInventory(parseInt(envId), 'inventory', response.observation)
      }

      // Return the updated inventory
      return this.getInventory(parseInt(envId))
    } catch (error) {
      console.error('❌ Failed to query inventory silently:', error)
      return []
    }
  }

  /**
   * Refresh inventory after potentially inventory-changing actions
   * @param {string} envId - Environment ID  
   * @param {string} lastAction - The action that was just performed
   */
  async refreshInventoryIfNeeded(envId, lastAction) {
    // Define actions that likely change inventory
    const inventoryChangingActions = [
      'get', 'mine', 'craft', 'smelt', 'use', 'place', 'drop'
    ]
    
    const actionLower = lastAction.toLowerCase()
    const shouldRefresh = inventoryChangingActions.some(action => actionLower.includes(action))
    
    if (shouldRefresh) {
      console.log(`🔄 Refreshing inventory after action: ${lastAction}`)
      return await this.queryInventory(envId)
    }
    
    return this.getInventory(parseInt(envId))
  }

  /**
   * Reset the environment
   * @param {string} envId - Environment ID
   * @param {number} dataIdx - Data index (default: 0)
   */
  async reset(envId, dataIdx = 0) {
    // Different environments have different reset parameter requirements
    let body
    
    if (this.currentEnvType === 'textcraft') {
      // TextCraft: data_idx is optional with default 0
      body = {
        id: parseInt(envId),
        data_idx: dataIdx
      }
    } else if (this.currentEnvType === 'babyai' || this.currentEnvType === 'sciworld') {
      // BabyAI and SciWorld: data_idx is required
      body = {
        id: parseInt(envId),
        data_idx: parseInt(dataIdx) || 0  // Ensure it's an integer
      }
    } else {
      // Default behavior for other environments
      body = {
        id: parseInt(envId),
        data_idx: dataIdx
      }
    }
    
    console.log(`🔄 Resetting ${this.currentEnvType} environment ${envId} with data_idx=${body.data_idx}`)
    
    const response = await this.request(this.endpoints.reset, {
      method: 'POST',
      body: JSON.stringify(body)
    })

    // Check for errors in response
    if (response.error) {
      console.error('Reset error:', response.error)
      throw new Error(`Server error: ${response.error}`)
    }

    // After reset, update environment configuration from observation
    if (response.observation) {
      const parsedInfo = this.parseObservationInfo(response.observation)
      const id = parseInt(envId)
      
      // Clear inventory on reset - environment should start empty
      this.environmentInventories.set(id, new Map())
      
      // Clear interaction history on reset - start fresh
      this.interactionHistories.set(id, [])
      
      // Do NOT parse inventory from reset observation - it should be empty
      // Only parse the actual inventory command response will populate it
      
      // Update stored configuration with parsed information
      if (this.environmentConfigs.has(id)) {
        const config = this.environmentConfigs.get(id)
        this.environmentConfigs.set(id, {
          ...config,
          commands: parsedInfo.commands || config.commands,
          goal: parsedInfo.goal || config.goal
        })
      } else {
        // Create new config if not exists
        this.environmentConfigs.set(id, {
          commands: parsedInfo.commands,
          goal: parsedInfo.goal,
          created: new Date().toISOString()
        })
      }
    }

    return response
  }

  /**
   * Get current observation
   * @param {string} envId - Environment ID
   */
  async getObservation(envId) {
    const params = new URLSearchParams({ id: envId })
    const response = await this.request(`${this.endpoints.observation}?${params.toString()}`)
    
    console.log('🔍 getObservation envId:', envId);
    console.log('🔍 getObservation endpoint:', this.endpoints.observation);
    console.log('🔍 getObservation response type:', typeof response, response?.constructor?.name);
    console.log('🔍 getObservation response:', response);
    
    // Special handling for WebArena - the client returns the full observation directly
    if (this.currentEnvType === 'webarena') {
      // For WebArena, if response is already a string, return it directly
      if (typeof response === 'string') {
        console.log('✅ WebArena: Response is already a string');
        return response;
      }
      
      // If response is an object, check for observation field first
      if (typeof response === 'object' && response !== null) {
        if (response.observation) {
          console.log('✅ WebArena: Extracted observation field from response:', response.observation);
          return typeof response.observation === 'string' 
            ? response.observation 
            : JSON.stringify(response.observation, null, 2);
        }
      }
    }
    
    // 如果响应是对象，提取观察文本
    if (typeof response === 'object' && response !== null) {
      // 尝试提取观察文本
      if (response.observation) {
        console.log('✅ Extracted observation field from response:', response.observation);
        return typeof response.observation === 'string' 
          ? response.observation 
          : JSON.stringify(response.observation, null, 2);
      } else if (response.text) {
        console.log('✅ Extracted text field from response:', response.text);
        return response.text;
      } else {
        // 如果没有明确的观察字段，转换整个对象为JSON字符串
        console.log('⚠️ Converting entire response object to JSON string');
        return JSON.stringify(response, null, 2);
      }
    }
    
    // 如果响应是字符串，直接返回
    if (typeof response === 'string') {
      console.log('✅ Response is already a string');
      return response;
    }
    
    // 处理数组或其他类型
    console.log('⚠️ Converting response to string for unknown type');
    return String(response || 'No observation available');
  }

  /**
   * Get comprehensive environment state (simplified version using only observation)
   * @param {string} envId - Environment ID
   */
  async getState(envId) {
    try {
      const observation = await this.getObservation(envId)
      const config = this.environmentConfigs.get(parseInt(envId))
      
      // Try to parse current observation for updated commands/goal
      const parsedInfo = this.parseObservationInfo(observation)
      
      // Use parsed info if available, otherwise fall back to stored config
      const commands = parsedInfo.commands || config?.commands || "Available commands: craft, mine, smelt, get, inventory"
      const goal = parsedInfo.goal || config?.goal || "Complete the crafting objectives"

      return {
        observation,
        commands,
        goal,
        detail: "TextCraft Environment Details",
        timestamp: new Date().toISOString()
      }
    } catch (error) {
      console.error('Failed to get complete state:', error)
      throw error
    }
  }

  /**
   * Check if server is reachable
   */
  async checkConnection() {
    try {
      await this.hello()
      return true
    } catch {
      return false
    }
  }

  /**
   * Get rendered image from environment (for BabyAI and other visual environments)
   * @param {string} envId - Environment ID
   */
  async render(envId) {
    try {
      // Use different render endpoints based on environment type
      if (this.currentEnvType === 'babyai') {
        // BabyAI expects POST request with id in body as object
        const response = await this.request(this.endpoints.render, {
          method: 'POST',
          body: JSON.stringify({ id: parseInt(envId) })
        })
        
        // BabyAI render should return base64 image data
        if (response && response.image) {
          return response.image  // Return the base64 data directly
        }
        return response
      } else {
        // Other environments might have different render formats
        const params = new URLSearchParams({ id: envId })
        return this.request(`${this.endpoints.render}?${params.toString()}`)
      }
    } catch (error) {
      console.warn(`Render failed for environment ${envId}:`, error)
      return null
    }
  }

  /**
   * Get environment configuration (commands and goal)
   * @param {string} envId - Environment ID
   */
  getEnvironmentConfig(envId) {
    return this.environmentConfigs.get(parseInt(envId))
  }

  /**
   * Start auto-run for an environment with conversation-based AI
   * @param {string} envId - Environment ID
   * @param {function} onStep - Callback for each step (observation, action, reward, done)
   * @param {number} maxRounds - Maximum number of rounds (default: 50)
   */
  async startAutoRun(envId, onStep, maxRounds = 50) {
    const id = parseInt(envId)
    this.autoRunning.set(id, true)
    
    let rounds = 0
    let done = false
    
    // Get environment configuration for goal and commands
    const config = this.environmentConfigs.get(id)
    const goal = config?.goal || ''
    const commands = config?.commands || ''
    
    // Get initial observation with retry logic
    let observationResponse;
    let initialRetryCount = 0;
    const maxInitialRetries = 3;
    
    while (initialRetryCount <= maxInitialRetries) {
      try {
        console.log(`🔍 Attempting to get initial observation (attempt ${initialRetryCount + 1}/${maxInitialRetries + 1})`);
        observationResponse = await this.getObservation(envId);
        break; // Success, exit retry loop
      } catch (error) {
        initialRetryCount++;
        console.error(`❌ Initial observation attempt ${initialRetryCount} failed:`, error);
        
        if (initialRetryCount <= maxInitialRetries) {
          console.log(`🔄 Retrying initial observation fetch in 3 seconds...`);
          await new Promise(resolve => setTimeout(resolve, 3000));
        } else {
          console.error(`❌ All initial observation attempts failed after ${maxInitialRetries + 1} tries`);
          throw new Error(`Failed to get initial observation after ${maxInitialRetries + 1} attempts: ${error.message}`);
        }
      }
    }
    
    // Extract actual observation string from response
    let initialObservation = '';
    if (typeof observationResponse === 'string') {
      initialObservation = observationResponse;
    } else if (observationResponse && typeof observationResponse === 'object') {
      // Handle client response format like {success: true, data: {...}}
      if (observationResponse.data) {
        if (typeof observationResponse.data === 'string') {
          initialObservation = observationResponse.data;
        } else if (observationResponse.data.raw) {
          // For SearchQA, use the raw observation text
          initialObservation = observationResponse.data.raw;
        } else if (observationResponse.data.question) {
          // If no raw but has question, construct a basic observation
          initialObservation = `Question: ${observationResponse.data.question}`;
        } else {
          // Fallback to JSON string
          initialObservation = JSON.stringify(observationResponse.data);
        }
      } else if (observationResponse.observation) {
        initialObservation = typeof observationResponse.observation === 'string' 
          ? observationResponse.observation 
          : JSON.stringify(observationResponse.observation);
      } else {
        // Last resort - stringify the whole object
        initialObservation = JSON.stringify(observationResponse);
      }
    } else {
      initialObservation = String(observationResponse || 'No observation available');
    }
    
    console.log('🔍 Initial observation type:', typeof initialObservation);
    console.log('🔍 Initial observation preview:', initialObservation.substring(0, 200) + '...');
    
    // Initialize conversation for this environment using new AIAgent
    const envType = this.currentEnvType || 'textcraft'
    aiAgent.initializeConversation(id.toString(), envType, initialObservation, maxRounds)
    
    console.log(`🚀 Started conversation-based auto-run for env ${id} (${envType})`)
    
    while (!done && rounds < maxRounds && this.autoRunning.get(id)) {
      try {
        // Generate action using conversation-based approach
        const actionResult = await aiAgent.generateNextAction(id.toString(), initialObservation)
        
        if (actionResult.shouldStop) {
          console.log(`🛑 AI Agent requested stop: ${actionResult.reason}`)
          done = true
          break
        }
        
        const action = actionResult.action
        
        if (onStep) {
          onStep({
            round: rounds + 1,
            observation: initialObservation, // Use the properly extracted observation
            action,
            generating: false
          })
        }
        
        // Execute action
        let stepResult;
        let retryCount = 0;
        const maxRetries = 2;
        
        while (retryCount <= maxRetries) {
          try {
            console.log(`🎯 Attempting step execution (attempt ${retryCount + 1}/${maxRetries + 1})`);
            stepResult = await this.step(envId, action);
            break; // Success, exit retry loop
          } catch (error) {
            retryCount++;
            console.error(`❌ Step attempt ${retryCount} failed:`, error);
            
            if (retryCount <= maxRetries) {
              console.log(`🔄 Retrying step execution in 2 seconds...`);
              await new Promise(resolve => setTimeout(resolve, 2000));
            } else {
              console.error(`❌ All step attempts failed after ${maxRetries + 1} tries`);
              throw error;
            }
          }
        }
        
        if (stepResult.success) {
          // Update observation for next round with retry logic
          let observationResponse;
          let obsRetryCount = 0;
          const maxObsRetries = 2;
          
          while (obsRetryCount <= maxObsRetries) {
            try {
              console.log(`🔍 Attempting to get observation (attempt ${obsRetryCount + 1}/${maxObsRetries + 1})`);
              observationResponse = await this.getObservation(envId);
              break; // Success, exit retry loop
            } catch (error) {
              obsRetryCount++;
              console.error(`❌ Observation attempt ${obsRetryCount} failed:`, error);
              
              if (obsRetryCount <= maxObsRetries) {
                console.log(`🔄 Retrying observation fetch in 1 second...`);
                await new Promise(resolve => setTimeout(resolve, 1000));
              } else {
                console.error(`❌ All observation attempts failed after ${maxObsRetries + 1} tries`);
                // Don't throw error, use previous observation as fallback
                console.log(`⚠️ Using previous observation as fallback`);
                observationResponse = initialObservation;
                break;
              }
            }
          }
          
          // Extract observation for next iteration
          if (typeof observationResponse === 'string') {
            initialObservation = observationResponse;
          } else if (observationResponse && observationResponse.data) {
            if (typeof observationResponse.data === 'string') {
              initialObservation = observationResponse.data;
            } else if (observationResponse.data.raw) {
              initialObservation = observationResponse.data.raw;
            } else {
              initialObservation = JSON.stringify(observationResponse.data);
            }
          }
          
          const reward = stepResult.data?.reward || 0
          done = stepResult.data?.done || false
          
          if (onStep) {
            onStep({
              round: rounds + 1,
              observation: initialObservation,
              action,
              reward,
              done,
              generating: false
            })
          }
          
          rounds++
          
          if (done) {
            console.log(`✅ Environment marked as done, stopping auto-run`)
            break
          }
          
        } else {
          console.error(`❌ Step failed:`, stepResult.error)
          done = true
          break
        }
        
        // Small delay to prevent overwhelming
        await new Promise(resolve => setTimeout(resolve, 100))
        
      } catch (error) {
        console.error(`❌ Auto-run error in round ${rounds + 1}:`, error)
        done = true
        break
      }
    }
    
    this.autoRunning.set(id, false)
    console.log(`🏁 Auto-run completed for env ${id} after ${rounds} rounds`)
  }

  /**
   * Stop auto-run for an environment
   * @param {string} envId - Environment ID
   */
  stopAutoRun(envId) {
    this.autoRunning.set(parseInt(envId), false)
  }

  /**
   * Check if auto-run is active for an environment
   * @param {string} envId - Environment ID
   */
  isAutoRunning(envId) {
    return this.autoRunning.get(parseInt(envId)) || false
  }

  /**
   * Generate action using AI or fallback to simple rules
   * @param {string} observation - Current observation
   * @param {string} goal - Environment goal
   * @param {string} commands - Available commands
   * @param {Array} interactionHistory - Interaction history (for backwards compatibility)
   */
  async generateAction(observation, goal = '', commands = '', interactionHistory = []) {
    try {
      // Use new AI Agent with correct signature: envType first
      const envType = this.environmentType || 'textcraft'
      const aiSuggestion = await aiAgent.getSuggestion(envType, observation, goal, commands, interactionHistory)
      if (aiSuggestion && aiSuggestion.trim()) {
        return aiSuggestion.trim()
      }
    } catch (error) {
      console.warn('AI suggestion failed, using fallback rules:', error)
    }

    // Fallback to simple rule-based action generation
    if (observation.includes('Inventory:') && observation.includes('acacia log')) {
      return 'craft 4 acacia planks using 1 acacia log'
    }
    
    if (observation.includes('Crafting commands:') && observation.includes('craft')) {
      // Extract crafting command from observation
      const craftMatch = observation.match(/craft [^\\n]+/i)
      if (craftMatch) {
        return craftMatch[0]
      }
    }
    
    if (observation.includes('need') || observation.includes('require')) {
      return 'get 1 acacia log'
    }
    
    // Default action
    return 'inventory'
  }

  /**
   * Parse observation text to extract Crafting commands and Goal
   * @param {string} observation - The observation text
   * @returns {object} Parsed information with commands and goal
   */
  /**
   * Update inventory based on action and observation
   * @param {number} envId - Environment ID
   * @param {string} action - The action that was executed
   * @param {string} observation - The observation from the environment
   */
  updateInventory(envId, action, observation) {
    console.log(`🎒 updateInventory called - envId: ${envId}, action: "${action}"`)
    
    // Initialize inventory for this environment if it doesn't exist
    if (!this.environmentInventories.has(envId)) {
      this.environmentInventories.set(envId, new Map())
      console.log(`🎒 initialized inventory for env ${envId}`)
    }

    const inventory = this.environmentInventories.get(envId)

    // Primary focus: Handle "inventory" action - this gives us the full current inventory
    if (action.toLowerCase().trim() === 'inventory') {
      console.log(`🎒 Processing inventory command response`)
      this.parseFullInventory(inventory, observation)
      console.log(`🎒 Updated inventory:`, Object.fromEntries(inventory))
      return
    }

    // Secondary: Still handle obvious incremental updates for better UX
    // But don't try to be too clever - inventory command will be the source of truth
    
    // Handle clear "got" responses from get actions
    const gotMatches = observation.matchAll(/Got\s+(\d+)\s+(.+?)(?:\n|$)/gi)
    for (const match of gotMatches) {
      const count = parseInt(match[1])
      let itemName = match[2].trim()
      
      // Clean item name
      if (itemName.startsWith('minecraft:')) {
        itemName = itemName.substring(10)
      }
      itemName = itemName.replace(/\s+/g, '_').toLowerCase()
      
      console.log(`🎒 Found "got" match: ${count} ${itemName}`)
      
      // Add to inventory
      const currentCount = inventory.get(itemName) || 0
      inventory.set(itemName, currentCount + count)
      console.log(`🎒 Updated ${itemName}: ${currentCount} -> ${currentCount + count}`)
    }

    // Handle clear "crafted" responses  
    const craftedMatches = observation.matchAll(/Crafted\s+(\d+)\s+(.+?)(?:\n|$)/gi)
    for (const match of craftedMatches) {
      const count = parseInt(match[1])
      let itemName = match[2].trim()
      
      // Clean item name
      if (itemName.startsWith('minecraft:')) {
        itemName = itemName.substring(10)
      }
      itemName = itemName.replace(/\s+/g, '_').toLowerCase()
      
      console.log(`🎒 Found "crafted" match: ${count} ${itemName}`)
      
      // Add crafted item to inventory
      const currentCount = inventory.get(itemName) || 0
      inventory.set(itemName, currentCount + count)
      console.log(`🎒 Updated crafted ${itemName}: ${currentCount} -> ${currentCount + count}`)
    }

    // Don't try to parse complex crafting logic - let inventory command handle the truth
    console.log(`🎒 Current inventory state:`, Object.fromEntries(inventory))
  }

  /**
   * Parse full inventory from inventory command response
   * @param {Map} inventory - The inventory map to update
   * @param {string} observation - The observation containing inventory info
   */
  parseFullInventory(inventory, observation) {
    console.log(`🎒 parseFullInventory - parsing observation`)
    
    // Look for inventory section in the observation
    const inventoryMatch = observation.match(/Inventory:\s*(.+?)(?:\n|$)/i)
    if (!inventoryMatch) {
      console.log(`🎒 parseFullInventory - no inventory section found`)
      return
    }
    
    const inventoryText = inventoryMatch[1].trim()
    console.log(`🎒 parseFullInventory - inventory text: "${inventoryText}"`)
    
    // Check if empty inventory
    if (inventoryText.includes("You are not carrying anything") || 
        inventoryText.includes('empty') ||
        inventoryText.includes('nothing') ||
        inventoryText === '') {
      console.log(`🎒 parseFullInventory - inventory is empty, clearing`)
      inventory.clear()
      return
    }
    
    // Parse items in format: [item name] (count)
    const itemMatches = Array.from(inventoryText.matchAll(/\[([^\]]+)\]\s*\((\d+)\)/g))
    if (itemMatches.length > 0) {
      console.log(`🎒 parseFullInventory - found ${itemMatches.length} items, rebuilding inventory`)
      inventory.clear()
      
      for (const match of itemMatches) {
        let itemName = match[1].trim()
        
        // Clean item name
        if (itemName.startsWith('minecraft:')) {
          itemName = itemName.substring(10)
        }
        itemName = itemName.replace(/\s+/g, '_').toLowerCase()
        
        const count = parseInt(match[2])
        inventory.set(itemName, count)
        console.log(`🎒 parseFullInventory - set ${itemName}: ${count}`)
      }
    } else {
      console.log(`🎒 parseFullInventory - no item matches found, inventory might be empty`)
      // Only clear if we're confident it's empty
      if (inventoryText.includes("You are not carrying anything") || 
          inventoryText.includes('empty') || 
          inventoryText.includes('nothing')) {
        console.log(`🎒 parseFullInventory - clearing inventory`)
        inventory.clear()
      }
    }
    
    console.log(`🎒 parseFullInventory - final inventory:`, Object.fromEntries(inventory))
  }

  /**
   * Get current inventory for an environment
   * @param {number} envId - Environment ID
   * @returns {Array} Array of inventory items with count and metadata
   */
  getInventory(envId) {
    console.log(`🎒 getInventory - called for envId: ${envId}`)
    
    const inventory = this.environmentInventories.get(envId)
    if (!inventory) {
      console.log(`🎒 getInventory - no inventory found for envId ${envId}`)
      return []
    }

    console.log(`🎒 getInventory - raw inventory:`, Object.fromEntries(inventory))

    // Convert Map to array of objects - we'll get item data in the component
    const result = []
    for (const [itemName, count] of inventory.entries()) {
      result.push({
        count,
        itemName
      })
    }

    console.log(`🎒 getInventory - returning:`, result)
    return result
  }

  parseObservationInfo(observation) {
    if (!observation || typeof observation !== 'string') {
      return { commands: null, goal: null }
    }

    let commands = null
    let goal = null

    // Extract Crafting commands
    const commandsMatch = observation.match(/Crafting commands:\s*([\s\S]*?)(?=Goal:|$)/i)
    if (commandsMatch) {
      commands = commandsMatch[1].trim()
    }

    // Extract Goal
    const goalMatch = observation.match(/Goal:\s*(.+?)(?:\n|$)/i)
    if (goalMatch) {
      goal = goalMatch[1].trim()
    }

    return { commands, goal }
  }

  /**
   * Get interaction history for an environment
   * @param {string} envId - Environment ID
   * @returns {Array} Array of {action, response} objects
   */
  getInteractionHistory(envId) {
    const id = parseInt(envId)
    return this.interactionHistories.get(id) || []
  }

  /**
   * Clear interaction history for an environment
   * @param {string} envId - Environment ID
   */
  clearInteractionHistory(envId) {
    const id = parseInt(envId)
    this.interactionHistories.set(id, [])
    console.log(`Interaction history cleared for environment ${id}`)
  }

  /**
   * Add interaction to history manually (for manual commands)
   * @param {string} envId - Environment ID
   * @param {string} action - Action taken
   * @param {string} response - Response received
   */
  addToInteractionHistory(envId, action, response) {
    const id = parseInt(envId)
    if (!this.interactionHistories.has(id)) {
      this.interactionHistories.set(id, [])
    }
    
    const history = this.interactionHistories.get(id)
    history.push({ action, response })
    
    // Keep only last 10 interactions
    if (history.length > 10) {
      history.splice(0, history.length - 10)
    }
    
    this.interactionHistories.set(id, history)
  }

  /**
   * Enable or disable verbose logging for auto inventory queries
   * @param {boolean} enabled - Whether to enable verbose logging
   */
  setVerboseLogging(enabled) {
    this.verboseLogging = enabled
    console.log(`Verbose logging ${enabled ? 'enabled' : 'disabled'}`)
  }

  /**
   * Get current verbose logging status
   * @returns {boolean} Current verbose logging status
   */
  isVerboseLogging() {
    return this.verboseLogging
  }

  // ========== Environment-Specific API Methods ==========

  /**
   * SciWorld-specific API methods
   */
  async getSciWorldInventory(envId) {
    if (this.currentEnvType !== 'sciworld') {
      throw new Error('API must be set to sciworld environment')
    }
    return this.request(`${this.endpoints.inventory}?id=${envId}`)
  }

  async getSciWorldGoals(envId) {
    if (this.currentEnvType !== 'sciworld') {
      throw new Error('API must be set to sciworld environment')
    }
    return this.request(`${this.endpoints.goals}?id=${envId}`)
  }

  async getSciWorldObjectTree(envId) {
    if (this.currentEnvType !== 'sciworld') {
      throw new Error('API must be set to sciworld environment')
    }
    return this.request(`${this.endpoints.objectTree}?id=${envId}`)
  }

  async getSciWorldPossibleActions(envId) {
    if (this.currentEnvType !== 'sciworld') {
      throw new Error('API must be set to sciworld environment')
    }
    return this.request(`${this.endpoints.possibleActions}?id=${envId}`)
  }

  async getSciWorldPossibleObjects(envId) {
    if (this.currentEnvType !== 'sciworld') {
      throw new Error('API must be set to sciworld environment')
    }
    return this.request(`${this.endpoints.possibleObjects}?id=${envId}`)
  }

  async getSciWorldTaskDescription(envId) {
    if (this.currentEnvType !== 'sciworld') {
      throw new Error('API must be set to sciworld environment')
    }
    return this.request(`${this.endpoints.taskDescription}?id=${envId}`)
  }

  async getSciWorldActionHint(envId) {
    if (this.currentEnvType !== 'sciworld') {
      throw new Error('API must be set to sciworld environment')
    }
    return this.request(`${this.endpoints.actionHint}?id=${envId}`)
  }

  /**
   * WebArena-specific API methods
   */
  async getWebArenaObservationMetadata(envId) {
    if (this.currentEnvType !== 'webarena') {
      throw new Error('API must be set to webarena environment')
    }
    return this.request(`${this.endpoints.observationMetadata}?env_idx=${envId}`)
  }

  /**
   * Generic method for environment-specific requests
   * Automatically handles the current environment type
   * @param {string} endpoint - The endpoint path
   * @param {object} params - Query parameters
   */
  async requestForCurrentEnv(endpoint, params = {}) {
    const queryString = new URLSearchParams(params).toString()
    const url = queryString ? `${endpoint}?${queryString}` : endpoint
    return this.request(url)
  }
}

// Export a singleton instance
const api = new MultiEnvironmentAPI()
export default api

// Also export the class for creating instances for other environments
export { MultiEnvironmentAPI, ENV_CONFIG }