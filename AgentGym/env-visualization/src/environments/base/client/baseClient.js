/**
 * Base Environment Client Template
 * 
 * This is a template for creating environment clients.
 * Copy this file and customize for each specific environment.
 * 
 * Examples:
 * - src/environments/babyai/client/BabyAIClient.js
 * - src/environments/sciworld/client/SciWorldClient.js
 * - src/environments/webarena/client/WebArenaClient.js
 * - src/environments/searchqa/client/SearchQAClient.js
 */

import apiService from '../../../services/api.js'

class BaseEnvironmentClient {
  constructor(environmentType = 'base') {
    this.environmentType = environmentType
    this.baseUrl = apiService.getBasePath(environmentType)
    this.environmentId = null
    this.currentState = {
      observation: '',
      reward: 0,
      done: false
    }
  }

  /**
   * Test connection to the environment server
   * @returns {Promise<Object>} Connection test result
   */
  async testConnection() {
    try {
      const data = await apiService.get(`${this.baseUrl}/`)
      return { success: true, data }
    } catch (error) {
      console.warn(`${this.environmentType} server (${this.baseUrl}) not available:`, error.message)
      return { success: false, error: error.message }
    }
  }

  /**
   * Create a new environment instance
   * @param {Object} config - Environment configuration
   * @returns {Promise<Object>} Creation result
   */
  async createEnvironment(config = {}) {
    try {
      const data = await apiService.post(`${this.baseUrl}/create`, config)
      
      if (data.error) {
        throw new Error(data.error)
      }
      
      this.environmentId = data.id || data.env_idx || data
      this.currentState = {
        observation: data.observation || '',
        reward: data.reward || 0,
        done: data.done || false
      }
      
      return { success: true, data }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Execute a step/action in the environment
   * @param {number|string} environmentId - Environment ID
   * @param {string} action - Action to execute
   * @returns {Promise<Object>} Step result
   */
  async step(environmentId, action) {
    try {
      const envId = environmentId || this.environmentId
      if (!envId) {
        throw new Error('No environment ID available. Create environment first.')
      }
      
      const data = await apiService.post(`${this.baseUrl}/step`, {
        id: envId,
        action: action
      })
      
      if (data.error) {
        console.warn('Step error:', data.error)
      }
      
      this.currentState = {
        observation: data.observation || '',
        reward: data.reward || 0,
        done: data.done || false
      }
      
      return { success: true, data }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Reset the environment
   * @param {number|string} environmentId - Environment ID
   * @param {number} dataIdx - Data index for reset
   * @returns {Promise<Object>} Reset result
   */
  async reset(environmentId, dataIdx = 0) {
    try {
      const envId = environmentId || this.environmentId
      if (!envId) {
        throw new Error('No environment ID available. Create environment first.')
      }
      
      const data = await apiService.post(`${this.baseUrl}/reset`, {
        id: envId,
        data_idx: dataIdx
      })
      
      if (data.error) {
        console.warn('Reset error:', data.error)
      }
      
      this.currentState = {
        observation: data.observation || '',
        reward: data.reward || 0,
        done: data.done || false
      }
      
      return { success: true, data }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Get current observation
   * @param {number|string} environmentId - Environment ID
   * @returns {Promise<Object>} Observation result
   */
  async getObservation(environmentId) {
    try {
      const envId = environmentId || this.environmentId
      if (!envId) {
        throw new Error('No environment ID available. Create environment first.')
      }
      
      const data = await apiService.get(`${this.baseUrl}/observation?id=${envId}`)
      return { success: true, data }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Close the environment
   * @param {number|string} environmentId - Environment ID
   * @returns {Promise<Object>} Close result
   */
  async close(environmentId) {
    try {
      const envId = environmentId || this.environmentId
      if (!envId) {
        throw new Error('No environment ID available. Create environment first.')
      }
      
      const data = await apiService.post(`${this.baseUrl}/close`, {
        id: envId
      })
      
      // Clear local state
      this.environmentId = null
      this.currentState = {
        observation: '',
        reward: 0,
        done: false
      }
      
      return { success: true, data }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Get current environment state
   * @returns {Object} Current state
   */
  getCurrentState() {
    return { ...this.currentState }
  }

  /**
   * Get environment ID
   * @returns {string|number|null} Environment ID
   */
  getEnvironmentId() {
    return this.environmentId
  }
}

export default BaseEnvironmentClient

// Example usage for specific environments:

// BabyAI Client
export class BabyAIClient extends BaseEnvironmentClient {
  constructor() {
    super('babyai')
  }
  
  // Override methods as needed for BabyAI-specific behavior
}

// SciWorld Client  
export class SciWorldClient extends BaseEnvironmentClient {
  constructor() {
    super('sciworld')
  }
  
  // Override methods as needed for SciWorld-specific behavior
}

// WebArena Client
export class WebArenaClient extends BaseEnvironmentClient {
  constructor() {
    super('webarena')
  }
  
  // Override methods as needed for WebArena-specific behavior
}

// SearchQA Client
export class SearchQAClient extends BaseEnvironmentClient {
  constructor() {
    super('searchqa')
  }
  
  // Override methods as needed for SearchQA-specific behavior
}