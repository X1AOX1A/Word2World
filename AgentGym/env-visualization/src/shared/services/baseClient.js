/**
 * BaseEnvClient - Abstract client class for AgentGym environments
 * 
 * This class defines the interface that all environment clients must implement.
 * It provides the standard methods for interacting with environment servers.
 */

class BaseEnvClient {
  /**
   * Create a new environment client
   * @param {string} baseUrl - Base URL for API requests
   */
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.environmentId = null;
  }

  /**
   * Test connection to the environment server
   * @returns {Promise<Object>} Connection test result
   */
  async testConnection() {
    throw new Error("Not implemented: testConnection must be implemented by subclass");
  }

  /**
   * Return the total size/count of available environments
   * @returns {Promise<number>} Number of available environments
   */
  async len() {
    throw new Error("Not implemented: len must be implemented by subclass");
  }

  /**
   * Get current observation from the environment
   * @param {string|number} envId - Environment ID
   * @returns {Promise<Object>} Observation result with success flag and data
   */
  async observe(envId) {
    throw new Error("Not implemented: observe must be implemented by subclass");
  }

  /**
   * Execute an action in the environment
   * @param {string|number} envId - Environment ID
   * @param {string} action - Action to execute
   * @returns {Promise<Object>} Step result with success flag and data (state, reward, done)
   */
  async step(envId, action) {
    throw new Error("Not implemented: step must be implemented by subclass");
  }

  /**
   * Reset the environment
   * @param {string|number} envId - Environment ID
   * @param {number} idx - Data index to reset to
   * @returns {Promise<Object>} Reset result with success flag and data
   */
  async reset(envId, idx = 0) {
    throw new Error("Not implemented: reset must be implemented by subclass");
  }

  /**
   * Create a new environment instance
   * @param {Object} config - Environment configuration
   * @returns {Promise<Object>} Creation result with success flag and data
   */
  async createEnvironment(config = {}) {
    throw new Error("Not implemented: createEnvironment must be implemented by subclass");
  }

  /**
   * Get current observation from the environment
   * @param {string|number} envId - Environment ID
   * @returns {Promise<Object>} Observation with success flag and data
   */
  async getObservation(envId) {
    throw new Error("Not implemented: getObservation must be implemented by subclass");
  }

  /**
   * Helper method to make HTTP requests
   * @param {string} endpoint - API endpoint
   * @param {Object} options - Request options
   * @returns {Promise<any>} Response data
   */
  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    
    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeout || 30000); // 30 second default timeout
    
    const config = {
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        ...options.headers
      },
      signal: controller.signal,
      ...options
    };

    try {
      console.log(`🔗 Making request to: ${url}`, config.method || 'GET');
      const response = await fetch(url, config);
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status} - ${response.statusText}`);
      }
      
      const contentType = response.headers.get('content-type');
      
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      } else {
        return await response.text();
      }
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error.name === 'AbortError') {
        const timeoutError = new Error(`Request timeout after ${(options.timeout || 30000) / 1000} seconds`);
        timeoutError.name = 'TimeoutError';
        console.error(`⏰ Request timeout for ${url}:`, timeoutError);
        throw timeoutError;
      }
      
      console.error(`❌ API Request Error for ${url}:`, error);
      throw error;
    }
  }

  /**
   * Process an observation to ensure it's a string
   * @param {any} data - Observation data to process
   * @returns {string} Processed observation string
   */
  processObservation(data) {
    if (typeof data === 'string') {
      return data;
    }
    
    if (data === null || data === undefined) {
      return '';
    }
    
    if (typeof data === 'object') {
      if (data.observation) {
        return this.processObservation(data.observation);
      }
      if (data.state) {
        return this.processObservation(data.state);
      }
      if (data.data) {
        return this.processObservation(data.data);
      }
      return JSON.stringify(data, null, 2);
    }
    
    return String(data);
  }
}

export default BaseEnvClient; 