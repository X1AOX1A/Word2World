/**
 * Environment Manager
 * 
 * Manages environment clients and provides a unified interface
 * for interacting with different environment types
 */

import { getEnvironmentClient, isEnvironmentSupported } from '../../environments/index.js'

class EnvironmentManager {
  constructor() {
    this.clients = new Map()
    this.loadedClients = new Map()
    this.loadingPromises = new Map() 
    this.lastErrors = new Map() 
  }

  /**
   * Get client for a specific environment
   * @param {string} environmentType - Environment type identifier
   * @returns {Object|null} Client instance or null if not found
   */
  async getClient(environmentType) {
    if (!environmentType || !isEnvironmentSupported(environmentType)) {
      console.warn(`Unsupported environment type: ${environmentType}`)
      return null
    }

    // Return cached client if exists
    if (this.loadedClients.has(environmentType)) {
      const cachedClient = this.loadedClients.get(environmentType)
      console.log(`✅ Using cached client for: ${environmentType}`)
      return cachedClient
    }

    // If already loading, return the same promise to prevent duplicate loading
    if (this.loadingPromises.has(environmentType)) {
      console.log(`⏳ Client loading in progress for: ${environmentType}`)
      return await this.loadingPromises.get(environmentType)
    }

    // Create loading promise
    const loadingPromise = this.loadClientInternal(environmentType)
    this.loadingPromises.set(environmentType, loadingPromise)

    try {
      const client = await loadingPromise
      return client
    } finally {
      // Clean up loading promise regardless of success/failure
      this.loadingPromises.delete(environmentType)
    }
  }

  /**
   * Internal method to load client
   * @param {string} environmentType - Environment type identifier
   * @returns {Object|null} Client instance or null if failed
   */
  async loadClientInternal(environmentType) {
    try {
      console.log(`🔧 Loading client for: ${environmentType}`)
      
      // Get client class dynamically
      const clientImport = getEnvironmentClient(environmentType)
      if (!clientImport) {
        throw new Error(`No client available for environment: ${environmentType}`)
      }

      // Load client class
      const ClientModule = await clientImport()
      const ClientClass = ClientModule.default || ClientModule

      if (!ClientClass || typeof ClientClass !== 'function') {
        throw new Error(`Invalid client class for environment: ${environmentType}`)
      }

      // Create client instance
      const clientInstance = new ClientClass()
      
      // Validate client instance
      if (!clientInstance || typeof clientInstance !== 'object') {
        throw new Error(`Failed to create client instance for: ${environmentType}`)
      }
      
      // Cache the client
      this.loadedClients.set(environmentType, clientInstance)
      
      // Clear any previous errors
      this.lastErrors.delete(environmentType)
      
      console.log(`✅ Client loaded and cached for: ${environmentType}`)
      return clientInstance

    } catch (error) {
      console.error(`❌ Failed to load client for ${environmentType}:`, error)
      
      // Store error for debugging
      this.lastErrors.set(environmentType, {
        error: error.message,
        timestamp: new Date().toISOString()
      })
      
      return null
    }
  }

  /**
   * Preload client for faster access
   * @param {string} environmentType - Environment type identifier
   */
  async preloadClient(environmentType) {
    console.log(`🚀 Preloading client for: ${environmentType}`)
    await this.getClient(environmentType)
  }

  /**
   * Clear cached client for an environment
   * @param {string} environmentType - Environment type identifier
   */
  clearClient(environmentType) {
    if (this.loadedClients.has(environmentType)) {
      const client = this.loadedClients.get(environmentType)
      
      // Close client if it has a close method
      if (client && typeof client.close === 'function') {
        try {
          client.close()
          console.log(`🔌 Client connection closed for: ${environmentType}`)
        } catch (error) {
          console.warn(`⚠️ Error closing client for ${environmentType}:`, error)
        }
      }
      
      this.loadedClients.delete(environmentType)
      this.lastErrors.delete(environmentType)
      console.log(`🧹 Client cleared for environment: ${environmentType}`)
    }
    
    // Also clear any pending loading promises
    if (this.loadingPromises.has(environmentType)) {
      this.loadingPromises.delete(environmentType)
      console.log(`🧹 Cleared loading promise for: ${environmentType}`)
    }
  }

  /**
   * Clear all cached clients
   */
  clearAllClients() {
    console.log(`🧹 Clearing all cached clients...`)
    for (const environmentType of this.loadedClients.keys()) {
      this.clearClient(environmentType)
    }
    
    // Clear any remaining loading promises
    this.loadingPromises.clear()
    console.log(`✅ All clients cleared`)
  }

  /**
   * Force reload client (clear cache and reload)
   * @param {string} environmentType - Environment type identifier
   * @returns {Object|null} New client instance
   */
  async reloadClient(environmentType) {
    console.log(`🔄 Force reloading client for: ${environmentType}`)
    this.clearClient(environmentType)
    return await this.getClient(environmentType)
  }

  /**
   * Get list of loaded clients
   * @returns {string[]} Array of loaded environment types
   */
  getLoadedClients() {
    return Array.from(this.loadedClients.keys())
  }

  /**
   * Check if client is loaded for environment
   * @param {string} environmentType - Environment type identifier
   * @returns {boolean} True if client is loaded
   */
  isClientLoaded(environmentType) {
    return this.loadedClients.has(environmentType)
  }

  /**
   * Check if client is currently loading
   * @param {string} environmentType - Environment type identifier
   * @returns {boolean} True if client is loading
   */
  isClientLoading(environmentType) {
    return this.loadingPromises.has(environmentType)
  }

  /**
   * Test connection for a specific environment
   * @param {string} environmentType - Environment type identifier
   * @returns {Promise<boolean>} Connection test result
   */
  async testConnection(environmentType) {
    try {
      const client = await this.getClient(environmentType)
      if (!client || typeof client.testConnection !== 'function') {
        console.warn(`⚠️ Client for ${environmentType} does not support connection testing`)
        return false
      }
      
      console.log(`🔍 Testing connection for: ${environmentType}`)
      const result = await client.testConnection()
      const isConnected = result && (result.success || false)
      
      console.log(`${isConnected ? '✅' : '❌'} Connection test for ${environmentType}: ${isConnected}`)
      return isConnected
    } catch (error) {
      console.error(`❌ Connection test failed for ${environmentType}:`, error)
      return false
    }
  }

  /**
   * Get client status information
   * @param {string} environmentType - Environment type identifier
   * @returns {Object} Status information
   */
  async getClientStatus(environmentType) {
    const isLoaded = this.isClientLoaded(environmentType)
    const isLoading = this.isClientLoading(environmentType)
    const lastError = this.lastErrors.get(environmentType)
    
    let isConnected = false
    if (isLoaded && !isLoading) {
      try {
        isConnected = await this.testConnection(environmentType)
      } catch (error) {
        console.error(`Error testing connection for status: ${error.message}`)
      }
    }
    
    return {
      environmentType,
      isLoaded,
      isLoading,
      isConnected,
      lastError,
      lastChecked: new Date().toISOString()
    }
  }

  /**
   * Get status for all environments
   * @returns {Promise<Object[]>} Array of status objects
   */
  async getAllClientsStatus() {
    const statuses = []
    
    // Check loaded clients
    for (const environmentType of this.loadedClients.keys()) {
      const status = await this.getClientStatus(environmentType)
      statuses.push(status)
    }
    
    // Also check for any clients that had errors but aren't loaded
    for (const environmentType of this.lastErrors.keys()) {
      if (!this.loadedClients.has(environmentType)) {
        const status = await this.getClientStatus(environmentType)
        statuses.push(status)
      }
    }
    
    return statuses
  }

  /**
   * Get debugging information
   * @returns {Object} Debug information
   */
  getDebugInfo() {
    return {
      loadedClients: Array.from(this.loadedClients.keys()),
      loadingPromises: Array.from(this.loadingPromises.keys()),
      lastErrors: Object.fromEntries(this.lastErrors),
      timestamp: new Date().toISOString()
    }
  }

  /**
   * Health check for all loaded clients
   * @returns {Promise<Object>} Health check results
   */
  async healthCheck() {
    const results = {
      timestamp: new Date().toISOString(),
      totalClients: this.loadedClients.size,
      healthyClients: 0,
      unhealthyClients: 0,
      details: []
    }

    for (const [environmentType, client] of this.loadedClients) {
      try {
        const isHealthy = await this.testConnection(environmentType)
        if (isHealthy) {
          results.healthyClients++
        } else {
          results.unhealthyClients++
        }
        
        results.details.push({
          environmentType,
          isHealthy,
          lastChecked: new Date().toISOString()
        })
      } catch (error) {
        results.unhealthyClients++
        results.details.push({
          environmentType,
          isHealthy: false,
          error: error.message,
          lastChecked: new Date().toISOString()
        })
      }
    }

    return results
  }
}

// Create singleton instance
const environmentManager = new EnvironmentManager()

// Clean up on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    console.log('🧹 Page unloading, cleaning up environment manager...')
    environmentManager.clearAllClients()
  })
  
  // Also clean up on page hide (mobile browsers)
  window.addEventListener('pagehide', () => {
    console.log('🧹 Page hiding, cleaning up environment manager...')
    environmentManager.clearAllClients()
  })
}

export default environmentManager