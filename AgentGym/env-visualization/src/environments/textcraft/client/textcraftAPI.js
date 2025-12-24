/**
 * TextCraft API Client - 完全修复版本
 * 
 * 处理与TextCraft服务器的HTTP通信，修复对象响应处理问题
 */

class TextCraftAPI {
  constructor() {
    this.baseURL = 'http://localhost:36001'
    this.currentEnvironmentId = null
    this.environmentStates = new Map()
    this.inventoryCache = new Map()
    this.interactionHistory = new Map()
  }

  /**
   * 通用请求方法 - 修复响应处理
   * @param {string} endpoint - API端点
   * @param {Object} options - 请求选项
   */
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`
    
    const config = {
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        ...options.headers
      },
      ...options
    }

    console.log(`🌐 TextCraft API Request: ${options.method || 'GET'} ${endpoint}`)
    if (options.body) {
      console.log(`📤 Request body:`, JSON.parse(options.body))
    }

    try {
      const response = await fetch(url, config)
      
      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status} - ${response.statusText}`)
      }
      
      const contentType = response.headers.get('content-type')
      let data
      
      if (contentType && contentType.includes('application/json')) {
        data = await response.json()
      } else {
        data = await response.text()
      }

      console.log(`✅ TextCraft API Response:`, data)
      console.log(`📋 Response type:`, typeof data)
      
      return data
    } catch (error) {
      console.error(`❌ TextCraft API Error for ${url}:`, error)
      throw error
    }
  }

  /**
   * 处理观察数据 - 统一方法
   * @param {any} responseData - 原始响应数据
   * @returns {string} 处理后的观察字符串
   */
  processObservationData(responseData) {
    let observationData = responseData
    
    // 如果响应是对象，提取观察数据
    if (typeof responseData === 'object' && responseData !== null) {
      if (responseData.observation !== undefined) {
        observationData = responseData.observation
      } else if (responseData.state !== undefined) {
        observationData = responseData.state
      } else if (responseData.data !== undefined) {
        observationData = responseData.data
      } else if (responseData.message !== undefined) {
        observationData = responseData.message
      } else {
        // 如果对象没有明确的观察字段，序列化整个对象
        observationData = JSON.stringify(responseData, null, 2)
      }
    }
    
    // 确保观察数据是字符串
    if (typeof observationData !== 'string') {
      observationData = String(observationData)
    }
    
    return observationData
  }

  /**
   * 测试与TextCraft服务器的连接 - 修复版本
   */
  async testConnection() {
    try {
      console.log('🔍 Testing connection to TextCraft server...')
      const response = await this.request('/')
      
      const message = this.processObservationData(response)
      console.log('✅ TextCraft server connection successful:', message)
      
      return {
        success: true,
        message: message,
        connected: true
      }
    } catch (error) {
      console.error('❌ TextCraft server connection failed:', error)
      return {
        success: false,
        error: error.message,
        connected: false,
        details: error
      }
    }
  }

  /**
   * 创建新的TextCraft环境 - 修复版本
   * @param {Object} config - 环境配置
   */
  async createEnvironment(config = {}) {
    try {
      const requestBody = {
        commands: config.commands || null,
        goal: config.goal || null
      }

      console.log(`🔨 Creating TextCraft environment with config:`, requestBody)
      
      const response = await this.request('/create', {
        method: 'POST',
        body: JSON.stringify(requestBody)
      })

      console.log(`📋 Create environment raw response:`, response)

      // 检查服务器返回的错误
      if (response && response.error) {
        throw new Error(`Server error: ${response.error}`)
      }

      // 提取环境ID - 支持多种格式
      let environmentId = null
      if (typeof response === 'object' && response !== null) {
        environmentId = response.id || response.environmentId || response.env_id
      } else if (typeof response === 'number') {
        environmentId = response
      }

      if (environmentId === undefined || environmentId === null) {
        throw new Error('No environment ID returned from create endpoint')
      }

      // 确保environmentId是数字类型
      const numericId = parseInt(environmentId)
      if (isNaN(numericId)) {
        throw new Error(`Invalid environment ID received: ${environmentId}`)
      }

      // 处理观察数据
      let observationData = ''
      if (typeof response === 'object' && response !== null) {
        observationData = this.processObservationData(response)
      }

      // 缓存环境状态
      this.currentEnvironmentId = numericId
      this.environmentStates.set(numericId, {
        id: numericId,
        goal: config.goal,
        commands: config.commands,
        created: new Date().toISOString(),
        lastObservation: observationData,
        inventory: []
      })

      // 初始化交互历史
      this.interactionHistory.set(numericId, [])

      console.log(`✅ TextCraft environment created successfully! ID: ${numericId}`)

      return {
        success: true,
        data: {
          id: numericId,
          environmentId: numericId,
          observation: observationData,
          reward: response.reward || 0,
          done: response.done || false
        }
      }
    } catch (error) {
      console.error(`❌ Failed to create TextCraft environment:`, error)
      return {
        success: false,
        error: error.message
      }
    }
  }

  /**
   * 执行动作 - 修复版本
   * @param {number} environmentId - 环境ID
   * @param {string} action - 动作命令
   */
  async executeAction(environmentId, action) {
    try {
      const numericId = parseInt(environmentId)
      if (isNaN(numericId)) {
        throw new Error(`Invalid environment ID: ${environmentId}`)
      }

      const cleanedAction = String(action).trim()
      if (!cleanedAction) {
        throw new Error('Action cannot be empty')
      }

      console.log(`🎮 Executing TextCraft action: "${cleanedAction}" in environment ${numericId}`)

      const requestBody = {
        id: numericId,
        action: cleanedAction
      }

      const response = await this.request('/step', {
        method: 'POST',
        body: JSON.stringify(requestBody)
      })

      console.log(`📋 Step raw response:`, response)

      // 处理观察数据
      const observationData = this.processObservationData(response)

      // 提取奖励和完成状态
      let reward = 0
      let done = false
      
      if (typeof response === 'object' && response !== null) {
        reward = response.reward || 0
        done = response.done || false
      }

      // 更新环境状态
      if (this.environmentStates.has(numericId)) {
        const state = this.environmentStates.get(numericId)
        state.lastObservation = observationData
        state.lastAction = cleanedAction
        state.lastResponse = response
        this.environmentStates.set(numericId, state)
      }

      // 更新库存缓存（如果是inventory动作）
      if (cleanedAction.toLowerCase() === 'inventory') {
        this.updateInventoryCache(numericId, observationData)
      }

      // 添加到交互历史
      this.addToHistory(numericId, cleanedAction, observationData)

      console.log(`✅ TextCraft action executed successfully:`, {
        action: cleanedAction,
        reward: reward,
        done: done,
        observationLength: observationData.length
      })

      return {
        success: true,
        data: {
          observation: observationData,
          reward: reward,
          done: done,
          terminated: done,
          info: response.info || {}
        }
      }
    } catch (error) {
      console.error(`❌ TextCraft step failed:`, error)
      return {
        success: false,
        error: error.message,
        data: {
          observation: `Error: ${error.message}`,
          reward: 0,
          done: false
        }
      }
    }
  }

  /**
   * 获取当前观察 - 修复版本
   * @param {number} environmentId - 环境ID
   */
  async getObservation(environmentId) {
    try {
      const numericId = parseInt(environmentId)
      if (isNaN(numericId)) {
        throw new Error(`Invalid environment ID: ${environmentId}`)
      }

      console.log(`👁️ Getting TextCraft observation for environment ${numericId}`)
      
      const response = await this.request(`/observation?id=${numericId}`)
      
      console.log(`📋 Raw observation response:`, response)
      
      // 处理观察数据
      const observationData = this.processObservationData(response)
      
      console.log(`📋 Processed observation:`, observationData.substring(0, 200) + '...')
      
      // 更新环境状态缓存
      if (this.environmentStates.has(numericId)) {
        const state = this.environmentStates.get(numericId)
        state.lastObservation = observationData
        this.environmentStates.set(numericId, state)
      }

      console.log(`✅ TextCraft observation retrieved successfully`)
      
      return {
        success: true,
        data: observationData
      }
    } catch (error) {
      console.error(`❌ Failed to get TextCraft observation:`, error)
      return {
        success: false,
        error: error.message,
        data: `Error: ${error.message}`
      }
    }
  }

  /**
   * 重置环境 - 修复版本
   * @param {number} environmentId - 环境ID
   * @param {number} dataIdx - 数据索引（默认0）
   */
  async resetEnvironment(environmentId, dataIdx = 0) {
    try {
      const numericId = parseInt(environmentId)
      if (isNaN(numericId)) {
        throw new Error(`Invalid environment ID: ${environmentId}`)
      }

      console.log(`🔄 Resetting TextCraft environment ${numericId} with data_idx=${dataIdx}`)
      
      const requestBody = {
        id: numericId,
        data_idx: parseInt(dataIdx)
      }
      
      const response = await this.request('/reset', {
        method: 'POST',
        body: JSON.stringify(requestBody)
      })

      console.log(`📋 Reset raw response:`, response)

      // 处理观察数据
      const observationData = this.processObservationData(response)

      // 提取奖励和完成状态
      let reward = 0
      let done = false
      
      if (typeof response === 'object' && response !== null) {
        reward = response.reward || 0
        done = response.done || false
      }

      // 清理相关缓存
      this.inventoryCache.delete(numericId)
      this.interactionHistory.set(numericId, [])
      
      // 更新环境状态
      if (this.environmentStates.has(numericId)) {
        const state = this.environmentStates.get(numericId)
        state.lastObservation = observationData
        state.lastAction = null
        state.lastResponse = response
        this.environmentStates.set(numericId, state)
      }

      console.log(`✅ TextCraft environment reset successfully`)

      return {
        success: true,
        data: {
          observation: observationData,
          reward: reward,
          done: done
        }
      }
    } catch (error) {
      console.error(`❌ TextCraft reset failed:`, error)
      return {
        success: false,
        error: error.message
      }
    }
  }

  /**
   * 更新库存缓存
   * @param {number} environmentId - 环境ID
   * @param {string} observation - 观察文本
   */
  updateInventoryCache(environmentId, observation) {
    if (!observation || typeof observation !== 'string') {
      return
    }

    const inventory = this.parseInventoryFromObservation(observation)
    this.inventoryCache.set(environmentId, inventory)
    
    console.log(`🎒 Updated inventory cache for environment ${environmentId}:`, inventory)
  }

  /**
   * 从观察文本中解析库存
   * @param {string} observation - 观察文本
   * @returns {Array} 库存物品数组
   */
  parseInventoryFromObservation(observation) {
    const inventory = []
    
    try {
      // 查找库存部分
      const inventoryMatch = observation.match(/Inventory:\s*(.+?)(?:\n|$)/i)
      
      if (inventoryMatch) {
        const inventoryText = inventoryMatch[1].trim()
        
        if (inventoryText.toLowerCase().includes('not carrying') || 
            inventoryText.toLowerCase().includes('empty')) {
          return inventory
        }
        
        // 解析物品格式: [item_name] (count)
        const itemMatches = inventoryText.matchAll(/\[([^\]]+)\]\s*\((\d+)\)/g)
        
        for (const match of itemMatches) {
          const itemName = match[1].trim()
          const count = parseInt(match[2], 10) || 0
          
          inventory.push({
            itemName: itemName,
            name: itemName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
            count: count,
            image: `/assets/minecraft/textures/item/${itemName.replace(/ /g, '_').toLowerCase()}.png`
          })
        }
      }
    } catch (error) {
      console.error('❌ Error parsing inventory:', error)
    }
    
    return inventory
  }

  /**
   * 获取缓存的库存
   * @param {number} environmentId - 环境ID
   * @returns {Array} 库存数组
   */
  getCachedInventory(environmentId) {
    return this.inventoryCache.get(environmentId) || []
  }

  /**
   * 添加到交互历史
   * @param {number} environmentId - 环境ID
   * @param {string} action - 动作
   * @param {string} observation - 观察
   */
  addToHistory(environmentId, action, observation) {
    if (!this.interactionHistory.has(environmentId)) {
      this.interactionHistory.set(environmentId, [])
    }
    
    const history = this.interactionHistory.get(environmentId)
    history.push({
      timestamp: new Date().toISOString(),
      action: action,
      observation: observation
    })
    
    // 限制历史记录长度
    if (history.length > 100) {
      history.splice(0, history.length - 100)
    }
  }

  /**
   * 获取交互历史
   * @param {number} environmentId - 环境ID
   * @returns {Array} 历史记录数组
   */
  getHistory(environmentId) {
    return this.interactionHistory.get(environmentId) || []
  }

  /**
   * 设置当前环境ID
   * @param {number} environmentId - 环境ID
   */
  setCurrentEnvironmentId(environmentId) {
    this.currentEnvironmentId = parseInt(environmentId)
  }

  /**
   * 获取当前环境ID
   * @returns {number|null} 当前环境ID
   */
  getCurrentEnvironmentId() {
    return this.currentEnvironmentId
  }

  /**
   * 获取环境状态
   * @param {number} environmentId - 环境ID
   * @returns {Object|null} 环境状态
   */
  getEnvironmentState(environmentId) {
    return this.environmentStates.get(environmentId) || null
  }
}

// 创建单例实例
const textCraftAPI = new TextCraftAPI()

export default textCraftAPI