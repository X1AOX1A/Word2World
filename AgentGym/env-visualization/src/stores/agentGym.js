// stores/agentGym.js - Pinia状态管理
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { AgentGymAPI } from '../api/agentGymAPI.js'

export const useAgentGymStore = defineStore('agentGym', () => {

  const api = new AgentGymAPI()
  const currentEnvironment = ref(null)
  const environmentId = ref(null)
  const isConnected = ref(false)
  const isAutoRunning = ref(false)
  const isLoading = ref(false)
  const error = ref(null)
  const success = ref(null)
  

  const environmentState = ref({
    observation: '',
    reward: 0,
    done: false,
    info: {}
  })
  

  const interactionHistory = ref([])
  

  const environmentData = ref({
    textcraft: {
      inventory: [],
      goal: '',
      crafting_table: []
    },
    babyai: {
      mission: '',
      grid: null,
      agent_pos: [0, 0],
      agent_dir: 0
    },
    sciworld: {
      room: '',
      inventory: [],
      task_description: '',
      available_actions: []
    },
    webarena: {
      url: '',
      page_content: '',
      screenshot: null
    },
    searchqa: {
      question: '',
      search_results: [],
      answer: ''
    }
  })
  

  const currentEnvData = computed(() => {
    if (!currentEnvironment.value) return null
    return environmentData.value[currentEnvironment.value.id] || {}
  })
  
  const totalReward = computed(() => {
    return interactionHistory.value
      .filter(item => item.type === 'response')
      .reduce((total, item) => total + (item.reward || 0), 0)
  })
  
  const episodeCount = computed(() => {
    return interactionHistory.value
      .filter(item => item.type === 'response' && item.done)
      .length
  })
  
  const setCurrentEnvironment = (env) => {
    currentEnvironment.value = env
    api.setEnvironmentType(env.id)
    clearError()
  }
  
  const createEnvironment = async (options = {}) => {
    if (!currentEnvironment.value) {
      throw new Error('No environment selected')
    }
    
    isLoading.value = true
    error.value = null
    
    try {
      const client = api.getEnvironmentClient()
      const result = await client.createEnvironment(options)
      
      if (result.success) {
        environmentId.value = result.data.id || result.data.env_idx
        isConnected.value = true
        
        await refreshObservation()
        
        success.value = `Environment ${currentEnvironment.value.name} created successfully!`
        setTimeout(() => success.value = null, 3000)
      } else {
        throw new Error(result.error)
      }
    } catch (err) {
      error.value = `Failed to create environment: ${err.message}`
    } finally {
      isLoading.value = false
    }
  }
  
  const resetEnvironment = async () => {
    if (!environmentId.value) return
    
    isLoading.value = true
    
    try {
      const client = api.getEnvironmentClient()
      const result = await client.reset(environmentId.value)
      
      if (result.success) {

        interactionHistory.value = []
        environmentState.value = {
          observation: '',
          reward: 0,
          done: false,
          info: {}
        }
        

        resetEnvironmentData()

        await refreshObservation()
        
        success.value = 'Environment reset successfully!'
        setTimeout(() => success.value = null, 3000)
      } else {
        throw new Error(result.error)
      }
    } catch (err) {
      error.value = `Failed to reset environment: ${err.message}`
    } finally {
      isLoading.value = false
    }
  }
  
  const sendAction = async (action) => {
    if (!environmentId.value || !action) return
    
    isLoading.value = true
    
    try {
      const client = api.getEnvironmentClient()
      const result = await client.step(environmentId.value, action)
      
      if (result.success) {

        addToHistory('action', action)
        addToHistory('response', result.data.observation, result.data.reward, result.data.done)
        

        environmentState.value = {
          observation: result.data.observation,
          reward: result.data.reward,
          done: result.data.done,
          info: result.data.info || {}
        }
        

        updateEnvironmentData(result.data)
        
        if (result.data.done) {
          success.value = 'Episode completed!'
          setTimeout(() => success.value = null, 3000)
        }
      } else {
        throw new Error(result.error)
      }
    } catch (err) {
      error.value = `Failed to send action: ${err.message}`
    } finally {
      isLoading.value = false
    }
  }
  
  const startAutoRun = async (maxRounds = 20) => {
    if (isAutoRunning.value || !environmentId.value) return
    
    isAutoRunning.value = true
    
    try {
      const aiAgent = api.getAIAgent()
      await aiAgent.initialize()
      

      aiAgent.initializeConversation(
        environmentId.value,
        currentEnvironment.value.id,
        environmentState.value.observation
      )
      
      let rounds = 0
      
      while (isAutoRunning.value && rounds < maxRounds && !environmentState.value.done) {
        try {

          const aiResponse = await aiAgent.generateResponse(
            environmentId.value,
            environmentState.value.observation
          )
          

          await sendAction(aiResponse.action)
          
          rounds++

          await new Promise(resolve => setTimeout(resolve, 1500))
          
        } catch (err) {
          console.error('Auto-run error:', err)
          break
        }
      }
      
      success.value = `Auto-run completed after ${rounds} rounds`
      setTimeout(() => success.value = null, 3000)
      
    } catch (err) {
      error.value = `Auto-run failed: ${err.message}`
    } finally {
      isAutoRunning.value = false
    }
  }
  
  const stopAutoRun = () => {
    isAutoRunning.value = false
    success.value = 'Auto-run stopped'
    setTimeout(() => success.value = null, 3000)
  }
  
  const refreshObservation = async () => {
    if (!environmentId.value) return
    
    try {
      const client = api.getEnvironmentClient()
      const result = await client.getObservation(environmentId.value)
      
      if (result.success) {
        environmentState.value.observation = result.data.observation || result.data
        updateEnvironmentData({ observation: environmentState.value.observation })
      }
    } catch (err) {
      console.warn('Failed to refresh observation:', err)
    }
  }
  
  const renderEnvironment = async () => {
    if (!environmentId.value || !currentEnvironment.value) return
    
    try {
      const client = api.getEnvironmentClient()
      const result = await client.render(environmentId.value)
      
      if (result.success && result.data.image) {
        return `data:image/png;base64,${result.data.image}`
      }
    } catch (err) {
      console.warn('Failed to render environment:', err)
    }
    
    return null
  }
  
  const getEnvironmentInfo = async (infoType) => {
    if (!environmentId.value) return null
    
    try {
      const client = api.getEnvironmentClient()
      const result = await client.getEnvironmentSpecificInfo(environmentId.value, infoType)
      
      if (result.success) {
        return result.data
      }
    } catch (err) {
      console.warn(`Failed to get ${infoType}:`, err)
    }
    
    return null
  }
  

  const addToHistory = (type, content, reward = 0, done = false) => {
    interactionHistory.value.unshift({
      id: Date.now() + Math.random(),
      type,
      content,
      reward,
      done,
      timestamp: Date.now()
    })
    

    if (interactionHistory.value.length > 100) {
      interactionHistory.value = interactionHistory.value.slice(0, 100)
    }
  }
  
  const updateEnvironmentData = (data) => {
    if (!currentEnvironment.value) return
    
    const envId = currentEnvironment.value.id
    const envData = environmentData.value[envId]
    
    switch (envId) {
      case 'textcraft':
        updateTextCraftData(envData, data)
        break
      case 'babyai':
        updateBabyAIData(envData, data)
        break
      case 'sciworld':
        updateSciWorldData(envData, data)
        break
      case 'webarena':
        updateWebArenaData(envData, data)
        break
      case 'searchqa':
        updateSearchQAData(envData, data)
        break
    }
  }
  
  const updateTextCraftData = (envData, data) => {
    if (data.observation) {
      const inventoryMatch = data.observation.match(/inventory[:\s]+([^.\n]+)/i)
      if (inventoryMatch) {
        envData.inventory = inventoryMatch[1]
          .split(',')
          .map(item => item.trim())
          .filter(item => item && item !== 'nothing')
      }
    }
  }
  
  const updateBabyAIData = (envData, data) => {
    if (data.mission) envData.mission = data.mission
    if (data.grid) envData.grid = data.grid
    if (data.agent_pos) envData.agent_pos = data.agent_pos
    if (data.agent_dir !== undefined) envData.agent_dir = data.agent_dir
  }
  
  const updateSciWorldData = (envData, data) => {
    if (data.observation) {
      const roomMatch = data.observation.match(/(?:in the|room)[:\s]+([^.\n]+)/i)
      if (roomMatch) envData.room = roomMatch[1]
    }
  }
  
  const updateWebArenaData = (envData, data) => {
    if (data.url) envData.url = data.url
    if (data.page_content) envData.page_content = data.page_content
    if (data.screenshot) envData.screenshot = data.screenshot
  }
  
  const updateSearchQAData = (envData, data) => {
    if (data.observation) {
      const questionMatch = data.observation.match(/question[:\s]+([^?\n]+\?)/i)
      if (questionMatch) envData.question = questionMatch[1]
      
      const answerMatch = data.observation.match(/answer[:\s]+([^.\n]+)/i)
      if (answerMatch) envData.answer = answerMatch[1]
    }
  }
  
  const resetEnvironmentData = () => {
    if (!currentEnvironment.value) return
    
    const envId = currentEnvironment.value.id
    const defaultData = {
      textcraft: { inventory: [], goal: '', crafting_table: [] },
      babyai: { mission: '', grid: null, agent_pos: [0, 0], agent_dir: 0 },
      sciworld: { room: '', inventory: [], task_description: '', available_actions: [] },
      webarena: { url: '', page_content: '', screenshot: null },
      searchqa: { question: '', search_results: [], answer: '' }
    }
    
    environmentData.value[envId] = { ...defaultData[envId] }
  }
  
  const clearError = () => {
    error.value = null
  }
  
  const clearSuccess = () => {
    success.value = null
  }
  
  const closeEnvironment = () => {
    environmentId.value = null
    isConnected.value = false
    isAutoRunning.value = false
    interactionHistory.value = []
    environmentState.value = {
      observation: '',
      reward: 0,
      done: false,
      info: {}
    }
    resetEnvironmentData()
    clearError()
    clearSuccess()
  }
  
  return {
    currentEnvironment,
    environmentId,
    isConnected,
    isAutoRunning,
    isLoading,
    error,
    success,
    environmentState,
    interactionHistory,
    environmentData,
    
    currentEnvData,
    totalReward,
    episodeCount,
    
    setCurrentEnvironment,
    createEnvironment,
    resetEnvironment,
    sendAction,
    startAutoRun,
    stopAutoRun,
    refreshObservation,
    renderEnvironment,
    getEnvironmentInfo,
    closeEnvironment,
    clearError,
    clearSuccess,
    
    api
  }
})
