// Environment configurations  
export const environmentList = [
  {
    id: 'textcraft',
    name: 'TextCraft',
    description: 'A text-based crafting and building environment where agents learn to create items and structures.',
    icon: '⚒️',
    color: '#4CAF50',
    tags: ['Crafting', 'Strategy', 'Sandbox']
  },
  {
    id: 'babyai',
    name: 'BabyAI',
    description: 'Simple gridworld environment for training agents to follow natural language instructions.',
    icon: '🧸',
    color: '#2196F3',
    tags: ['Learning', 'Navigation', 'Simple']
  },
  {
    id: 'sciworld',
    name: 'SciWorld',
    description: 'Science simulation environment for conducting virtual experiments and learning scientific concepts.',
    icon: '🧪',
    color: '#9C27B0',
    tags: ['Science', 'Simulation', 'Physics']
  },
  {
    id: 'webarena',
    name: 'WebArena',
    description: 'Web-based environment for testing agents in realistic web browsing and interaction scenarios.',
    icon: '🌐',
    color: '#FF9800',
    tags: ['Web', 'Browsing', 'Interactive']
  },
  {
    id: 'searchqa',
    name: 'SearchQA',
    description: 'Question-answering environment where agents search for information and provide accurate answers.',
    icon: '🔍',
    color: '#607D8B',
    tags: ['Search', 'Q&A', 'Knowledge']
  }
]

// Environment clients mapping
const environmentClients = {
  textcraft: () => import('./textcraft/client/textcraftClient.js'),
  babyai: () => import('./babyai/client/babyaiClient.js'),
  sciworld: () => import('./sciworld/client/sciworldClient.js'),
  webarena: () => import('./webarena/client/webarenaClient.js'),
  searchqa: () => import('./searchqa/client/searchqaClient.js')
}

// Environment viewers mapping - 移除 defineAsyncComponent 修复无限循环
const environmentViewers = {
  textcraft: () => import('./textcraft/components/TextcraftViewer.vue'),
  babyai: () => import('./babyai/components/BabyAIViewer.vue'),
  sciworld: () => import('./sciworld/components/SciWorldViewer.vue'),
  webarena: () => import('./webarena/components/WebArenaViewer.vue'),
  searchqa: () => import('./searchqa/components/SearchQAViewer.vue')
}

// Environment configuration components mapping (optional)
const environmentConfigComponents = {
  // textcraft: () => import('./textcraft/components/TextcraftConfig.vue'),
  // Other environments can have their own config components if needed
  babyai: () => import('./babyai/components/BabyAIConfig.vue'),
  // sciworld: () => import('./sciworld/components/SciWorldConfig.vue'),
}

/**
 * Get environment client class
 * @param {string} environmentId - Environment identifier
 * @returns {Function|null} Dynamic import function for the client class
 */
export function getEnvironmentClient(environmentId) {
  const client = environmentClients[environmentId]
  if (!client) {
    console.warn(`No client found for environment: ${environmentId}`)
    return null
  }
  return client
}

/**
 * Get environment configuration by ID
 * @param {string} environmentId - Environment identifier
 * @returns {Object|null} Environment configuration object
 */
export function getEnvironment(environmentId) {
  return environmentList.find(env => env.id === environmentId) || null
}

/**
 * Get environment viewer component - 修复无限循环问题
 * @param {string} environmentId - Environment identifier
 * @returns {Function|null} Dynamic import function for the viewer component
 */
export function getEnvironmentViewer(environmentId) {
  // 移除重复的日志，减少控制台噪音
  const viewer = environmentViewers[environmentId]
  if (!viewer) {
    console.warn(`No viewer found for environment: ${environmentId}`)
    return null
  }
  // 直接返回动态导入函数，Vue会处理异步组件
  return viewer
}

/**
 * Get environment configuration component (if exists)
 * @param {string} environmentId - Environment identifier
 * @returns {Function|null} Dynamic import function for the config component
 */
export function getEnvironmentConfigComponent(environmentId) {
  const configComponent = environmentConfigComponents[environmentId]
  if (!configComponent) {
    // Not all environments need config components, this is normal
    return null
  }
  return configComponent
}

/**
 * Check if environment has configuration component
 * @param {string} environmentId - Environment identifier
 * @returns {boolean} True if environment has config component
 */
export function hasEnvironmentConfig(environmentId) {
  return !!environmentConfigComponents[environmentId]
}

/**
 * Get all available environment IDs
 * @returns {string[]} Array of environment IDs
 */
export function getAvailableEnvironments() {
  return environmentList.map(env => env.id)
}

/**
 * Validate if environment ID is supported
 * @param {string} environmentId - Environment identifier
 * @returns {boolean} True if environment is supported
 */
export function isEnvironmentSupported(environmentId) {
  return getAvailableEnvironments().includes(environmentId)
}

// Export default environment for fallback
export const DEFAULT_ENVIRONMENT = 'textcraft'

// Export for backward compatibility
export default {
  environmentList,
  getEnvironment,
  getEnvironmentClient,
  getEnvironmentViewer,
  getEnvironmentConfigComponent,
  hasEnvironmentConfig,
  getAvailableEnvironments,
  isEnvironmentSupported,
  DEFAULT_ENVIRONMENT
}