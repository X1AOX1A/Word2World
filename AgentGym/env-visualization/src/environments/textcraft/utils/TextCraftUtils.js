/**
 * TextCraft工具函数集
 * 
 * 提供TextCraft环境中常用的数据处理、格式化、验证等工具函数
 */

/**
 * 物品名称和图标映射
 */
export const ITEM_ICONS = {
  // 原材料
  'wood': '🪵', 'oak_log': '🪵', 'log': '🪵', 'oak_wood': '🪵',
  'stone': '🪨', 'cobblestone': '🪨', 'rock': '🪨',
  'iron': '⚙️', 'iron_ingot': '⚙️', 'iron_ore': '⚙️',
  'diamond': '💎', 'diamond_gem': '💎', 'diamond_ore': '💎',
  'coal': '🔥', 'coal_ore': '🔥', 'charcoal': '🔥',
  'gold': '🥇', 'gold_ingot': '🥇', 'gold_ore': '🥇',
  
  // 基础物品
  'stick': '🪄', 'wooden_stick': '🪄',
  'plank': '📋', 'planks': '📋', 'wooden_planks': '📋', 'oak_planks': '📋',
  'string': '🧵', 'rope': '🧵',
  
  // 工具
  'pickaxe': '⛏️', 'wooden_pickaxe': '⛏️', 'stone_pickaxe': '⛏️', 'iron_pickaxe': '⛏️',
  'axe': '🪓', 'wooden_axe': '🪓', 'stone_axe': '🪓', 'iron_axe': '🪓',
  'shovel': '🗿', 'wooden_shovel': '🗿', 'stone_shovel': '🗿', 'iron_shovel': '🗿',
  'hoe': '🔨', 'wooden_hoe': '🔨', 'stone_hoe': '🔨', 'iron_hoe': '🔨',
  
  // 武器
  'sword': '⚔️', 'wooden_sword': '⚔️', 'stone_sword': '⚔️', 'iron_sword': '⚔️',
  'bow': '🏹', 'arrow': '🏹',
  
  // 容器和家具
  'chest': '📦', 'wooden_chest': '📦',
  'door': '🚪', 'wooden_door': '🚪', 'oak_door': '🚪',
  'table': '🪑', 'crafting_table': '🪑', 'workbench': '🪑',
  'bed': '🛏️', 'wooden_bed': '🛏️',
  
  // 交通工具
  'boat': '⛵', 'wooden_boat': '⛵', 'oak_boat': '⛵',
  'minecart': '🚂', 'cart': '🚂',
  
  // 食物
  'bread': '🍞', 'wheat': '🌾',
  'apple': '🍎', 'carrot': '🥕', 'potato': '🥔',
  'meat': '🥩', 'beef': '🥩', 'pork': '🥩', 'chicken': '🍗',
  
  // 其他
  'torch': '🕯️', 'lantern': '🏮',
  'fence': '🚧', 'wooden_fence': '🚧',
  'ladder': '🪜', 'wooden_ladder': '🪜'
}

/**
 * 物品类型分类
 */
export const ITEM_CATEGORIES = {
  MATERIALS: {
    name: 'Materials',
    icon: '🧱',
    color: '#8B4513',
    keywords: ['wood', 'stone', 'iron', 'diamond', 'coal', 'gold', 'ore', 'ingot']
  },
  TOOLS: {
    name: 'Tools',
    icon: '🔧',
    color: '#4682B4',
    keywords: ['pickaxe', 'axe', 'shovel', 'hoe', 'tool']
  },
  WEAPONS: {
    name: 'Weapons',
    icon: '⚔️',
    color: '#DC143C',
    keywords: ['sword', 'bow', 'arrow', 'weapon']
  },
  FURNITURE: {
    name: 'Furniture',
    icon: '🪑',
    color: '#8B4513',
    keywords: ['chest', 'door', 'table', 'bed', 'furniture']
  },
  FOOD: {
    name: 'Food',
    icon: '🍎',
    color: '#32CD32',
    keywords: ['bread', 'apple', 'carrot', 'potato', 'meat', 'beef', 'pork', 'chicken', 'food']
  },
  MISC: {
    name: 'Miscellaneous',
    icon: '📦',
    color: '#696969',
    keywords: []
  }
}

/**
 * 获取物品图标
 * @param {string} itemName - 物品名称
 * @returns {string} 物品图标emoji
 */
export function getItemIcon(itemName) {
  if (!itemName || typeof itemName !== 'string') {
    return '📦'
  }

  const normalizedName = itemName.toLowerCase().replace(/^minecraft:/, '')
  
  // 精确匹配
  if (ITEM_ICONS[normalizedName]) {
    return ITEM_ICONS[normalizedName]
  }
  
  // 模糊匹配
  for (const [key, icon] of Object.entries(ITEM_ICONS)) {
    if (normalizedName.includes(key) || key.includes(normalizedName)) {
      return icon
    }
  }
  
  // 基于类别的回退
  const category = getItemCategory(itemName)
  return ITEM_CATEGORIES[category].icon
}

/**
 * 格式化物品名称显示
 * @param {string} itemName - 原始物品名
 * @returns {string} 格式化后的显示名称
 */
export function formatItemName(itemName) {
  if (!itemName || typeof itemName !== 'string') {
    return 'Unknown Item'
  }

  return itemName
    .replace(/^minecraft:/, '')           // 移除minecraft前缀
    .replace(/_/g, ' ')                   // 下划线转空格
    .replace(/\b\w/g, l => l.toUpperCase()) // 首字母大写
}

/**
 * 获取物品类别
 * @param {string} itemName - 物品名称
 * @returns {string} 物品类别键
 */
export function getItemCategory(itemName) {
  if (!itemName || typeof itemName !== 'string') {
    return 'MISC'
  }

  const normalizedName = itemName.toLowerCase()
  
  for (const [categoryKey, category] of Object.entries(ITEM_CATEGORIES)) {
    if (category.keywords.some(keyword => 
      normalizedName.includes(keyword) || keyword.includes(normalizedName)
    )) {
      return categoryKey
    }
  }
  
  return 'MISC'
}

/**
 * 解析TextCraft观察文本
 * @param {string} observation - 观察文本
 * @returns {Object} 解析后的数据
 */
export function parseObservation(observation) {
  if (!observation || typeof observation !== 'string') {
    return {
      inventory: [],
      commands: [],
      goal: null,
      status: null
    }
  }

  const result = {
    inventory: [],
    commands: [],
    goal: null,
    status: null
  }

  // 解析库存
  const inventoryMatch = observation.match(/Inventory:\s*(.+?)(?=\n|$)/i)
  if (inventoryMatch) {
    const inventoryText = inventoryMatch[1].trim()
    if (!inventoryText.includes("You are not carrying anything")) {
      const itemMatches = Array.from(inventoryText.matchAll(/\[([^\]]+)\]\s*\((\d+)\)/g))
      result.inventory = itemMatches.map(match => ({
        itemName: match[1].trim(),
        count: parseInt(match[2]),
        displayName: formatItemName(match[1].trim()),
        icon: getItemIcon(match[1].trim()),
        category: getItemCategory(match[1].trim())
      }))
    }
  }

  // 解析可用命令
  const commandsMatch = observation.match(/Crafting commands:\s*([\s\S]*?)(?=Goal:|$)/i)
  if (commandsMatch) {
    const commandsText = commandsMatch[1].trim()
    result.commands = commandsText.split('\n')
      .filter(line => line.trim().startsWith('craft'))
      .map(line => line.trim())
  }

  // 解析目标
  const goalMatch = observation.match(/Goal:\s*(.+?)(?:\n|$)/i)
  if (goalMatch) {
    result.goal = goalMatch[1].trim()
  }

  // 解析状态信息
  if (observation.includes('Crafted')) {
    result.status = 'crafted_item'
  } else if (observation.includes('Got')) {
    result.status = 'collected_item'
  } else if (observation.includes('Could not')) {
    result.status = 'action_failed'
  }

  return result
}

/**
 * 验证TextCraft动作格式
 * @param {string} action - 动作字符串
 * @returns {Object} 验证结果
 */
export function validateAction(action) {
  if (!action || typeof action !== 'string') {
    return {
      valid: false,
      type: null,
      error: 'Action must be a non-empty string',
      suggestion: 'Try: inventory, get 1 wood, craft 1 stick using 1 wood'
    }
  }

  const trimmed = action.trim()
  if (trimmed.length === 0) {
    return {
      valid: false,
      type: null,
      error: 'Action cannot be empty',
      suggestion: 'Try: inventory, get 1 wood, craft 1 stick using 1 wood'
    }
  }

  // 动作类型检测
  const actionPatterns = [
    { type: 'inventory', pattern: /^inventory$/i, valid: true },
    { type: 'look', pattern: /^look around$/i, valid: true },
    { type: 'get', pattern: /^get \d+ \w+/i, valid: true },
    { type: 'craft', pattern: /^craft \d+ \w+ using .+/i, valid: true },
    { type: 'examine', pattern: /^examine \w+/i, valid: true },
    { type: 'use', pattern: /^use \w+/i, valid: true }
  ]

  for (const { type, pattern, valid } of actionPatterns) {
    if (pattern.test(trimmed)) {
      return {
        valid,
        type,
        cleaned: trimmed,
        error: null,
        suggestion: null
      }
    }
  }

  return {
    valid: false,
    type: 'unknown',
    cleaned: trimmed,
    error: 'Unknown action format',
    suggestion: 'Try: inventory, get 1 wood, craft 1 stick using 1 wood'
  }
}

/**
 * 生成建议动作
 * @param {Array} inventory - 当前库存
 * @param {Array} commands - 可用命令
 * @param {string} goal - 当前目标
 * @returns {Array} 建议动作列表
 */
export function generateSuggestions(inventory = [], commands = [], goal = '') {
  const suggestions = []

  // 基础动作
  suggestions.push('inventory', 'look around')

  // 基于库存的建议
  if (inventory.length === 0) {
    suggestions.push('get 1 wood', 'get 1 stone')
  } else {
    // 基于现有物品建议合成
    const hasWood = inventory.some(item => item.itemName.toLowerCase().includes('wood'))
    const hasStick = inventory.some(item => item.itemName.toLowerCase().includes('stick'))
    const hasPlanks = inventory.some(item => item.itemName.toLowerCase().includes('plank'))

    if (hasWood && !hasStick) {
      suggestions.push('craft 4 planks using 1 wood')
    }
    if (hasPlanks && hasStick) {
      suggestions.push('craft 1 wooden_pickaxe using 3 planks, 2 stick')
    }
  }

  // 基于可用命令的建议
  suggestions.push(...commands.slice(0, 3))

  // 基于目标的建议
  if (goal) {
    const goalLower = goal.toLowerCase()
    if (goalLower.includes('pickaxe')) {
      suggestions.push('get 1 wood', 'craft 1 stick using 1 wood')
    }
    if (goalLower.includes('house') || goalLower.includes('build')) {
      suggestions.push('get 10 wood', 'craft 40 planks using 10 wood')
    }
  }

  // 去重并限制数量
  return [...new Set(suggestions)].slice(0, 8)
}

/**
 * 计算库存统计信息
 * @param {Array} inventory - 库存数组
 * @returns {Object} 统计信息
 */
export function calculateInventoryStats(inventory = []) {
  const stats = {
    totalItems: 0,
    uniqueTypes: inventory.length,
    categories: {},
    mostCommon: null,
    totalValue: 0
  }

  // 按类别分组统计
  for (const item of inventory) {
    stats.totalItems += item.count
    
    const category = getItemCategory(item.itemName)
    if (!stats.categories[category]) {
      stats.categories[category] = {
        name: ITEM_CATEGORIES[category].name,
        icon: ITEM_CATEGORIES[category].icon,
        count: 0,
        items: 0
      }
    }
    stats.categories[category].count += item.count
    stats.categories[category].items += 1
  }

  // 找出最多的物品
  if (inventory.length > 0) {
    stats.mostCommon = inventory.reduce((max, item) => 
      item.count > max.count ? item : max
    )
  }

  return stats
}

/**
 * 格式化时间显示
 * @param {Date} timestamp - 时间戳
 * @param {boolean} includeDate - 是否包含日期
 * @returns {string} 格式化的时间字符串
 */
export function formatTimestamp(timestamp, includeDate = false) {
  if (!timestamp || !(timestamp instanceof Date)) {
    return 'Unknown time'
  }

  const now = new Date()
  const diff = now - timestamp
  const seconds = Math.floor(diff / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)

  if (seconds < 60) {
    return 'Just now'
  } else if (minutes < 60) {
    return `${minutes}m ago`
  } else if (hours < 24 && !includeDate) {
    return `${hours}h ago`
  } else {
    return includeDate 
      ? timestamp.toLocaleString()
      : timestamp.toLocaleTimeString()
  }
}

/**
 * 生成合成建议
 * @param {string} itemName - 物品名称
 * @param {Array} inventory - 当前库存
 * @returns {Array} 合成建议列表
 */
export function generateCraftingSuggestions(itemName, inventory = []) {
  const suggestions = []
  const normalizedName = itemName.toLowerCase()

  // 基于物品类型生成建议
  if (normalizedName.includes('wood') || normalizedName.includes('log')) {
    suggestions.push(
      `craft 4 planks using 1 ${itemName}`,
      `craft 1 stick using 1 ${itemName}`
    )
  }
  
  if (normalizedName.includes('plank')) {
    const stickCount = inventory.find(item => 
      item.itemName.toLowerCase().includes('stick')
    )?.count || 0
    
    if (stickCount >= 2) {
      suggestions.push(
        `craft 1 wooden_pickaxe using 3 ${itemName}, 2 stick`,
        `craft 1 wooden_axe using 3 ${itemName}, 2 stick`,
        `craft 1 wooden_sword using 2 ${itemName}, 1 stick`
      )
    }
  }
  
  if (normalizedName.includes('stick')) {
    const plankCount = inventory.find(item => 
      item.itemName.toLowerCase().includes('plank')
    )?.count || 0
    
    if (plankCount >= 3) {
      suggestions.push(
        `craft 1 wooden_pickaxe using 3 planks, 2 ${itemName}`,
        `craft 1 wooden_axe using 3 planks, 2 ${itemName}`
      )
    }
  }

  return suggestions.slice(0, 5)
}

/**
 * 颜色工具函数
 */
export const COLORS = {
  primary: '#667eea',
  secondary: '#764ba2',
  success: '#48bb78',
  warning: '#ed8936',
  danger: '#f56565',
  info: '#4299e1',
  dark: '#2d3748',
  light: '#f8fafc',
  muted: '#718096'
}

/**
 * 获取类别颜色
 * @param {string} category - 类别名称
 * @returns {string} 颜色值
 */
export function getCategoryColor(category) {
  return ITEM_CATEGORIES[category]?.color || COLORS.muted
}

/**
 * 深拷贝对象
 * @param {*} obj - 要拷贝的对象
 * @returns {*} 深拷贝后的对象
 */
export function deepClone(obj) {
  if (obj === null || typeof obj !== 'object') {
    return obj
  }
  
  if (obj instanceof Date) {
    return new Date(obj.getTime())
  }
  
  if (obj instanceof Array) {
    return obj.map(item => deepClone(item))
  }
  
  if (typeof obj === 'object') {
    const cloned = {}
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        cloned[key] = deepClone(obj[key])
      }
    }
    return cloned
  }
  
  return obj
}

/**
 * 防抖函数
 * @param {Function} func - 要防抖的函数
 * @param {number} delay - 延迟时间（毫秒）
 * @returns {Function} 防抖后的函数
 */
export function debounce(func, delay) {
  let timeoutId
  return function (...args) {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => func.apply(this, args), delay)
  }
}

/**
 * 节流函数
 * @param {Function} func - 要节流的函数
 * @param {number} delay - 延迟时间（毫秒）
 * @returns {Function} 节流后的函数
 */
export function throttle(func, delay) {
  let lastCall = 0
  return function (...args) {
    const now = Date.now()
    if (now - lastCall >= delay) {
      lastCall = now
      return func.apply(this, args)
    }
  }
}

// 默认导出所有工具函数
export default {
  ITEM_ICONS,
  ITEM_CATEGORIES,
  COLORS,
  getItemIcon,
  formatItemName,
  getItemCategory,
  parseObservation,
  validateAction,
  generateSuggestions,
  calculateInventoryStats,
  formatTimestamp,
  generateCraftingSuggestions,
  getCategoryColor,
  deepClone,
  debounce,
  throttle
}