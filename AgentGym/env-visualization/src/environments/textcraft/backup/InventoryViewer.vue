<template>
  <div class="inventory-viewer">
    <div class="inventory-header">
      <h4>🎒 Inventory</h4>
      <div class="inventory-count">
        <span class="count-badge">{{ totalItems }}</span>
        <span class="count-text">items</span>
      </div>
    </div>
    
    <!-- 空库存状态 -->
    <div v-if="inventoryItems.length === 0" class="empty-inventory">
      <div class="empty-icon">📦</div>
      <p class="empty-text">Inventory is empty</p>
      <small class="empty-hint">Use 'inventory' command to check what you have</small>
    </div>
    
    <!-- 库存物品网格 -->
    <div v-else class="inventory-grid">
      <div 
        v-for="item in inventoryItems" 
        :key="item.itemName"
        class="inventory-item"
        :class="{ 'item-highlighted': item.count > 1 }"
      >
        <div class="item-icon">
          {{ getItemIcon(item.itemName) }}
        </div>
        <div class="item-info">
          <div class="item-name">{{ item.displayName || item.itemName }}</div>
          <div class="item-count" v-if="item.count > 1">× {{ item.count }}</div>
        </div>
        <div class="item-badge" v-if="item.count > 1">
          {{ item.count }}
        </div>
      </div>
    </div>
    
    <!-- 库存统计 -->
    <div v-if="inventoryItems.length > 0" class="inventory-stats">
      <div class="stat-item">
        <span class="stat-label">Total Types:</span>
        <span class="stat-value">{{ inventoryItems.length }}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Total Items:</span>
        <span class="stat-value">{{ totalItems }}</span>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'

export default {
  name: 'InventoryViewer',
  props: {
    environmentId: {
      type: [String, Number],
      required: true
    },
    currentState: {
      type: Object,
      default: () => ({})
    }
  },
  setup(props) {
    // 从环境状态中解析库存
    const inventoryItems = computed(() => {
      const observation = props.currentState?.observation || ''
      if (!observation || typeof observation !== 'string') {
        return []
      }
      
      return parseInventoryFromObservation(observation)
    })
    
    // 计算总物品数量
    const totalItems = computed(() => {
      return inventoryItems.value.reduce((total, item) => total + (item.count || 1), 0)
    })
    
    // 解析库存观察文本
    const parseInventoryFromObservation = (observation) => {
      const inventory = []
      
      // 查找库存部分
      const inventoryMatch = observation.match(/Inventory:\s*(.+?)(?:\n|$)/i)
      if (inventoryMatch) {
        const inventoryText = inventoryMatch[1].trim()
        
        // 检查是否为空库存
        if (inventoryText.includes("You are not carrying anything") || 
            inventoryText.includes('empty') || 
            inventoryText.toLowerCase().includes('nothing') ||
            inventoryText === '') {
          return []
        }
        
        // 解析物品格式: [item name] (count) 或 item name (count)
        const patterns = [
          /\[([^\]]+)\]\s*\((\d+)\)/g,  // [item name] (count)
          /([a-zA-Z_\s]+)\s*\((\d+)\)/g, // item name (count)
          /([a-zA-Z_\s]+):\s*(\d+)/g,   // item name: count
        ]
        
        for (const pattern of patterns) {
          const itemMatches = Array.from(inventoryText.matchAll(pattern))
          for (const match of itemMatches) {
            const itemName = match[1].trim()
            const count = parseInt(match[2]) || 1
            
            if (itemName && !inventory.find(item => item.itemName === itemName)) {
              inventory.push({
                itemName,
                count,
                displayName: formatItemName(itemName)
              })
            }
          }
        }
        
        // 如果没有匹配到数量格式，尝试简单的物品名称
        if (inventory.length === 0 && inventoryText !== '') {
          const simpleItems = inventoryText.split(',').map(item => item.trim())
          for (const item of simpleItems) {
            if (item && !item.includes('You are not') && !item.includes('nothing')) {
              inventory.push({
                itemName: item,
                count: 1,
                displayName: formatItemName(item)
              })
            }
          }
        }
      }
      
      return inventory
    }
    
    // 格式化物品名称
    const formatItemName = (itemName) => {
      return itemName
        .replace(/^minecraft:/, '')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, letter => letter.toUpperCase())
    }
    
    // 获取物品图标
    const getItemIcon = (itemName) => {
      const name = itemName.toLowerCase()
      
      // 武器和工具
      if (name.includes('sword')) return '⚔️'
      if (name.includes('pickaxe')) return '⛏️'
      if (name.includes('axe')) return '🪓'
      if (name.includes('shovel') || name.includes('spade')) return '🥄'
      if (name.includes('hoe')) return '🔨'
      
      // 护甲
      if (name.includes('helmet')) return '⛑️'
      if (name.includes('chestplate')) return '🦺'
      if (name.includes('leggings')) return '👖'
      if (name.includes('boots')) return '👢'
      
      // 材料
      if (name.includes('wood') || name.includes('log')) return '🪵'
      if (name.includes('stone')) return '🪨'
      if (name.includes('iron')) return '⚙️'
      if (name.includes('gold') || name.includes('golden')) return '💛'
      if (name.includes('diamond')) return '💎'
      if (name.includes('emerald')) return '💚'
      
      // 食物
      if (name.includes('bread')) return '🍞'
      if (name.includes('apple')) return '🍎'
      if (name.includes('carrot')) return '🥕'
      if (name.includes('potato')) return '🥔'
      if (name.includes('meat') || name.includes('beef')) return '🥩'
      
      // 其他常见物品
      if (name.includes('book')) return '📚'
      if (name.includes('arrow')) return '🏹'
      if (name.includes('bow')) return '🏹'
      if (name.includes('torch')) return '🕯️'
      if (name.includes('coal')) return '⚫'
      if (name.includes('stick')) return '🥢'
      
      // 默认图标
      return '📦'
    }
    
    return {
      inventoryItems,
      totalItems,
      getItemIcon
    }
  }
}
</script>

<style scoped>
.inventory-viewer {
  background: var(--bg-card-solid);
  border-radius: var(--radius-medium);
  border: 2px solid var(--border-light);
  overflow: hidden;
}

.inventory-header {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  padding: 1rem 1.5rem;
  border-bottom: 2px solid var(--border-light);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.inventory-header h4 {
  margin: 0;
  color: var(--text-primary);
  font-size: 1.2rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.inventory-count {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.count-badge {
  background: var(--primary-color);
  color: white;
  font-size: 0.9rem;
  font-weight: 700;
  padding: 0.3rem 0.7rem;
  border-radius: var(--radius-large);
  min-width: 24px;
  text-align: center;
}

.count-text {
  color: var(--text-secondary);
  font-size: 0.9rem;
  font-weight: 500;
}

.empty-inventory {
  padding: 3rem 1.5rem;
  text-align: center;
  color: var(--text-secondary);
}

.empty-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  opacity: 0.6;
}

.empty-text {
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0 0 0.5rem 0;
  color: var(--text-secondary);
}

.empty-hint {
  font-size: 0.9rem;
  color: var(--text-light);
  font-style: italic;
}

.inventory-grid {
  padding: 1rem;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 0.75rem;
  max-height: 300px;
  overflow-y: auto;
}

.inventory-item {
  background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
  border: 2px solid var(--border-light);
  border-radius: var(--radius-medium);
  padding: 1rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  transition: all var(--transition-normal);
  position: relative;
  cursor: pointer;
}

.inventory-item:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-medium);
  border-color: var(--primary-light);
}

.inventory-item.item-highlighted {
  border-color: var(--accent-color);
  background: linear-gradient(135deg, #fff8f0 0%, #ffebe0 100%);
}

.item-icon {
  font-size: 1.8rem;
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #f1f3f4 0%, #e8eaed 100%);
  border-radius: var(--radius-small);
  border: 1px solid var(--border-light);
}

.item-info {
  flex: 1;
  min-width: 0;
}

.item-name {
  font-weight: 600;
  color: var(--text-primary);
  font-size: 0.95rem;
  line-height: 1.3;
  word-break: break-word;
}

.item-count {
  color: var(--text-secondary);
  font-size: 0.8rem;
  font-weight: 500;
  margin-top: 0.2rem;
}

.item-badge {
  position: absolute;
  top: -8px;
  right: -8px;
  background: var(--accent-color);
  color: white;
  font-size: 0.75rem;
  font-weight: 700;
  padding: 0.2rem 0.5rem;
  border-radius: var(--radius-large);
  min-width: 20px;
  text-align: center;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
}

.inventory-stats {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  padding: 1rem 1.5rem;
  border-top: 1px solid var(--border-light);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.stat-label {
  color: var(--text-secondary);
  font-size: 0.9rem;
  font-weight: 500;
}

.stat-value {
  color: var(--text-primary);
  font-size: 0.9rem;
  font-weight: 700;
  background: var(--bg-card-solid);
  padding: 0.2rem 0.6rem;
  border-radius: var(--radius-small);
  border: 1px solid var(--border-light);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .inventory-grid {
    grid-template-columns: 1fr;
    gap: 0.5rem;
  }
  
  .inventory-item {
    padding: 0.75rem;
  }
  
  .item-icon {
    font-size: 1.5rem;
    width: 35px;
    height: 35px;
  }
  
  .inventory-stats {
    flex-direction: column;
    gap: 0.5rem;
    align-items: stretch;
  }
}
</style>