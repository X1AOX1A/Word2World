<template>
  <div class="textcraft-inventory-viewer">
    <div class="inventory-header">
      <h4 class="inventory-title">
        <span class="title-icon">🎒</span>
        <span>Inventory</span>
        <span class="item-count">({{ inventory.length }} items)</span>
      </h4>
      
      <div class="inventory-controls">
        <button 
          @click="refreshInventory" 
          :disabled="loading"
          class="refresh-btn"
          title="Refresh inventory"
        >
          {{ loading ? '⏳' : '🔄' }}
        </button>
        
        <button 
          @click="toggleView" 
          class="view-toggle-btn"
          :title="`Switch to ${viewMode === 'grid' ? 'list' : 'grid'} view`"
        >
          {{ viewMode === 'grid' ? '📋' : '⊞' }}
        </button>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-state">
      <div class="loading-spinner"></div>
      <span>Loading inventory...</span>
    </div>

    <!-- 空库存状态 -->
    <div v-else-if="inventory.length === 0" class="empty-inventory">
      <div class="empty-icon">📦</div>
      <div class="empty-text">
        <h5>Your inventory is empty</h5>
        <p>Use <code>get [amount] [item_name]</code> to collect items</p>
      </div>
    </div>

    <!-- 库存网格视图 -->
    <div v-else-if="viewMode === 'grid'" class="inventory-grid">
      <div 
        v-for="item in sortedInventory" 
        :key="item.itemName"
        class="inventory-item-card"
        @click="selectItem(item)"
        :class="{ 'selected': selectedItem?.itemName === item.itemName }"
      >
        <div class="item-icon">{{ getItemIcon(item.itemName) }}</div>
        <div class="item-info">
          <div class="item-name" :title="item.itemName">
            {{ formatItemName(item.itemName) }}
          </div>
          <div class="item-count">{{ item.count }}</div>
        </div>
        <div class="item-actions">
          <button 
            @click.stop="suggestCraftingWith(item)"
            class="action-btn"
            title="Suggest crafting recipes using this item"
          >
            ⚒️
          </button>
        </div>
      </div>
    </div>

    <!-- 库存列表视图 -->
    <div v-else class="inventory-list">
      <div 
        v-for="item in sortedInventory" 
        :key="item.itemName"
        class="inventory-item-row"
        @click="selectItem(item)"
        :class="{ 'selected': selectedItem?.itemName === item.itemName }"
      >
        <div class="row-icon">{{ getItemIcon(item.itemName) }}</div>
        <div class="row-info">
          <div class="row-name">{{ formatItemName(item.itemName) }}</div>
          <div class="row-meta">{{ item.itemName }}</div>
        </div>
        <div class="row-count">{{ item.count }}</div>
        <div class="row-actions">
          <button 
            @click.stop="suggestCraftingWith(item)"
            class="action-btn"
            title="Suggest crafting with this item"
          >
            ⚒️
          </button>
        </div>
      </div>
    </div>

    <!-- 选中物品详情 -->
    <div v-if="selectedItem" class="item-detail-panel">
      <div class="detail-header">
        <span class="detail-icon">{{ getItemIcon(selectedItem.itemName) }}</span>
        <div class="detail-title">
          <h5>{{ formatItemName(selectedItem.itemName) }}</h5>
          <p>{{ selectedItem.itemName }}</p>
        </div>
        <button @click="selectedItem = null" class="close-detail-btn">×</button>
      </div>
      
      <div class="detail-content">
        <div class="detail-stats">
          <div class="stat">
            <span class="stat-label">Count:</span>
            <span class="stat-value">{{ selectedItem.count }}</span>
          </div>
          <div class="stat">
            <span class="stat-label">Type:</span>
            <span class="stat-value">{{ getItemType(selectedItem.itemName) }}</span>
          </div>
        </div>
        
        <div class="detail-actions">
          <button 
            @click="suggestCraftingWith(selectedItem)"
            class="detail-action-btn primary"
          >
            🛠️ Find Crafting Recipes
          </button>
          <button 
            @click="suggestActionsFor(selectedItem)"
            class="detail-action-btn secondary"
          >
            ⚡ Suggest Actions
          </button>
        </div>
        
        <!-- 建议的动作 -->
        <div v-if="suggestedActions.length > 0" class="suggested-actions">
          <h6>💡 Suggested Actions:</h6>
          <div class="suggestion-list">
            <button 
              v-for="action in suggestedActions"
              :key="action"
              @click="emitSuggestedAction(action)"
              class="suggestion-btn"
            >
              {{ action }}
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- 统计信息 -->
    <div class="inventory-stats">
      <div class="stat-item">
        <span class="stat-icon">📊</span>
        <span class="stat-text">Total Items: {{ totalItemCount }}</span>
      </div>
      <div class="stat-item">
        <span class="stat-icon">🏷️</span>
        <span class="stat-text">Unique Types: {{ inventory.length }}</span>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, watch, onMounted } from 'vue'
import TextCraftClient from '../client/textcraftClient.js'

export default {
  name: 'TextCraftInventoryViewer',
  props: {
    environmentId: {
      type: [String, Number],
      required: true
    },
    currentState: {
      type: Object,
      default: () => ({})
    },
    autoRefresh: {
      type: Boolean,
      default: true
    }
  },
  emits: ['suggest-action', 'item-selected'],
  setup(props, { emit }) {
    const loading = ref(false)
    const viewMode = ref('grid') // 'grid' | 'list'
    const selectedItem = ref(null)
    const suggestedActions = ref([])
    const client = new TextCraftClient()

    // 库存数据
    const inventory = ref([])

    // 计算属性
    const sortedInventory = computed(() => {
      return [...inventory.value].sort((a, b) => {
        // 按数量降序，然后按名称升序
        if (a.count !== b.count) {
          return b.count - a.count
        }
        return a.itemName.localeCompare(b.itemName)
      })
    })

    const totalItemCount = computed(() => {
      return inventory.value.reduce((total, item) => total + item.count, 0)
    })

    // 刷新库存
    const refreshInventory = async () => {
      if (!props.environmentId) return
      
      loading.value = true
      try {
        const inventoryData = await client.queryInventory(props.environmentId)
        inventory.value = inventoryData
        console.log(`🎒 Inventory refreshed: ${inventoryData.length} items found`)
      } catch (error) {
        console.error('❌ Failed to refresh inventory:', error)
      } finally {
        loading.value = false
      }
    }

    // 切换视图模式
    const toggleView = () => {
      viewMode.value = viewMode.value === 'grid' ? 'list' : 'grid'
    }

    // 选择物品
    const selectItem = (item) => {
      selectedItem.value = selectedItem.value?.itemName === item.itemName ? null : item
      if (selectedItem.value) {
        emit('item-selected', selectedItem.value)
        suggestActionsFor(selectedItem.value)
      }
    }

    // 获取物品图标
    const getItemIcon = (itemName) => {
      const iconMap = {
        'wood': '🪵', 'oak_log': '🪵', 'log': '🪵',
        'stick': '🪄', 'wooden_stick': '🪄',
        'stone': '🪨', 'cobblestone': '🪨',
        'iron': '⚙️', 'iron_ingot': '⚙️',
        'diamond': '💎', 'diamond_gem': '💎',
        'coal': '🔥', 'coal_ore': '🔥',
        'wooden_pickaxe': '⛏️', 'pickaxe': '⛏️',
        'wooden_axe': '🪓', 'axe': '🪓',
        'wooden_sword': '⚔️', 'sword': '⚔️',
        'wooden_shovel': '🗿', 'shovel': '🗿',
        'plank': '📋', 'planks': '📋', 'wooden_planks': '📋',
        'chest': '📦', 'wooden_chest': '📦',
        'door': '🚪', 'wooden_door': '🚪',
        'boat': '⛵', 'wooden_boat': '⛵'
      }

      // 查找完全匹配
      if (iconMap[itemName]) {
        return iconMap[itemName]
      }

      // 查找部分匹配
      for (const [key, icon] of Object.entries(iconMap)) {
        if (itemName.toLowerCase().includes(key) || key.includes(itemName.toLowerCase())) {
          return icon
        }
      }

      // 根据物品类型返回默认图标
      if (itemName.includes('tool') || itemName.includes('pickaxe') || itemName.includes('axe')) {
        return '🔧'
      }
      if (itemName.includes('weapon') || itemName.includes('sword')) {
        return '⚔️'
      }
      if (itemName.includes('wood') || itemName.includes('oak')) {
        return '🪵'
      }
      if (itemName.includes('stone') || itemName.includes('rock')) {
        return '🪨'
      }

      return '📦' // 默认图标
    }

    // 格式化物品名称
    const formatItemName = (itemName) => {
      return itemName
        .replace(/^minecraft:/, '')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, letter => letter.toUpperCase())
    }

    // 获取物品类型
    const getItemType = (itemName) => {
      if (itemName.includes('tool') || itemName.includes('pickaxe') || itemName.includes('axe') || itemName.includes('shovel')) {
        return 'Tool'
      }
      if (itemName.includes('weapon') || itemName.includes('sword')) {
        return 'Weapon'
      }
      if (itemName.includes('wood') || itemName.includes('log') || itemName.includes('plank')) {
        return 'Wood'
      }
      if (itemName.includes('stone') || itemName.includes('cobblestone')) {
        return 'Stone'
      }
      if (itemName.includes('ore') || itemName.includes('ingot')) {
        return 'Ore/Metal'
      }
      return 'Material'
    }

    // 建议使用物品进行合成
    const suggestCraftingWith = (item) => {
      const suggestions = []
      const itemName = item.itemName.toLowerCase()

      // 基于物品类型提供合成建议
      if (itemName.includes('wood') || itemName.includes('log')) {
        suggestions.push(`craft 4 planks using 1 ${item.itemName}`)
        if (item.count >= 1) {
          suggestions.push(`craft 1 stick using 1 ${item.itemName}`)
        }
      }
      
      if (itemName.includes('plank') && item.count >= 3) {
        suggestions.push(`craft 1 wooden_pickaxe using 3 ${item.itemName}, 2 stick`)
        suggestions.push(`craft 1 wooden_axe using 3 ${item.itemName}, 2 stick`)
        suggestions.push(`craft 1 wooden_sword using 2 ${item.itemName}, 1 stick`)
      }
      
      if (itemName.includes('stick') && item.count >= 2) {
        suggestions.push(`craft 1 wooden_pickaxe using 3 planks, 2 ${item.itemName}`)
        suggestions.push(`craft 1 wooden_axe using 3 planks, 2 ${item.itemName}`)
      }

      if (itemName.includes('stone') && item.count >= 3) {
        suggestions.push(`craft 1 stone_pickaxe using 3 ${item.itemName}, 2 stick`)
        suggestions.push(`craft 1 stone_axe using 3 ${item.itemName}, 2 stick`)
      }

      suggestedActions.value = suggestions.slice(0, 5) // 限制建议数量
    }

    // 为物品建议动作
    const suggestActionsFor = (item) => {
      const suggestions = []
      
      // 通用动作
      suggestions.push(`examine ${item.itemName}`)
      
      // 基于物品的特定动作
      if (item.itemName.includes('tool') || item.itemName.includes('pickaxe')) {
        suggestions.push(`use ${item.itemName}`)
        suggestions.push(`mine with ${item.itemName}`)
      }
      
      if (item.itemName.includes('food')) {
        suggestions.push(`eat ${item.itemName}`)
      }

      // 合成建议
      suggestCraftingWith(item)
    }

    // 发射建议动作
    const emitSuggestedAction = (action) => {
      emit('suggest-action', action)
      selectedItem.value = null // 关闭详情面板
    }

    // 监听环境状态变化
    watch(() => props.currentState, (newState) => {
      if (newState && props.autoRefresh) {
        // 从状态中更新库存
        inventory.value = client.getCachedInventory(props.environmentId)
      }
    }, { deep: true })

    // 监听环境ID变化
    watch(() => props.environmentId, () => {
      if (props.environmentId) {
        refreshInventory()
      }
    }, { immediate: true })

    onMounted(() => {
      if (props.environmentId) {
        refreshInventory()
      }
    })

    return {
      loading,
      viewMode,
      selectedItem,
      suggestedActions,
      inventory,
      sortedInventory,
      totalItemCount,
      refreshInventory,
      toggleView,
      selectItem,
      getItemIcon,
      formatItemName,
      getItemType,
      suggestCraftingWith,
      suggestActionsFor,
      emitSuggestedAction
    }
  }
}
</script>

<style scoped>
.textcraft-inventory-viewer {
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  height: 100%;
}

.inventory-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem;
  background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
  border-bottom: 2px solid #e2e8f0;
}

.inventory-title {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 0;
  color: #2d3748;
  font-size: 1.3rem;
  font-weight: 700;
}

.title-icon {
  font-size: 1.5rem;
}

.item-count {
  font-size: 0.9rem;
  color: #718096;
  font-weight: 400;
}

.inventory-controls {
  display: flex;
  gap: 0.5rem;
}

.refresh-btn, .view-toggle-btn {
  padding: 0.5rem;
  background: white;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 1.1rem;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.refresh-btn:hover, .view-toggle-btn:hover {
  background: #667eea;
  color: white;
  border-color: #667eea;
  transform: translateY(-1px);
}

.refresh-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 3rem;
  color: #718096;
}

.loading-spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #e2e8f0;
  border-top: 3px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.empty-inventory {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 3rem;
  text-align: center;
  color: #a0aec0;
}

.empty-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.empty-text h5 {
  color: #4a5568;
  margin-bottom: 0.5rem;
}

.empty-text p {
  color: #718096;
  margin: 0;
}

.empty-text code {
  background: #edf2f7;
  padding: 0.2rem 0.4rem;
  border-radius: 4px;
  font-family: 'Fira Code', monospace;
  font-size: 0.9rem;
}

/* 网格视图 */
.inventory-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 1rem;
  padding: 1.5rem;
  flex: 1;
  overflow-y: auto;
}

.inventory-item-card {
  background: #f8fafc;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  padding: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  position: relative;
}

.inventory-item-card:hover {
  border-color: #667eea;
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(102, 126, 234, 0.2);
}

.inventory-item-card.selected {
  border-color: #667eea;
  background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
}

.item-icon {
  font-size: 2.5rem;
  margin-bottom: 0.75rem;
}

.item-info {
  flex: 1;
  width: 100%;
}

.item-name {
  font-weight: 600;
  color: #2d3748;
  font-size: 0.9rem;
  margin-bottom: 0.25rem;
  line-height: 1.2;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.item-count {
  color: #667eea;
  font-weight: 700;
  font-size: 1.1rem;
}

.item-actions {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  opacity: 0;
  transition: opacity 0.3s ease;
}

.inventory-item-card:hover .item-actions {
  opacity: 1;
}

/* 列表视图 */
.inventory-list {
  flex: 1;
  overflow-y: auto;
}

.inventory-item-row {
  display: flex;
  align-items: center;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid #e2e8f0;
  cursor: pointer;
  transition: all 0.3s ease;
}

.inventory-item-row:hover {
  background: #f8fafc;
}

.inventory-item-row.selected {
  background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
  border-left: 4px solid #667eea;
}

.row-icon {
  font-size: 1.8rem;
  margin-right: 1rem;
  min-width: 40px;
}

.row-info {
  flex: 1;
}

.row-name {
  font-weight: 600;
  color: #2d3748;
  margin-bottom: 0.25rem;
}

.row-meta {
  font-size: 0.8rem;
  color: #718096;
}

.row-count {
  font-weight: 700;
  color: #667eea;
  font-size: 1.2rem;
  margin-right: 1rem;
  min-width: 40px;
  text-align: center;
}

.row-actions {
  display: flex;
  gap: 0.5rem;
}

.action-btn {
  padding: 0.5rem;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 1rem;
}

.action-btn:hover {
  background: #667eea;
  color: white;
  border-color: #667eea;
}

/* 物品详情面板 */
.item-detail-panel {
  border-top: 2px solid #e2e8f0;
  background: #f8fafc;
}

.detail-header {
  display: flex;
  align-items: center;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid #e2e8f0;
}

.detail-icon {
  font-size: 2rem;
  margin-right: 1rem;
}

.detail-title {
  flex: 1;
}

.detail-title h5 {
  margin: 0 0 0.25rem 0;
  color: #2d3748;
  font-weight: 600;
}

.detail-title p {
  margin: 0;
  color: #718096;
  font-size: 0.9rem;
}

.close-detail-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: #a0aec0;
  transition: color 0.3s ease;
}

.close-detail-btn:hover {
  color: #e53e3e;
}

.detail-content {
  padding: 1.5rem;
}

.detail-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.stat {
  display: flex;
  justify-content: space-between;
  padding: 0.75rem;
  background: white;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}

.stat-label {
  font-weight: 600;
  color: #4a5568;
}

.stat-value {
  color: #2d3748;
}

.detail-actions {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.detail-action-btn {
  flex: 1;
  padding: 0.75rem 1rem;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.3s ease;
}

.detail-action-btn.primary {
  background: #667eea;
  color: white;
}

.detail-action-btn.primary:hover {
  background: #5a67d8;
  transform: translateY(-1px);
}

.detail-action-btn.secondary {
  background: white;
  color: #667eea;
  border: 2px solid #667eea;
}

.detail-action-btn.secondary:hover {
  background: #667eea;
  color: white;
}

.suggested-actions {
  margin-top: 1.5rem;
}

.suggested-actions h6 {
  margin: 0 0 1rem 0;
  color: #4a5568;
  font-weight: 600;
}

.suggestion-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.suggestion-btn {
  padding: 0.75rem;
  background: white;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  cursor: pointer;
  text-align: left;
  transition: all 0.3s ease;
  font-family: 'Fira Code', monospace;
  font-size: 0.9rem;
}

.suggestion-btn:hover {
  border-color: #667eea;
  background: #f0f4ff;
  transform: translateX(4px);
}

/* 统计信息 */
.inventory-stats {
  display: flex;
  justify-content: space-around;
  padding: 1rem 1.5rem;
  background: #f8fafc;
  border-top: 1px solid #e2e8f0;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #4a5568;
  font-size: 0.9rem;
  font-weight: 500;
}

.stat-icon {
  font-size: 1.1rem;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .inventory-grid {
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 0.75rem;
    padding: 1rem;
  }
  
  .inventory-header {
    padding: 1rem;
  }
  
  .inventory-title {
    font-size: 1.1rem;
  }
  
  .detail-actions {
    flex-direction: column;
  }
  
  .inventory-stats {
    flex-direction: column;
    gap: 0.5rem;
    text-align: center;
  }
}
</style>