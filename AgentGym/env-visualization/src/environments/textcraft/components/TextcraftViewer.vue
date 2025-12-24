<template>
  <div class="textcraft-viewer">
    <div class="content-container">
      <!-- Goal and Inventory -->
      <div class="main-column">
        <!-- Goal Section -->
        <div class="panel goal-panel">
          <h3 class="panel-title">
            <span class="panel-icon">🎯</span> Goal
          </h3>
          <div class="goal-content">{{ goal || 'No goal specified' }}</div>
        </div>
        
        <!-- Inventory Section -->
        <div class="panel inventory-panel">
          <h3 class="panel-title">
            <span class="panel-icon">🎒</span> Inventory
            <button @click="updateInventory" class="refresh-inventory-btn" :disabled="loading">
              <span class="refresh-icon">↻</span>
            </button>
          </h3>
          
          <div v-if="inventory.length > 0" class="inventory-container">
            <div class="inventory-grid">
              <div v-for="item in inventory" :key="item.name || item.itemName" class="inventory-item">
                <div class="item-image">
                  <img v-if="item.image" :src="item.image" :alt="item.name" @error="handleImageError(item)" />
                  <div v-else class="placeholder-image">{{ (item.name || item.itemName).charAt(0) }}</div>
                </div>
                <div class="item-name">{{ item.name || item.itemName }}</div>
                <div class="item-count">x{{ item.count }}</div>
              </div>
            </div>
          </div>
          
          <div v-else class="empty-inventory">
            <span class="empty-icon">🔍</span>
            <span>Your inventory is empty</span>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Loading State -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner"></div>
      <div>Loading...</div>
    </div>
  </div>
</template>

<script>
import assetManager from '../../../shared/services/assetManager.js';

export default {
  name: 'TextcraftViewer',
  props: {
    environmentId: {
      type: [String, Number],
      default: null
    },
    client: {
      type: Object,
      default: null
    },
    envState: {
      type: Object,
      default: () => null
    }
  },
  emits: [
    'action-executed',
    'goal-completed',
    'state-updated',
    'environment-reset'  // 添加重置事件的emit声明
  ],
  data() {
    return {
      loading: false,
      inventory: [],
      goal: '',
      
      // Lock mechanism to prevent multiple simultaneous operations
      isOperationLocked: false,
      operationLockTimeout: null,

      // 添加debug模式用于调试
      debug: true
    }
  },
  watch: {
    // Watch for environment ID changes
    environmentId(newId, oldId) {
      if (newId && newId !== oldId) {
        console.log(`📌 TextcraftViewer: Environment ID changed from ${oldId} to ${newId}`);
        this.resetState();
        this.initializeEnvironment();
      }
    },
    
    // Watch for external state updates
    envState: {
      handler(newState) {
        if (newState && newState.observation) {
          console.log('📌 TextcraftViewer: Received new state data from parent');
          
          // 检查是否是重置操作（环境steps为0表示重置）
          const isReset = newState.steps === 0;
          
          // 如果是重置操作，先清空状态
          if (isReset) {
            console.log('🔄 TextcraftViewer: Detected environment reset');
            
            // 重置内部状态
            this.inventory = [];
            this.goal = '';
            
            // 重新初始化
            this.resetState();
          }
          
          // 处理状态数据
          this.processStateData(newState);
          
          // 更新库存
          setTimeout(() => this.updateInventory(), 100);
        }
      },
      deep: true
    }
  },
  methods: {
    // Initialize environment on first load
    async initializeEnvironment() {
      if (!this.environmentId || !this.client) {
        console.log('⚠️ Cannot initialize: Missing environment ID or client');
        return;
      }
      
      // If we're already in a loading state, don't try to initialize again
      if (this.loading || this.isOperationLocked) {
        console.log('⚠️ Skipping initialization - already loading or locked');
        return;
      }
      
      try {
        this.lockOperation('Initializing environment');
        this.loading = true;
        
        // First get the current state to extract goal and commands
        const stateResult = await this.client.getObservation(this.environmentId);
        if (stateResult && stateResult.success) {
          this.processStateData(stateResult);
        }
        
        // Then get the inventory
        await this.updateInventory();
      } catch (error) {
        console.error('❌ Error initializing TextCraft environment:', error);
      } finally {
        this.loading = false;
        this.unlockOperation();
      }
    },
    
    // Reset internal state when environment changes
    resetState() {
      this.loading = false;
      this.inventory = [];
      this.goal = '';
      this.unlockOperation();
    },
    
    // Process observation data to extract goal, commands, etc.
    processStateData(stateData) {
      if (!stateData) return;
      
      // Extract the observation string
      const observation = this.extractObservationText(stateData);
      if (!observation) return;
      
      // Extract goal using regex - handle both "Goal: craft X" and "Goal: X" formats
      const goalMatch = observation.match(/Goal:\s*(?:craft\s+)?([^.]+)\.?/i);
      if (goalMatch && goalMatch[1]) {
        const newGoal = goalMatch[1].trim();
        const goalChanged = this.goal !== newGoal;
        
        // 检测到goal变化，表示环境可能已重置
        if (goalChanged && this.goal) {
          console.log(`🔄 TextcraftViewer: Goal changed from "${this.goal}" to "${newGoal}"`);
          
          // 发送环境重置事件，通知父组件环境已重置
          this.$emit('environment-reset', {
            reason: 'goal_changed',
            oldGoal: this.goal,
            newGoal: newGoal,
            observation: observation
          });
        }
        
        this.goal = newGoal;
        console.log('📝 Extracted goal:', this.goal);
      } else {
        console.log('⚠️ No goal found in observation text');
      }
      
      // Parse inventory items from the observation - 从观察文本中直接解析库存
      this.parseInventoryFromObservation(observation);
      
      // Emit state update so parent components know we've processed this
      this.$emit('state-processed', {
        goal: this.goal,
        observation
      });
    },
    
    // Extract observation text from different data formats
    extractObservationText(data) {
      if (!data) return '';
      
      // Handle string observation
      if (typeof data === 'string') {
        return data;
      }
      
      // Handle object with observation field
      if (typeof data === 'object') {
        // Check for observation in data or data.data
        if (data.observation) {
          return typeof data.observation === 'string' 
            ? data.observation 
            : JSON.stringify(data.observation);
        }
        
        // Check if data.data contains the observation
        if (data.data) {
          if (typeof data.data === 'string') {
            return data.data;
          }
          if (typeof data.data === 'object' && data.data.observation) {
            return typeof data.data.observation === 'string'
              ? data.data.observation
              : JSON.stringify(data.data.observation);
          }
        }
        
        // Last resort: stringify the whole object
        return JSON.stringify(data);
      }
      
      // Handle other types
      return String(data);
    },
    
    // Parse inventory directly from observation
    parseInventoryFromObservation(observation) {
      if (!observation) return;
      
      // Try to extract inventory information manually
      // Match format: Inventory: [wheat] (2) [cocoa beans] (1)
      const inventoryMatch = observation.match(/Inventory:\s*(.+?)(?:\n|$)/i);
      if (inventoryMatch && inventoryMatch[1]) {
        const inventoryText = inventoryMatch[1].trim();
        if (this.debug) console.log('📋 Raw inventory text:', inventoryText);
        
        if (inventoryText.includes('empty') || inventoryText === '' || 
            inventoryText.includes('not carrying anything')) {
          this.inventory = [];
          return;
        }
        
        const itemRegex = /\[([^\]]+)\]\s*\((\d+)\)/g;
        const matches = Array.from(inventoryText.matchAll(itemRegex));
        
        if (matches && matches.length > 0) {
          const newInventory = [];
          
          matches.forEach(match => {
            const itemName = match[1];
            const count = parseInt(match[2]);
            
            // 规范化物品名称，用于查找贴图
            const normalizedName = this.normalizeItemName(itemName);
            
            newInventory.push({
              name: itemName,
              itemName: normalizedName,
              normalizedName: normalizedName, // 保存规范化名称
              count: count,
              image: `/assets/minecraft/textures/item/${normalizedName}.png`
            });
          });
          
          if (this.debug) {
            console.log('📦 Parsed inventory items:', newInventory);
            console.log('🖼️ Image paths:', newInventory.map(item => item.image).join(', '));
          }
          this.inventory = newInventory;
        }
      }
    },

    // 规范化物品名称，处理特殊情况
    normalizeItemName(itemName) {
      if (!itemName) return '';
      
      // 转换为小写，替换空格为下划线
      let normalized = itemName.toLowerCase().replace(/\s+/g, '_');
      
      // 处理形状变体 - 提取基础材料名称
      const shapeVariants = [
        '_slab', '_stairs', '_wall', '_fence', '_gate', '_button', 
        '_pressure_plate', '_door', '_trapdoor', '_sign', '_hanging_sign'
      ];
      
      let baseMaterial = normalized;
      let isShapeVariant = false;
      
      // 检查是否是形状变体
      for (const variant of shapeVariants) {
        if (normalized.endsWith(variant)) {
          baseMaterial = normalized.replace(variant, '');
          isShapeVariant = true;
          console.log(`🔧 Shape variant detected: ${normalized} -> base material: ${baseMaterial}`);
          break;
        }
      }
      
      // 扩展的特殊物品名称映射
      const specialItems = {
        // 原有映射
        'crimson_stem': 'crimson_stem',
        'crimson_hyphae': 'crimson_hyphae',
        'cocoa_bean': 'cocoa_beans',
        'cocoa_beans': 'cocoa_beans',
        'cookie': 'cookie',
        'wheat': 'wheat',
        
        // 基础材料映射（用于形状变体fallback）
        'blackstone': 'blackstone',
        'polished_blackstone': 'polished_blackstone',
        'cobblestone': 'cobblestone',
        'stone_bricks': 'stone_bricks',
        'bricks': 'bricks',
        'nether_bricks': 'nether_bricks',
        'end_stone_bricks': 'end_stone_bricks',
        'prismarine': 'prismarine',
        'dark_prismarine': 'dark_prismarine',
        'purpur_block': 'purpur_block',
        'quartz_block': 'quartz_block',
        'sandstone': 'sandstone',
        'red_sandstone': 'red_sandstone',
        'smooth_stone': 'smooth_stone',
        'smooth_sandstone': 'smooth_sandstone',
        'smooth_red_sandstone': 'smooth_red_sandstone',
        'smooth_quartz': 'quartz_block',
        'cut_sandstone': 'sandstone',
        'cut_red_sandstone': 'red_sandstone',
        'chiseled_sandstone': 'sandstone',
        'chiseled_red_sandstone': 'red_sandstone',
        'chiseled_stone_bricks': 'stone_bricks',
        'chiseled_nether_bricks': 'nether_bricks',
        'chiseled_polished_blackstone': 'polished_blackstone',
        'cracked_stone_bricks': 'stone_bricks',
        'cracked_nether_bricks': 'nether_bricks',
        'cracked_polished_blackstone_bricks': 'polished_blackstone',
        'mossy_stone_bricks': 'stone_bricks',
        'mossy_cobblestone': 'cobblestone',
        
        // 木材系列映射
        'birch_wood': 'birch_log',
        'oak_wood': 'oak_log',
        'spruce_wood': 'spruce_log',
        'jungle_wood': 'jungle_log',
        'acacia_wood': 'acacia_log',
        'dark_oak_wood': 'dark_oak_log',
        'cherry_wood': 'cherry_log',
        'mangrove_wood': 'mangrove_log',
        'bamboo_wood': 'bamboo_planks',
        
        // 木板映射
        'birch_plank': 'birch_planks',
        'oak_plank': 'oak_planks',
        'spruce_plank': 'spruce_planks',
        'jungle_plank': 'jungle_planks',
        'acacia_plank': 'acacia_planks',
        'dark_oak_plank': 'dark_oak_planks',
        'cherry_plank': 'cherry_planks',
        'mangrove_plank': 'mangrove_planks',
        
        // 工具和武器映射
        'wood_pickaxe': 'wooden_pickaxe',
        'wood_axe': 'wooden_axe',
        'wood_shovel': 'wooden_shovel',
        'wood_sword': 'wooden_sword',
        'wood_hoe': 'wooden_hoe',
        
        'stone_pickaxe': 'stone_pickaxe',
        'stone_axe': 'stone_axe',
        'stone_shovel': 'stone_shovel',
        'stone_sword': 'stone_sword',
        'stone_hoe': 'stone_hoe',
        
        'iron_pickaxe': 'iron_pickaxe',
        'iron_axe': 'iron_axe',
        'iron_shovel': 'iron_shovel',
        'iron_sword': 'iron_sword',
        'iron_hoe': 'iron_hoe',
        
        // 金属锭映射
        'iron': 'iron_ingot',
        'gold': 'gold_ingot',
        'copper': 'copper_ingot',
        'netherite': 'netherite_ingot',
        
        // 矿石映射
        'iron_ore': 'raw_iron',
        'gold_ore': 'raw_gold',
        'copper_ore': 'raw_copper',
        'coal_ore': 'coal',
        'diamond_ore': 'diamond',
        'emerald_ore': 'emerald',
        'redstone_ore': 'redstone',
        'lapis_ore': 'lapis_lazuli',
        
        // 食物映射
        'cooked_beef': 'cooked_beef',
        'raw_beef': 'beef',
        'cooked_pork': 'cooked_porkchop',
        'raw_pork': 'porkchop',
        'cooked_chicken': 'cooked_chicken',
        'raw_chicken': 'chicken',
        'cooked_fish': 'cooked_cod',
        'raw_fish': 'cod',
        'bread': 'bread',
        'apple': 'apple',
        'carrot': 'carrot',
        'potato': 'potato',
        'baked_potato': 'baked_potato',
        
        // 其他常见物品
        'stick': 'stick',
        'string': 'string',
        'dirt': 'dirt',
        'grass_block': 'grass_block',
        'sand': 'sand',
        'gravel': 'gravel',
        'stone': 'stone',
        'glass': 'glass',
        'wool': 'white_wool',
        'water_bucket': 'water_bucket',
        'lava_bucket': 'lava_bucket',
        'bucket': 'bucket'
      };
      
      // 优先检查完整名称映射
      if (specialItems[normalized]) {
        normalized = specialItems[normalized];
      } 
      // 如果是形状变体，尝试使用基础材料映射
      else if (isShapeVariant && specialItems[baseMaterial]) {
        normalized = specialItems[baseMaterial];
        console.log(`🔄 Using base material mapping: ${baseMaterial} -> ${normalized}`);
      }
      // 如果是形状变体但没有特殊映射，直接使用基础材料
      else if (isShapeVariant) {
        normalized = baseMaterial;
        console.log(`🔄 Using base material directly: ${normalized}`);
      }
      
      // 添加日志以便调试
      if (this.debug) {
        console.log(`🔄 规范化物品名称: ${itemName} -> ${normalized}`);
      }
      
      return normalized;
    },

    // Handle image loading errors with enhanced fallback
    handleImageError(item) {
      console.log(`🖼️ Failed to load image for ${item.name || item.itemName}`);
      
      const itemName = item.normalizedName || item.itemName;
      const originalName = item.name || item.itemName;
      
      // Try multiple fallback strategies
      if (item.image.includes('/item/')) {
        // Strategy 1: Try block texture
        const blockTexture = `/assets/minecraft/textures/block/${itemName}.png`;
        console.log(`🔄 Trying block texture as fallback for ${itemName}`);
        item.image = blockTexture;
      } else if (item.image.includes('/block/')) {
        // Strategy 2: Try alternative naming patterns
        const alternatives = this.generateAlternativeNames(originalName);
        
        for (const altName of alternatives) {
          if (altName !== itemName) {
            console.log(`🔄 Trying alternative name: ${altName}`);
            item.image = `/assets/minecraft/textures/item/${altName}.png`;
            return; // Try one at a time
          }
        }
        
        // Strategy 3: Try generic patterns
        const genericName = this.tryGenericMapping(originalName);
        if (genericName && genericName !== itemName) {
          console.log(`🔄 Trying generic mapping: ${genericName}`);
          item.image = `/assets/minecraft/textures/item/${genericName}.png`;
          return;
        }
        
        // Strategy 4: Last resort - remove image
        console.log(`❌ No texture found for ${itemName}, using placeholder`);
        item.image = null;
      } else {
        // Already tried alternatives, use placeholder
        console.log(`❌ No texture found for ${itemName}, using placeholder`);
        item.image = null;
      }
    },
    
    // Generate alternative names for items
    generateAlternativeNames(originalName) {
      const alternatives = [];
      const normalized = originalName.toLowerCase().replace(/\s+/g, '_');
      
      // Remove common suffixes/prefixes
      const patterns = [
        // Remove "_item", "_block" suffixes
        normalized.replace(/_item$/, ''),
        normalized.replace(/_block$/, ''),
        
        // Try with "raw_" prefix for ores
        `raw_${normalized.replace(/_ore$/, '')}`,
        
        // Try plural/singular forms
        normalized.replace(/s$/, ''), // remove trailing 's'
        `${normalized}s`, // add trailing 's'
        
        // Try with "cooked_" prefix for food
        `cooked_${normalized}`,
        normalized.replace(/^cooked_/, ''),
        
        // Try with different wood prefixes
        normalized.replace(/^(oak|birch|spruce|jungle|acacia|dark_oak|cherry|mangrove)_/, 'wooden_'),
        normalized.replace(/^wooden_/, 'oak_'),
        
        // Try with ingot suffix for metals
        `${normalized}_ingot`,
        normalized.replace(/_ingot$/, ''),
        
        // Try with different tool materials
        normalized.replace(/^(wood|wooden)_/, 'wooden_'),
        normalized.replace(/^wooden_/, 'wood_'),
      ];
      
      // Add unique alternatives
      patterns.forEach(pattern => {
        if (pattern && pattern !== normalized && !alternatives.includes(pattern)) {
          alternatives.push(pattern);
        }
      });
      
      return alternatives;
    },
    
    // Try generic mapping for common item types
    tryGenericMapping(originalName) {
      const normalized = originalName.toLowerCase().replace(/\s+/g, '_');
      
      // Generic mappings for common item categories
      if (normalized.includes('wood') || normalized.includes('log')) {
        return 'oak_log';
      }
      if (normalized.includes('plank')) {
        return 'oak_planks';
      }
      if (normalized.includes('pickaxe')) {
        return 'iron_pickaxe';
      }
      if (normalized.includes('axe') && !normalized.includes('pickaxe')) {
        return 'iron_axe';
      }
      if (normalized.includes('sword')) {
        return 'iron_sword';
      }
      if (normalized.includes('shovel') || normalized.includes('spade')) {
        return 'iron_shovel';
      }
      if (normalized.includes('ore') && !normalized.includes('core')) {
        return 'iron_ore';
      }
      if (normalized.includes('ingot') || (normalized.includes('iron') && !normalized.includes('ore'))) {
        return 'iron_ingot';
      }
      if (normalized.includes('stone') && !normalized.includes('cobble')) {
        return 'stone';
      }
      if (normalized.includes('cobble')) {
        return 'cobblestone';
      }
      if (normalized.includes('dirt')) {
        return 'dirt';
      }
      if (normalized.includes('grass')) {
        return 'grass_block';
      }
      
      return null;
    },
    
    // Update inventory with latest data
    async updateInventory() {
      if (!this.environmentId || !this.client) {
        console.log('⚠️ Cannot update inventory: Missing environment ID or client');
        return;
      }
      
      if (this.isOperationLocked) {
        console.log('⚠️ Skipping inventory update - operation locked');
        return;
      }
      
      try {
        this.lockOperation('Updating inventory');
        this.loading = true;
        
        const response = await this.client.step(this.environmentId, 'inventory');
        
        if (response && response.success) {
          const observation = this.extractObservationText(response.data);
          if (this.debug) console.log('📦 Raw inventory response:', observation);
          
          // 直接解析观察文本中的库存信息
          this.parseInventoryFromObservation(observation);
        } else {
          console.warn('⚠️ Failed to get inventory:', response?.error);
        }
      } catch (error) {
        console.error('❌ Error updating inventory:', error);
      } finally {
        this.loading = false;
        this.unlockOperation();
      }
    },
    
    // Method to retry loading images for all inventory items
    retryLoadImages() {
      if (!this.inventory || this.inventory.length === 0) return;
      
      console.log('🔄 Retrying image loading for all inventory items');
      
      this.inventory.forEach(item => {
        if (!item.image || item.image === null) {
          // Reset to original item texture path and try again
          const normalizedName = this.normalizeItemName(item.name || item.itemName);
          item.image = `/assets/minecraft/textures/item/${normalizedName}.png`;
          console.log(`🔄 Retrying image for ${item.name}: ${item.image}`);
        }
      });
      
      // Force re-render
      this.$forceUpdate();
    },
    
    // Lock mechanism to prevent concurrent operations
    lockOperation(reason = 'Operation') {
      if (this.isOperationLocked) {
        console.log(`⚠️ Operation already locked: ${reason}`);
        return false;
      }
      
      console.log(`🔒 Locking operations: ${reason}`);
      this.isOperationLocked = true;
      
      // Auto-unlock after timeout (safety mechanism)
      this.operationLockTimeout = setTimeout(() => {
        console.log('🔄 Auto-unlocking operation after timeout');
        this.unlockOperation();
      }, 5000); // 5 second timeout
      
      return true;
    },
    
    unlockOperation() {
      if (this.operationLockTimeout) {
        clearTimeout(this.operationLockTimeout);
        this.operationLockTimeout = null;
      }
      
      if (this.isOperationLocked) {
        console.log('🔓 Unlocking operations');
        this.isOperationLocked = false;
      }
    }
  },
  mounted() {
    console.log('📌 TextcraftViewer mounted, environment ID:', this.environmentId);
    if (this.environmentId) {
      this.initializeEnvironment();
    }
  },
  beforeUnmount() {
    console.log('📌 TextcraftViewer unmounting, environment ID:', this.environmentId);
    // Clean up any pending operations
    if (this.operationLockTimeout) {
      clearTimeout(this.operationLockTimeout);
    }
  }
}
</script>

<style scoped>
.textcraft-viewer {
  padding: 1rem;
  height: 100%;
  width: 100%;
  overflow-y: auto;
  position: relative;
  font-family: 'Nunito', 'Poppins', sans-serif;
  display: flex;
  flex-direction: column;
  background-color: #f8f9fa;
}

/* Content container */
.content-container {
  display: flex;
  gap: 1.5rem;
  flex: 1;
  min-height: 0;
}

.main-column {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  flex: 1;
  min-width: 0;
}

/* Panel styling */
.panel {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
}

/* Goal Panel */
.goal-panel {
  flex: 0 0 auto;
}

.goal-content {
  background: #f1f8e9;
  padding: 1rem;
  border-radius: 10px;
  font-weight: 600;
  color: #2e7d32;
  border-left: 5px solid #4caf50;
  font-size: 0.9rem;
}

/* Inventory Panel */
.inventory-panel {
  flex: 1;
  min-height: 0;
}

.panel-title {
  margin: 0 0 0.8rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #333;
  font-size: 1.1rem;
  font-weight: 700;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid #e9ecef;
}

.inventory-container {
  flex: 1;
  overflow-y: auto;
  min-height: 0;
}

.inventory-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 3.6rem;
  padding: 1.5rem 0;
}

.inventory-item {
  background: #f9f9f9;
  border-radius: 24px;
  padding: 3rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  transition: transform 0.2s, box-shadow 0.2s;
  border: 3px solid transparent;
}

.inventory-item:hover {
  transform: translateY(-6px);
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.2);
  border-color: #4CAF50;
}

.item-image {
  width: 216px;
  height: 216px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 2.4rem;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 18px;
  padding: 0.6rem;
  overflow: hidden;
}

.item-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  image-rendering: pixelated;
  border-radius: 12px;
}

.placeholder-image {
  width: 180px;
  height: 180px;
  background: linear-gradient(135deg, #e9ecef, #dee2e6);
  border-radius: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 5.4rem;
  font-weight: bold;
  color: #6c757d;
  border: 3px dashed #adb5bd;
}

.item-name {
  font-size: 2.7rem;
  margin-bottom: 1.2rem;
  color: #333;
  word-break: break-word;
  font-weight: 600;
  line-height: 1.3;
}

.item-count {
  font-size: 2.55rem;
  font-weight: 700;
  color: #4CAF50;
  background: rgba(76, 175, 80, 0.15);
  padding: 0.75rem 1.8rem;
  border-radius: 24px;
  border: 2px solid rgba(76, 175, 80, 0.3);
}

.empty-inventory {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 6rem 3rem;
  color: #888;
  background: #f9f9f9;
  border-radius: 18px;
  flex: 1;
  min-height: 400px;
  justify-content: center;
}

.empty-icon {
  font-size: 9rem;
  margin-bottom: 3rem;
}

/* Loading state */
.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.8);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 10;
  color: #4CAF50;
  font-weight: 600;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #e9ecef;
  border-top: 4px solid #4CAF50;
  border-radius: 50%;
  margin-bottom: 1rem;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* 刷新按钮样式 */
.refresh-inventory-btn {
  margin-left: auto;
  background: none;
  border: none;
  color: #666;
  cursor: pointer;
  font-size: 1.2rem;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: all 0.2s;
}

.refresh-inventory-btn:hover:not([disabled]) {
  background-color: #e8f5e8;
  color: #4CAF50;
  transform: rotate(90deg);
}

.refresh-inventory-btn[disabled] {
  opacity: 0.5;
  cursor: not-allowed;
}

.refresh-icon {
  font-size: 1.1rem;
}

/* Responsive design */
@media (max-width: 1600px) {
  .inventory-grid {
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 3rem;
  }
  
  .inventory-item {
    padding: 2.5rem;
  }
  
  .item-image {
    width: 180px;
    height: 180px;
    margin-bottom: 2rem;
  }
  
  .placeholder-image {
    width: 150px;
    height: 150px;
    font-size: 4.5rem;
  }
  
  .item-name {
    font-size: 2.2rem;
    margin-bottom: 1rem;
  }
  
  .item-count {
    font-size: 2rem;
    padding: 0.6rem 1.5rem;
  }
}

@media (max-width: 1200px) {
  .inventory-grid {
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 2.4rem;
  }
  
  .inventory-item {
    padding: 2rem;
  }
  
  .item-image {
    width: 144px;
    height: 144px;
    margin-bottom: 1.6rem;
    border-radius: 12px;
  }
  
  .placeholder-image {
    width: 120px;
    height: 120px;
    font-size: 3.6rem;
    border-radius: 12px;
  }
  
  .item-name {
    font-size: 1.8rem;
    margin-bottom: 0.8rem;
  }
  
  .item-count {
    font-size: 1.6rem;
    padding: 0.5rem 1.2rem;
    border-radius: 16px;
  }
}

@media (max-width: 992px) {
  .inventory-grid {
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 1.8rem;
  }
  
  .inventory-item {
    padding: 1.5rem;
    border-radius: 16px;
  }
  
  .item-image {
    width: 108px;
    height: 108px;
    margin-bottom: 1.2rem;
    border-radius: 10px;
  }
  
  .placeholder-image {
    width: 90px;
    height: 90px;
    font-size: 2.7rem;
    border-radius: 10px;
  }
  
  .item-name {
    font-size: 1.4rem;
    margin-bottom: 0.6rem;
  }
  
  .item-count {
    font-size: 1.2rem;
    padding: 0.4rem 1rem;
    border-radius: 12px;
  }
}

@media (max-width: 768px) {
  .textcraft-viewer {
    padding: 0.8rem;
  }
  
  .panel {
    padding: 1rem;
  }
  
  .inventory-grid {
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 1.2rem;
    padding: 1rem 0;
  }
  
  .inventory-item {
    padding: 1rem;
    border-radius: 12px;
  }
  
  .item-image {
    width: 84px;
    height: 84px;
    margin-bottom: 1rem;
    border-radius: 8px;
  }
  
  .placeholder-image {
    width: 70px;
    height: 70px;
    font-size: 2.1rem;
    border-radius: 8px;
  }
  
  .item-name {
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
  }
  
  .item-count {
    font-size: 1rem;
    padding: 0.3rem 0.8rem;
    border-radius: 10px;
  }
}

@media (max-width: 480px) {
  .content-container {
    flex-direction: column;
  }
  
  .inventory-grid {
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 1rem;
  }
  
  .inventory-item {
    padding: 0.8rem;
  }
  
  .item-image {
    width: 72px;
    height: 72px;
    margin-bottom: 0.8rem;
  }
  
  .placeholder-image {
    width: 60px;
    height: 60px;
    font-size: 1.8rem;
  }
  
  .item-name {
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
  }
  
  .item-count {
    font-size: 0.8rem;
    padding: 0.25rem 0.6rem;
  }
}
</style>