/**
 * Asset Manager Service
 * 
 * Manages visual assets for TextCraft items, including loading manifests,
 * parsing item names from environment state, and providing fallback images.
 */

class AssetManager {
  constructor() {
    this.manifest = null
    this.loaded = false
  }

  /**
   * Load the asset manifest from public/assets/manifest.json
   */
  async loadManifest() {
    if (this.loaded) return this.manifest

    try {
      const response = await fetch('/assets/manifest.json')
      if (!response.ok) {
        throw new Error(`Failed to load manifest: ${response.status}`)
      }
      
      this.manifest = await response.json()
      this.loaded = true
      console.log('Asset manifest loaded successfully:', this.manifest)
      return this.manifest
    } catch (error) {
      console.error('Error loading asset manifest:', error)
      // Provide fallback manifest
      this.manifest = {
        items: {},
        fallback: {
          image: '/assets/minecraft/textures/items/barrier.png',
          name: 'Unknown Item',
          category: 'miscellaneous',
          description: 'Item not found in texture pack'
        },
        categories: {
          miscellaneous: { name: 'Miscellaneous', color: '#A9A9A9', icon: '📦' }
        }
      }
      this.loaded = true
      return this.manifest
    }
  }

  /**
   * Get item information by item name
   * @param {string} itemName - The name of the item (e.g., 'wooden_pickaxe')
   * @returns {object} Item information with image path, name, description, etc.
   */
  getItem(itemName) {
    if (!this.loaded) {
      console.warn('Asset manifest not loaded. Call loadManifest() first.')
      return this.manifest?.fallback || { 
        image: '/assets/minecraft/textures/item/barrier.png', 
        name: 'Unknown Item',
        category: 'miscellaneous',
        description: 'Manifest not loaded'
      }
    }

    // Normalize item name (remove minecraft: prefix, convert to lowercase, handle spaces)
    let normalizedName = itemName
      .replace(/^minecraft:/, '')  // Remove minecraft: prefix
      .replace(/\s+/g, '_')        // Convert spaces to underscores
      .toLowerCase()
    
    // Check direct match first
    let item = this.manifest.items[normalizedName]
    
    if (!item) {
      // Try common aliases and variations
      const aliases = {
        'cocoa': 'cocoa_beans',
        'cacao_beans': 'cocoa_beans',
        'wooden_pickaxe': 'wood_pickaxe',
        'wooden_axe': 'wood_axe',
        'wooden_shovel': 'wood_shovel',
        'wooden_sword': 'wood_sword',
        'wooden_hoe': 'wood_hoe',
        'oak_logs': 'oak_log',
        'wooden_planks': 'oak_planks',
        'planks': 'oak_planks',
        'wood_planks': 'oak_planks'
      }
      
      if (aliases[normalizedName]) {
        item = this.manifest.items[aliases[normalizedName]]
        normalizedName = aliases[normalizedName]
      }
    }
    
    if (!item) {
      // Try removing plural 's' or adding it
      const singularName = normalizedName.replace(/s$/, '')
      const pluralName = normalizedName + 's'
      
      item = this.manifest.items[singularName] || this.manifest.items[pluralName]
      if (item) {
        normalizedName = this.manifest.items[singularName] ? singularName : pluralName
      }
    }
    
    if (item) {
      return {
        ...item,
        originalName: itemName,
        normalizedName
      }
    }

    // Return fallback for unknown items
    console.log(`🔍 Item not found: "${itemName}" (normalized: "${normalizedName}")`)
    return {
      ...this.manifest.fallback,
      originalName: itemName,
      normalizedName,
      name: itemName.replace(/^minecraft:/, '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    }
  }

  /**
   * Parse environment observation text to extract inventory items
   * @param {string} observation - The environment observation text
   * @returns {array} Array of inventory items with counts and metadata
   */
  parseInventory(observation) {
    if (!observation) {
      return []
    }
    
    // Convert observation to string if it's not already
    const obsStr = typeof observation === 'string' 
      ? observation 
      : JSON.stringify(observation)

    console.log('🔍 Parsing inventory from:', obsStr.substring(0, 100) + (obsStr.length > 100 ? '...' : ''));

    const inventory = []

    // Parse TextCraft inventory format: "Inventory: [item name] (count) [item2 name] (count2)"
    const inventoryMatch = obsStr.match(/Inventory:\s*(.+?)(?:\n|$)/i)
    if (inventoryMatch) {
      const inventoryText = inventoryMatch[1]
      console.log('📋 Extracted inventory text:', inventoryText);
      
      // Check if empty inventory
      if (inventoryText.includes("You are not carrying anything") || 
          inventoryText.trim() === '' || 
          inventoryText.includes('empty') ||
          inventoryText.includes('nothing')) {
        console.log('🔄 Empty inventory detected');
        return []
      }
      
      // Parse items in format: [item name] (count)
      const itemMatches = Array.from(inventoryText.matchAll(/\[([^\]]+)\]\s*\((\d+)\)/g))
      console.log('🔢 Found item matches:', itemMatches.length);
      
      for (const match of itemMatches) {
        const itemName = match[1].trim().replace(/\s+/g, '_').toLowerCase()
        const count = parseInt(match[2])
        const itemData = this.getItem(itemName)
        
        inventory.push({
          count,
          itemName,
          ...itemData
        })
        
        console.log(`📦 Added to inventory: ${count}x ${itemName}`);
      }
      
      // If no matches found with the pattern above, try alternative pattern
      if (inventory.length === 0) {
        // Try to match format: "item_name: count"
        const altItemMatches = Array.from(inventoryText.matchAll(/([a-zA-Z_\s]+):\s*(\d+)/g))
        for (const match of altItemMatches) {
          const itemName = match[1].trim().replace(/\s+/g, '_').toLowerCase()
          const count = parseInt(match[2])
          const itemData = this.getItem(itemName)
          
          inventory.push({
            count,
            itemName,
            ...itemData
          })
          
          console.log(`📦 Added to inventory (alt pattern): ${count}x ${itemName}`);
        }
      }
    }

    // Also check for "Got X item_name" format
    const gotMatches = Array.from(obsStr.matchAll(/Got\s+(\d+)\s+(.+?)(?:\n|$)/gi))
    for (const match of gotMatches) {
      const count = parseInt(match[1])
      let itemName = match[2].trim()
      
      // Remove minecraft: prefix if present
      if (itemName.startsWith('minecraft:')) {
        itemName = itemName.substring(10)
      }
      
      itemName = itemName.replace(/\s+/g, '_').toLowerCase()
      const itemData = this.getItem(itemName)
      
      // Check if item already exists in inventory
      const existingIndex = inventory.findIndex(inv => inv.itemName === itemName)
      if (existingIndex >= 0) {
        inventory[existingIndex].count += count
      } else {
        inventory.push({
          count,
          itemName,
          ...itemData
        })
      }
      
      console.log(`📦 Added to inventory (got pattern): ${count}x ${itemName}`);
    }

    console.log(`📊 Final inventory contains ${inventory.length} items`);
    return inventory
  }

  /**
   * Parse environment observation to extract visible world items
   * @param {string} observation - The environment observation text  
   * @returns {array} Array of world items that can be interacted with
   */
  parseWorldItems(observation) {
    if (!observation) {
      return []
    }
    
    // Convert observation to string if it's not already
    const obsStr = typeof observation === 'string' 
      ? observation 
      : JSON.stringify(observation)

    const worldItems = []

    // Look for items that can be mined or interacted with
    const mineablePattern = /(?:you see|there (?:is|are)|you can mine)\s+([a-zA-Z_:]+)/gi
    const matches = obsStr.matchAll(mineablePattern)
    
    for (const match of matches) {
      const itemName = match[1]
      const itemData = this.getItem(itemName)
      
      worldItems.push({
        itemName,
        ...itemData,
        interactionType: 'mine'
      })
    }

    return worldItems
  }

  /**
   * Get category information
   * @param {string} categoryName - Name of the category
   * @returns {object} Category information
   */
  getCategory(categoryName) {
    return this.manifest?.categories?.[categoryName] || { name: categoryName, color: '#666' }
  }

  /**
   * Group inventory items by category
   * @param {array} inventory - Array of inventory items
   * @returns {object} Inventory grouped by category
   */
  groupByCategory(inventory) {
    const grouped = {}
    
    for (const item of inventory) {
      const category = item.category || 'misc'
      if (!grouped[category]) {
        grouped[category] = {
          info: this.getCategory(category),
          items: []
        }
      }
      grouped[category].items.push(item)
    }

    return grouped
  }
}

// Export singleton instance
export default new AssetManager()
