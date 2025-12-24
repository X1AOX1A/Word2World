/**
 * Comprehensive Minecraft Texture Manifest Generator
 * 
 * This script generates a complete manifest.json by:
 * 1. Scanning all AgentEnv-TextCraft recipes to find required items
 * 2. Loading all available textures from the Minecraft resource pack
 * 3. Creating composite textures for missing items
 * 4. Ensuring all recipe items have proper textures
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths
const itemTexturesPath = path.join(__dirname, '../public/assets/minecraft/textures/item');
const blockTexturesPath = path.join(__dirname, '../public/assets/minecraft/textures/block');
const recipesPath = path.join(__dirname, '../../../agentenv-textcraft/agentenv_textcraft/recipes');
const manifestPath = path.join(__dirname, '../public/assets/manifest.json');

// Category mapping for better organization
const categoryMap = {
  // Tools & Weapons
  'pickaxe': 'tools', 'axe': 'tools', 'shovel': 'tools', 'hoe': 'tools',
  'sword': 'tools', 'bow': 'tools', 'crossbow': 'tools', 'trident': 'tools',
  'shears': 'tools', 'flint_and_steel': 'tools', 'fishing_rod': 'tools',
  
  // Armor
  'helmet': 'armor', 'chestplate': 'armor', 'leggings': 'armor', 'boots': 'armor',
  'horse_armor': 'armor',
  
  // Food
  'apple': 'food', 'bread': 'food', 'beef': 'food', 'chicken': 'food',
  'porkchop': 'food', 'mutton': 'food', 'rabbit': 'food', 'cod': 'food',
  'salmon': 'food', 'potato': 'food', 'carrot': 'food', 'beetroot': 'food',
  'wheat': 'food', 'cookie': 'food', 'cake': 'food', 'pie': 'food',
  'stew': 'food', 'soup': 'food', 'berries': 'food', 'melon': 'food',
  
  // Materials
  'log': 'materials', 'wood': 'materials', 'planks': 'materials',
  'stick': 'materials', 'cobblestone': 'materials', 'stone': 'materials',
  'iron': 'materials', 'gold': 'materials', 'diamond': 'materials',
  'emerald': 'materials', 'coal': 'materials', 'lapis': 'materials',
  'redstone': 'materials', 'quartz': 'materials', 'netherite': 'materials',
  'ingot': 'materials', 'nugget': 'materials', 'dust': 'materials',
  'powder': 'materials', 'brick': 'materials', 'leather': 'materials',
  'string': 'materials', 'feather': 'materials', 'bone': 'materials',
  
  // Building blocks
  'slab': 'building', 'stairs': 'building', 'wall': 'building',
  'fence': 'building', 'gate': 'building', 'door': 'building',
  'trapdoor': 'building', 'button': 'building', 'plate': 'building',
  'glass': 'building', 'wool': 'building', 'concrete': 'building',
  
  // Transportation
  'boat': 'transportation', 'minecart': 'transportation', 'rail': 'transportation',
  
  // Dyes & Colors
  'dye': 'dyes', 'banner': 'dyes', 'bed': 'dyes', 'carpet': 'dyes',
  'stained': 'dyes', 'glazed': 'dyes', 'terracotta': 'dyes',
  
  // Redstone & Technical
  'redstone': 'redstone', 'comparator': 'redstone', 'repeater': 'redstone',
  'piston': 'redstone', 'dispenser': 'redstone', 'dropper': 'redstone',
  'hopper': 'redstone', 'observer': 'redstone', 'lever': 'redstone',
  
  // Magic & Brewing
  'potion': 'potions', 'bottle': 'potions', 'eye': 'potions',
  'blaze': 'potions', 'tear': 'potions', 'spider': 'potions',
  
  // Default
  'spawn_egg': 'spawn_eggs', 'music_disc': 'music'
};

function getCategory(itemName) {
  // Check each category pattern
  for (const [pattern, category] of Object.entries(categoryMap)) {
    if (itemName.includes(pattern)) {
      return category;
    }
  }
  return 'miscellaneous';
}

function formatItemName(name) {
  return name
    .replace(/^minecraft:/, '')
    .replace(/_/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Scan all recipe files to extract required items
 */
function scanRecipeItems() {
  console.log('📖 Scanning AgentEnv-TextCraft recipes...');
  
  if (!fs.existsSync(recipesPath)) {
    console.warn('⚠️  Recipes path not found:', recipesPath);
    return new Set();
  }
  
  const recipeFiles = fs.readdirSync(recipesPath).filter(f => f.endsWith('.json'));
  const requiredItems = new Set();
  
  recipeFiles.forEach(fileName => {
    try {
      const filePath = path.join(recipesPath, fileName);
      const recipe = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      
      // Extract result item
      if (recipe.result) {
        const resultItem = recipe.result.item || recipe.result;
        if (typeof resultItem === 'string') {
          const cleanItem = resultItem.replace('minecraft:', '');
          requiredItems.add(cleanItem);
        }
      }
      
      // Extract ingredient items
      if (recipe.key) {
        Object.values(recipe.key).forEach(ingredient => {
          if (ingredient.item) {
            const cleanItem = ingredient.item.replace('minecraft:', '');
            requiredItems.add(cleanItem);
          }
        });
      }
      
      if (recipe.ingredient) {
        const ingredient = Array.isArray(recipe.ingredient) ? recipe.ingredient[0] : recipe.ingredient;
        if (ingredient.item) {
          const cleanItem = ingredient.item.replace('minecraft:', '');
          requiredItems.add(cleanItem);
        }
      }
      
      if (recipe.ingredients) {
        recipe.ingredients.forEach(ingredient => {
          if (ingredient.item) {
            const cleanItem = ingredient.item.replace('minecraft:', '');
            requiredItems.add(cleanItem);
          }
        });
      }
      
    } catch (error) {
      console.warn(`⚠️  Error reading recipe ${fileName}:`, error.message);
    }
  });
  
  console.log(`📦 Found ${requiredItems.size} unique items in ${recipeFiles.length} recipes`);
  return requiredItems;
}

/**
 * Load available textures from the resource pack
 */
function loadAvailableTextures() {
  console.log('🎨 Loading available textures...');
  
  const textures = {
    items: new Map(),
    blocks: new Map()
  };
  
  // Load item textures
  if (fs.existsSync(itemTexturesPath)) {
    const itemFiles = fs.readdirSync(itemTexturesPath);
    itemFiles.filter(f => f.endsWith('.png')).forEach(fileName => {
      const itemName = fileName.replace('.png', '');
      textures.items.set(itemName, `/assets/minecraft/textures/item/${fileName}`);
    });
    console.log(`   📦 Loaded ${textures.items.size} item textures`);
  }
  
  // Load block textures
  if (fs.existsSync(blockTexturesPath)) {
    const blockFiles = fs.readdirSync(blockTexturesPath);
    blockFiles.filter(f => f.endsWith('.png')).forEach(fileName => {
      const itemName = fileName.replace('.png', '');
      textures.blocks.set(itemName, `/assets/minecraft/textures/block/${fileName}`);
    });
    console.log(`   🧱 Loaded ${textures.blocks.size} block textures`);
  }
  
  return textures;
}

/**
 * Generate composite items for missing textures
 */
function generateCompositeItems(requiredItems, availableTextures, manifest) {
  console.log('🔧 Generating composite items...');
  
  const generatedItems = {};
  let generatedCount = 0;
  
  // Shape patterns for composite items
  const shapePatterns = {
    '_slab': { base: '', suffix: '_slab', name: 'Slab', category: 'building' },
    '_stairs': { base: '', suffix: '_stairs', name: 'Stairs', category: 'building' },
    '_wall': { base: '', suffix: '_wall', name: 'Wall', category: 'building' },
    '_fence': { base: '', suffix: '_fence', name: 'Fence', category: 'building' },
    '_fence_gate': { base: '', suffix: '_fence_gate', name: 'Fence Gate', category: 'building' },
    '_button': { base: '', suffix: '_button', name: 'Button', category: 'building' },
    '_pressure_plate': { base: '', suffix: '_pressure_plate', name: 'Pressure Plate', category: 'building' },
    '_door': { base: '', suffix: '_door', name: 'Door', category: 'building' },
    '_trapdoor': { base: '', suffix: '_trapdoor', name: 'Trapdoor', category: 'building' },
    '_sign': { base: '', suffix: '_sign', name: 'Sign', category: 'building' },
    '_boat': { base: '', suffix: '_boat', name: 'Boat', category: 'transportation' }
  };
  
  // Wood type mappings - use log texture for wood items
  const woodMappings = {
    'acacia': 'acacia_log',
    'birch': 'birch_log', 
    'dark_oak': 'dark_oak_log',
    'jungle': 'jungle_log',
    'oak': 'oak_log',
    'spruce': 'spruce_log',
    'mangrove': 'mangrove_log',
    'cherry': 'cherry_log',
    'bamboo': 'bamboo_block',
    'crimson': 'crimson_stem',
    'warped': 'warped_stem'
  };
  
  // Color variants mapping 
  const colorVariants = {
    'white': 'white_wool',
    'orange': 'orange_wool', 
    'magenta': 'magenta_wool',
    'light_blue': 'light_blue_wool',
    'yellow': 'yellow_wool',
    'lime': 'lime_wool',
    'pink': 'pink_wool',
    'gray': 'gray_wool',
    'light_gray': 'light_gray_wool',
    'cyan': 'cyan_wool',
    'purple': 'purple_wool',
    'blue': 'blue_wool',
    'brown': 'brown_wool',
    'green': 'green_wool',
    'red': 'red_wool',
    'black': 'black_wool'
  };
  
  // Items that use color variants
  const coloredItems = ['banner', 'bed', 'carpet', 'concrete', 'concrete_powder', 
                       'glazed_terracotta', 'shulker_box', 'stained_glass', 
                       'stained_glass_pane', 'terracotta', 'wool', 'candle'];
                       
  // Special item mappings
  const specialMappings = {
    'barrel': 'barrel_side',
    'composter': 'composter_side',
    'beehive': 'bee_nest_side',
    'lectern': 'lectern_base',
    'stonecutter': 'stonecutter_top',
    'cartography_table': 'cartography_table_top',
    'fletching_table': 'fletching_table_top',
    'smithing_table': 'smithing_table_top',
    'loom': 'loom_top',
    'cauldron': 'cauldron_inner',
    'hopper': 'hopper_outside',
    'furnace': 'furnace_front',
    'blast_furnace': 'blast_furnace_front',
    'smoker': 'smoker_front',
    'dispenser': 'dispenser_front',
    'dropper': 'dropper_front',
    'observer': 'observer_front',
    'piston': 'piston_top',
    'sticky_piston': 'piston_top_sticky',
    'redstone_lamp': 'redstone_lamp_off',
    'daylight_detector': 'daylight_detector_top',
    'tripwire_hook': 'trip_wire_source',
    'repeater': 'repeater_off',
    'comparator': 'comparator_off',
    'target': 'target_top',
    'lightning_rod': 'lightning_rod',
    'respawn_anchor': 'respawn_anchor_bottom',
    'lodestone': 'lodestone_top',
    'bell': 'bell_bottom',
    'lantern': 'lantern',
    'chain': 'chain',
    'flower_pot': 'flower_pot',
    'item_frame': 'item_frame',
    'painting': 'painting_back',
    'armor_stand': 'item_frame', // fallback
    'end_rod': 'end_rod',
    'chorus_plant': 'chorus_plant',
    'scaffolding': 'scaffolding_top',
    'honeycomb_block': 'honeycomb_block',
    'soul_campfire': 'soul_campfire',
    'campfire': 'campfire',
    'grindstone': 'grindstone_round',
    'anvil': 'anvil_top',
    'enchanting_table': 'enchanting_table_top',
    'ender_chest': 'ender_chest_top',
    'brewing_stand': 'brewing_stand_base',
    'cake': 'cake_top',
    'note_block': 'note_block',
    'jukebox': 'jukebox_top',
    'glass_pane': 'glass',
    'iron_bars': 'iron_bars',
    'cobweb': 'cobweb',
    'grass': 'grass_block_top',
    'dirt_path': 'dirt_path_top',
    'mycelium': 'mycelium_top',
    'podzol': 'podzol_top',
    'coarse_dirt': 'coarse_dirt',
    'farmland': 'farmland_moist',
    'grass_path': 'dirt_path_top'
  };
  
  requiredItems.forEach(itemName => {
    // Skip if already exists in manifest
    if (manifest.items[itemName]) return;
    
    // Check if it's available as texture
    if (availableTextures.items.has(itemName)) {
      manifest.items[itemName] = {
        name: formatItemName(itemName),
        image: availableTextures.items.get(itemName),
        category: getCategory(itemName),
        description: `${formatItemName(itemName)} from Minecraft`
      };
      return;
    }
    
    if (availableTextures.blocks.has(itemName)) {
      manifest.items[itemName] = {
        name: formatItemName(itemName),
        image: availableTextures.blocks.get(itemName),
        category: getCategory(itemName),
        description: `${formatItemName(itemName)} from Minecraft`
      };
      return;
    }
    
    // Check special mappings first
    if (specialMappings[itemName]) {
      const mappedName = specialMappings[itemName];
      let mappedTexture = null;
      
      if (availableTextures.blocks.has(mappedName)) {
        mappedTexture = availableTextures.blocks.get(mappedName);
      } else if (availableTextures.items.has(mappedName)) {
        mappedTexture = availableTextures.items.get(mappedName);
      }
      
      if (mappedTexture) {
        generatedItems[itemName] = {
          name: formatItemName(itemName),
          image: mappedTexture,
          category: getCategory(itemName),
          description: `${formatItemName(itemName)} - Mapped texture`,
          composite: true,
          baseTexture: mappedName
        };
        generatedCount++;
        console.log(`   🎯 Mapped: ${itemName} to ${mappedName}`);
        return;
      }
    }
    
    // Try wood variants (acacia_button -> acacia_log)
    for (const [woodType, logName] of Object.entries(woodMappings)) {
      if (itemName.startsWith(woodType + '_')) {
        let baseTexture = null;
        if (availableTextures.blocks.has(logName)) {
          baseTexture = availableTextures.blocks.get(logName);
        } else if (availableTextures.items.has(logName)) {
          baseTexture = availableTextures.items.get(logName);
        }
        
        if (baseTexture) {
          const itemType = itemName.replace(woodType + '_', '');
          generatedItems[itemName] = {
            name: `${formatItemName(woodType)} ${formatItemName(itemType)}`,
            image: baseTexture,
            category: getCategory(itemName),
            description: `${formatItemName(woodType)} ${formatItemName(itemType)} - Generated from ${woodType} wood`,
            composite: true,
            baseTexture: logName
          };
          generatedCount++;
          console.log(`   🌳 Wood variant: ${itemName} from ${logName}`);
          return;
        }
      }
    }
    
    // Try color variants (black_bed -> black_wool + bed)
    for (const color of Object.keys(colorVariants)) {
      if (itemName.startsWith(color + '_')) {
        const itemType = itemName.replace(color + '_', '');
        
        if (coloredItems.includes(itemType)) {
          // Use colored wool as base texture
          const woolTexture = colorVariants[color];
          let baseTexture = null;
          
          if (availableTextures.blocks.has(woolTexture)) {
            baseTexture = availableTextures.blocks.get(woolTexture);
          } else if (availableTextures.items.has(woolTexture)) {
            baseTexture = availableTextures.items.get(woolTexture);
          }
          
          if (baseTexture) {
            generatedItems[itemName] = {
              name: `${formatItemName(color)} ${formatItemName(itemType)}`,
              image: baseTexture,
              category: getCategory(itemType),
              description: `${formatItemName(color)} ${formatItemName(itemType)} - Generated from ${color} wool`,
              composite: true,
              baseTexture: woolTexture
            };
            generatedCount++;
            console.log(`   🎨 Color variant: ${itemName} from ${woolTexture}`);
            return;
          }
        }
      }
    }
    
    // Try to generate composite items
    for (const [pattern, info] of Object.entries(shapePatterns)) {
      if (itemName.endsWith(pattern)) {
        const baseName = itemName.replace(pattern, '');
        
        // Look for base texture
        let baseTexture = null;
        if (availableTextures.blocks.has(baseName)) {
          baseTexture = availableTextures.blocks.get(baseName);
        } else if (availableTextures.items.has(baseName)) {
          baseTexture = availableTextures.items.get(baseName);
        }
        
        if (baseTexture) {
          generatedItems[itemName] = {
            name: `${formatItemName(baseName)} ${info.name}`,
            image: baseTexture,
            category: info.category,
            description: `${formatItemName(baseName)} ${info.name} - Generated from base texture`,
            composite: true,
            baseTexture: baseName
          };
          generatedCount++;
          console.log(`   ➕ Generated: ${itemName} from ${baseName}`);
        }
        break;
      }
    }
    
    // Try alternative names and patterns
    const alternativeNames = [
      itemName.replace('_', ''),
      itemName + 's',
      itemName.replace(/s$/, ''),
      'minecraft_' + itemName,
      itemName.replace(/^minecraft_/, ''),
      itemName.split('_').reverse().join('_'), // reverse order
      itemName.replace(/([a-z])([A-Z])/g, '$1_$2').toLowerCase()
    ];
    
    for (const altName of alternativeNames) {
      if (availableTextures.items.has(altName) || availableTextures.blocks.has(altName)) {
        const texture = availableTextures.items.get(altName) || availableTextures.blocks.get(altName);
        generatedItems[itemName] = {
          name: formatItemName(itemName),
          image: texture,
          category: getCategory(itemName),
          description: `${formatItemName(itemName)} - Alternative mapping`,
          composite: true,
          baseTexture: altName
        };
        generatedCount++;
        console.log(`   🔄 Alternative: ${itemName} -> ${altName}`);
        break;
      }
    }
  });
  
  // Add generated items to manifest
  Object.assign(manifest.items, generatedItems);
  
  console.log(`🔧 Generated ${generatedCount} composite items`);
  return generatedCount;
}

/**
 * Main generation function
 */
function generateComprehensiveManifest() {
  console.log('🎯 Generating comprehensive texture manifest...');
  
  // Step 1: Scan recipes for required items
  const requiredItems = scanRecipeItems();
  
  // Step 2: Load available textures
  const availableTextures = loadAvailableTextures();
  
  // Step 3: Initialize manifest
  const manifest = {
    generated: new Date().toISOString(),
    version: "2.0.0",
    source: "Minecraft Default Resource Pack + AgentEnv-TextCraft Recipes",
    totalItems: 0,
    requiredItems: Array.from(requiredItems),
    items: {},
    fallback: {
      image: "/assets/minecraft/textures/item/barrier.png",
      name: "Unknown Item",
      category: "miscellaneous",
      description: "An unknown item"
    },
    categories: {
      materials: { name: "Materials", color: "#8B4513", icon: "🧱" },
      tools: { name: "Tools", color: "#708090", icon: "🔧" },
      armor: { name: "Armor", color: "#4682B4", icon: "🛡️" },
      food: { name: "Food", color: "#32CD32", icon: "🍖" },
      building: { name: "Building", color: "#DEB887", icon: "🏗️" },
      transportation: { name: "Transportation", color: "#FF6347", icon: "🚗" },
      dyes: { name: "Dyes", color: "#FF69B4", icon: "🎨" },
      redstone: { name: "Redstone", color: "#DC143C", icon: "🔴" },
      potions: { name: "Potions", color: "#FF1493", icon: "🧪" },
      spawn_eggs: { name: "Spawn Eggs", color: "#FFD700", icon: "🥚" },
      music: { name: "Music", color: "#9370DB", icon: "🎵" },
      miscellaneous: { name: "Miscellaneous", color: "#A9A9A9", icon: "📦" }
    }
  };
  
  // Step 4: Add all available item textures
  console.log('📦 Adding item textures...');
  availableTextures.items.forEach((imagePath, itemName) => {
    manifest.items[itemName] = {
      name: formatItemName(itemName),
      image: imagePath,
      category: getCategory(itemName),
      description: `${formatItemName(itemName)} from Minecraft`
    };
  });
  
  // Step 5: Add selected block textures (commonly used as items)
  console.log('🧱 Adding block textures...');
  availableTextures.blocks.forEach((imagePath, blockName) => {
    // Only add blocks that are commonly used as items or are in recipes
    if (requiredItems.has(blockName) || 
        blockName.includes('ore') || blockName.includes('log') || 
        blockName.includes('planks') || blockName.includes('stone') ||
        blockName.includes('dirt') || blockName.includes('sand') ||
        blockName.includes('gravel') || blockName.includes('glass') ||
        blockName.includes('wool') || blockName.includes('concrete')) {
      
      // Don't override if item texture already exists
      if (!manifest.items[blockName]) {
        manifest.items[blockName] = {
          name: formatItemName(blockName),
          image: imagePath,
          category: getCategory(blockName),
          description: `${formatItemName(blockName)} block from Minecraft`
        };
      }
    }
  });
  
  // Step 6: Generate composite items for missing textures
  const compositeCount = generateCompositeItems(requiredItems, availableTextures, manifest);
  
  // Step 7: Add common aliases
  const aliases = {
    'cocoa_beans': ['cocoa', 'cacao_beans'],
    'wooden_pickaxe': ['wood_pickaxe'],
    'wooden_axe': ['wood_axe'],
    'wooden_shovel': ['wood_shovel'],
    'wooden_sword': ['wood_sword'],
    'wooden_hoe': ['wood_hoe'],
    'oak_planks': ['planks', 'wooden_planks'],
    'gunpowder': ['sulphur'],
    'gold_ingot': ['gold'],
    'iron_ingot': ['iron'],
    'diamond': ['diamond_gem']
  };
  
  Object.entries(aliases).forEach(([original, aliasList]) => {
    if (manifest.items[original]) {
      aliasList.forEach(alias => {
        manifest.items[alias] = { ...manifest.items[original] };
      });
    }
  });
  
  // Step 8: Final statistics
  manifest.totalItems = Object.keys(manifest.items).length;
  
  const categoryStats = {};
  Object.values(manifest.items).forEach(item => {
    categoryStats[item.category] = (categoryStats[item.category] || 0) + 1;
  });
  
  // Step 9: Write manifest
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  
  console.log(`✅ Generated comprehensive manifest with ${manifest.totalItems} items`);
  console.log(`📝 Manifest saved to: ${manifestPath}`);
  console.log(`🔧 Generated ${compositeCount} composite items`);
  
  // Check coverage
  const missingItems = [];
  requiredItems.forEach(item => {
    if (!manifest.items[item]) {
      missingItems.push(item);
    }
  });
  
  if (missingItems.length > 0) {
    console.log(`⚠️  Missing textures for ${missingItems.length} items:`);
    missingItems.slice(0, 10).forEach(item => console.log(`   - ${item}`));
    if (missingItems.length > 10) {
      console.log(`   ... and ${missingItems.length - 10} more`);
    }
  } else {
    console.log('🎉 All recipe items have textures!');
  }
  
  console.log('\\n📊 Category breakdown:');
  Object.entries(categoryStats)
    .sort((a, b) => b[1] - a[1])
    .forEach(([category, count]) => {
      const categoryInfo = manifest.categories[category];
      console.log(`   ${categoryInfo.icon} ${categoryInfo.name}: ${count} items`);
    });
}

// Run the generator
generateComprehensiveManifest();
