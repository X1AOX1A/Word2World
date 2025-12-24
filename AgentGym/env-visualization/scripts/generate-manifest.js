/**
 * Generate manifest.json from Minecraft texture pack
 * 
 * This script scans the public/assets/minecraft/textures/item directory
 * and generates a manifest.json file for the AssetManager.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Path to the Minecraft textures
const itemTexturesPath = path.join(__dirname, '../public/assets/minecraft/textures/items');
const blockTexturesPath = path.join(__dirname, '../public/assets/minecraft/textures/blocks');
const manifestPath = path.join(__dirname, '../public/assets/manifest.json');

// Category mapping based on item types
const categoryMap = {
  // Tools
  'pickaxe': 'tools',
  'axe': 'tools', 
  'shovel': 'tools',
  'hoe': 'tools',
  'sword': 'tools',
  'bow': 'tools',
  'crossbow': 'tools',
  'trident': 'tools',
  'mace': 'tools',
  'shears': 'tools',
  'flint_and_steel': 'tools',
  'fishing_rod': 'tools',
  'spyglass': 'tools',
  'brush': 'tools',

  // Armor
  'helmet': 'armor',
  'chestplate': 'armor', 
  'leggings': 'armor',
  'boots': 'armor',
  'horse_armor': 'armor',
  'wolf_armor': 'armor',

  // Weapons
  'arrow': 'combat',
  'spectral_arrow': 'combat',
  'tipped_arrow': 'combat',

  // Food
  'apple': 'food',
  'bread': 'food',
  'beef': 'food',
  'chicken': 'food',
  'porkchop': 'food',
  'mutton': 'food',
  'rabbit': 'food',
  'cod': 'food',
  'salmon': 'food',
  'potato': 'food',
  'carrot': 'food',
  'beetroot': 'food',
  'wheat': 'food',
  'seeds': 'food',
  'cookie': 'food',
  'cake': 'food',
  'pie': 'food',
  'stew': 'food',
  'soup': 'food',
  'berries': 'food',
  'kelp': 'food',
  'fruit': 'food',

  // Materials
  'log': 'materials',
  'wood': 'materials',
  'planks': 'materials',
  'stick': 'materials',
  'stone': 'materials',
  'cobblestone': 'materials',
  'iron': 'materials',
  'gold': 'materials',
  'diamond': 'materials',
  'emerald': 'materials',
  'coal': 'materials',
  'lapis': 'materials',
  'redstone': 'materials',
  'quartz': 'materials',
  'copper': 'materials',
  'netherite': 'materials',
  'ingot': 'materials',
  'nugget': 'materials',
  'dust': 'materials',
  'powder': 'materials',
  'brick': 'materials',
  'clay': 'materials',
  'leather': 'materials',
  'wool': 'materials',
  'string': 'materials',
  'feather': 'materials',
  'bone': 'materials',
  'blaze': 'materials',
  'pearl': 'materials',
  'eye': 'materials',
  'tear': 'materials',
  'membrane': 'materials',
  'shell': 'materials',
  'shard': 'materials',
  'scrap': 'materials',
  'fragment': 'materials',

  // Blocks/Building
  'door': 'building',
  'sign': 'building', 
  'boat': 'transportation',
  'minecart': 'transportation',
  'saddle': 'transportation',

  // Spawn eggs
  'spawn_egg': 'spawn_eggs',

  // Music discs
  'music_disc': 'music',

  // Potions
  'potion': 'potions',
  'bottle': 'potions',

  // Dyes
  'dye': 'dyes',

  // Default fallback
  'default': 'miscellaneous'
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

function formatItemName(fileName) {
  // Remove .png extension and convert to title case
  const baseName = fileName.replace('.png', '');
  return baseName
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function generateCompositeItems(existingItems) {
  const compositeItems = {};
  
  // Define shape suffixes and their properties
  const shapeTypes = {
    'slab': { name: 'Slab', category: 'building' },
    'stairs': { name: 'Stairs', category: 'building' },
    'button': { name: 'Button', category: 'building' },
    'pressure_plate': { name: 'Pressure Plate', category: 'building' },
    'wall': { name: 'Wall', category: 'building' },
    'fence': { name: 'Fence', category: 'building' },
    'fence_gate': { name: 'Fence Gate', category: 'building' },
    'door': { name: 'Door', category: 'building' },
    'trapdoor': { name: 'Trapdoor', category: 'building' }
  };
  
  // Define base materials that commonly have variants
  const baseMaterials = [
    'cobblestone', 'stone', 'smooth_stone', 'stone_brick',
    'oak', 'birch', 'spruce', 'jungle', 'acacia', 'dark_oak',
    'sandstone', 'red_sandstone', 'brick', 'nether_brick',
    'quartz', 'granite', 'diorite', 'andesite'
  ];
  
  baseMaterials.forEach(baseMaterial => {
    // Check if we have the base material
    const baseBlock = existingItems[baseMaterial] || existingItems[`${baseMaterial}_planks`] || existingItems[`${baseMaterial}_log`];
    
    if (baseBlock) {
      Object.entries(shapeTypes).forEach(([shapeKey, shapeInfo]) => {
        const compositeKey = `${baseMaterial}_${shapeKey}`;
        
        // Only create if it doesn't already exist
        if (!existingItems[compositeKey]) {
          const displayName = `${formatItemName(baseMaterial)} ${shapeInfo.name}`;
          
          compositeItems[compositeKey] = {
            name: displayName,
            image: baseBlock.image, // Use the base material's texture
            category: shapeInfo.category,
            description: `${displayName} made from ${formatItemName(baseMaterial)}`,
            composite: true,
            baseTexture: baseMaterial
          };
        }
      });
    }
  });
  
  return compositeItems;
}

function generateManifest() {
  console.log('🎨 Generating manifest from Minecraft texture pack...');
  
  if (!fs.existsSync(itemTexturesPath)) {
    console.error('❌ Minecraft item textures not found at:', itemTexturesPath);
    console.error('Please make sure you have copied the texture pack to public/assets/minecraft/');
    return;
  }

  if (!fs.existsSync(blockTexturesPath)) {
    console.error('❌ Minecraft block textures not found at:', blockTexturesPath);
    console.error('Please make sure you have copied the texture pack to public/assets/minecraft/');
    return;
  }

  // Get item files
  const itemFiles = fs.readdirSync(itemTexturesPath);
  const itemPngFiles = itemFiles.filter(file => file.endsWith('.png') && !file.startsWith('.'));
  
  // Get block files (include more types since we have complete resource pack)
  const blockFiles = fs.readdirSync(blockTexturesPath);
  const blockPngFiles = blockFiles.filter(file => 
    file.endsWith('.png') && 
    !file.startsWith('.') &&
    // Include common blocks that might appear in inventory
    (file.includes('cobblestone') ||
     file.includes('stone') ||
     file.includes('dirt') ||
     file.includes('wood') ||
     file.includes('log') ||
     file.includes('planks') ||
     file.includes('sand') ||
     file.includes('gravel') ||
     file.includes('ore') ||
     file.includes('brick') ||
     file.includes('quartz') ||
     file.includes('granite') ||
     file.includes('diorite') ||
     file.includes('andesite') ||
     file.includes('obsidian') ||
     file.includes('glass') ||
     file.includes('wool') ||
     file.includes('concrete') ||
     file.includes('cocoa') ||
     file.includes('leaves') ||
     file.includes('sapling') ||
     file.includes('flower') ||
     file.includes('grass') ||
     file.includes('mushroom') ||
     file.includes('cactus') ||
     file.includes('pumpkin') ||
     file.includes('melon') ||
     file.includes('wheat') ||
     file.includes('carrot') ||
     file.includes('potato') ||
     file.includes('beetroot'))
  );
  
  console.log(`📦 Found ${itemPngFiles.length} item textures and ${blockPngFiles.length} block textures`);

  const manifest = {
    generated: new Date().toISOString(),
    version: "1.0.0",
    source: "Minecraft Default Resource Pack",
    items: {},
    fallback: {
      image: "/assets/minecraft/textures/items/barrier.png",
      name: "Unknown Item",
      category: "miscellaneous",
      description: "An unknown item"
    },
    categories: {
      materials: { name: "Materials", color: "#8B4513", icon: "🧱" },
      tools: { name: "Tools", color: "#708090", icon: "🔧" },
      armor: { name: "Armor", color: "#4682B4", icon: "🛡️" },
      combat: { name: "Combat", color: "#DC143C", icon: "⚔️" },
      food: { name: "Food", color: "#32CD32", icon: "🍖" },
      building: { name: "Building", color: "#DEB887", icon: "🏗️" },
      blocks: { name: "Blocks", color: "#CD853F", icon: "🧱" },
      transportation: { name: "Transportation", color: "#FF6347", icon: "🚗" },
      spawn_eggs: { name: "Spawn Eggs", color: "#FFD700", icon: "🥚" },
      music: { name: "Music", color: "#9370DB", icon: "🎵" },
      potions: { name: "Potions", color: "#FF1493", icon: "🧪" },
      dyes: { name: "Dyes", color: "#FF69B4", icon: "🎨" },
      miscellaneous: { name: "Miscellaneous", color: "#A9A9A9", icon: "📦" }
    }
  };

  // Generate items from item textures
  itemPngFiles.forEach(fileName => {
    const itemName = fileName.replace('.png', '');
    const displayName = formatItemName(fileName);
    const category = getCategory(itemName);
    
    manifest.items[itemName] = {
      name: displayName,
      image: `/assets/minecraft/textures/items/${fileName}`,
      category: category,
      description: `${displayName} from Minecraft`
    };
  });

  // Generate items from block textures  
  blockPngFiles.forEach(fileName => {
    const itemName = fileName.replace('.png', '');
    const displayName = formatItemName(fileName);
    const category = 'blocks'; // Blocks get their own category
    
    manifest.items[itemName] = {
      name: displayName,
      image: `/assets/minecraft/textures/blocks/${fileName}`,
      category: category,
      description: `${displayName} block from Minecraft`
    };
  });

  // Special handling for cocoa beans - use the mature stage
  if (blockPngFiles.includes('cocoa_stage_2.png')) {
    manifest.items['cocoa_beans'] = {
      name: 'Cocoa Beans',
      image: '/assets/minecraft/textures/blocks/cocoa_stage_2.png',
      category: 'food',
      description: 'Cocoa Beans from Minecraft'
    };
    
    // Also add the alias
    manifest.items['cocoa'] = {
      name: 'Cocoa Beans',
      image: '/assets/minecraft/textures/blocks/cocoa_stage_2.png',
      category: 'food', 
      description: 'Cocoa Beans from Minecraft'
    };
    
    console.log('🍫 Added special handling for cocoa beans');
  }

  // Add some common aliases and variations
  const aliases = {
    'cocoa_beans': ['cocoa', 'cacao_beans'],
    'oak_log': ['oak_logs'],
    'oak_planks': ['wooden_planks', 'planks', 'wood_planks'],
    'wooden_pickaxe': ['wood_pickaxe'],
    'wooden_axe': ['wood_axe'],
    'wooden_shovel': ['wood_shovel'], 
    'wooden_sword': ['wood_sword'],
    'wooden_hoe': ['wood_hoe'],
    'blaze_powder': ['blaze_dust'],
    'gunpowder': ['sulphur'],
    'glow_berries': ['glowberries'],
    'sweet_berries': ['sweetberries']
  };

  // Add aliases to manifest
  Object.entries(aliases).forEach(([original, aliasList]) => {
    if (manifest.items[original]) {
      aliasList.forEach(alias => {
        manifest.items[alias] = { ...manifest.items[original] };
      });
    }
  });

  // Generate composite items (like cobblestone_slab, oak_stairs, etc.)
  const compositeItems = generateCompositeItems(manifest.items);
  Object.assign(manifest.items, compositeItems);
  
  console.log(`🔧 Generated ${Object.keys(compositeItems).length} composite items`);

  // Write manifest
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  
  console.log(`✅ Generated manifest with ${Object.keys(manifest.items).length} items`);
  console.log(`📝 Manifest saved to: ${manifestPath}`);
  
  // Print category statistics
  const categoryStats = {};
  Object.values(manifest.items).forEach(item => {
    categoryStats[item.category] = (categoryStats[item.category] || 0) + 1;
  });
  
  console.log('\n📊 Category breakdown:');
  Object.entries(categoryStats)
    .sort((a, b) => b[1] - a[1])
    .forEach(([category, count]) => {
      const categoryInfo = manifest.categories[category];
      console.log(`   ${categoryInfo.icon} ${categoryInfo.name}: ${count} items`);
    });
}

// Run the generator
generateManifest();

export { generateManifest };
