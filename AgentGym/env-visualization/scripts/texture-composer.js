/**
 * Texture Composer for Minecraft Items
 * 
 * This script generates missing textures by compositing base materials
 * with shape templates (slab, stairs, button, etc.)
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Base textures path
const blockTexturesPath = path.join(__dirname, '../public/assets/minecraft/textures/block');
const itemTexturesPath = path.join(__dirname, '../public/assets/minecraft/textures/item');
const generatedTexturesPath = path.join(__dirname, '../public/assets/generated');

// Shape templates (we'll create these as simple overlays)
const SHAPE_TEMPLATES = {
  slab: {
    name: 'Slab',
    description: 'A half-height block',
    overlay: 'slab_overlay.png',
    category: 'building'
  },
  stairs: {
    name: 'Stairs',
    description: 'Stepped blocks for climbing',
    overlay: 'stairs_overlay.png',
    category: 'building'
  },
  button: {
    name: 'Button',
    description: 'A pressable button',
    overlay: 'button_overlay.png',
    category: 'building'
  },
  pressure_plate: {
    name: 'Pressure Plate',
    description: 'A pressure-sensitive plate',
    overlay: 'pressure_plate_overlay.png',
    category: 'building'
  },
  wall: {
    name: 'Wall',
    description: 'A defensive wall',
    overlay: 'wall_overlay.png',
    category: 'building'
  },
  fence: {
    name: 'Fence',
    description: 'A barrier fence',
    overlay: 'fence_overlay.png',
    category: 'building'
  },
  fence_gate: {
    name: 'Fence Gate',
    description: 'An openable fence gate',
    overlay: 'fence_gate_overlay.png',
    category: 'building'
  }
};

// Base materials that can be combined with shapes
const BASE_MATERIALS = [
  'cobblestone', 'stone', 'smooth_stone', 'stone_bricks',
  'oak_planks', 'birch_planks', 'spruce_planks', 'jungle_planks',
  'acacia_planks', 'dark_oak_planks', 'cherry_planks', 'mangrove_planks',
  'bamboo_planks', 'crimson_planks', 'warped_planks',
  'sandstone', 'red_sandstone', 'brick', 'nether_brick',
  'quartz_block', 'purpur_block', 'prismarine', 'dark_prismarine',
  'blackstone', 'polished_blackstone', 'polished_blackstone_bricks',
  'granite', 'diorite', 'andesite', 'polished_granite',
  'polished_diorite', 'polished_andesite', 'deepslate',
  'polished_deepslate', 'deepslate_bricks'
];

/**
 * Create SVG overlay templates for different shapes
 */
function createShapeOverlays() {
  if (!fs.existsSync(generatedTexturesPath)) {
    fs.mkdirSync(generatedTexturesPath, { recursive: true });
  }

  const overlays = {
    slab_overlay: `
      <svg width="16" height="16" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="slabPattern" x="0" y="0" width="16" height="16" patternUnits="userSpaceOnUse">
            <rect x="0" y="8" width="16" height="8" fill="rgba(0,0,0,0.2)"/>
            <rect x="0" y="7" width="16" height="1" fill="rgba(255,255,255,0.3)"/>
          </pattern>
        </defs>
        <rect width="16" height="16" fill="url(#slabPattern)"/>
      </svg>
    `,
    stairs_overlay: `
      <svg width="16" height="16" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="stairPattern" x="0" y="0" width="16" height="16" patternUnits="userSpaceOnUse">
            <polygon points="0,16 8,16 8,8 16,8 16,0 16,16" fill="rgba(0,0,0,0.2)"/>
            <polyline points="0,8 8,8 8,0" stroke="rgba(255,255,255,0.3)" stroke-width="1" fill="none"/>
          </pattern>
        </defs>
        <rect width="16" height="16" fill="url(#stairPattern)"/>
      </svg>
    `,
    button_overlay: `
      <svg width="16" height="16" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="buttonPattern" x="0" y="0" width="16" height="16" patternUnits="userSpaceOnUse">
            <rect x="4" y="6" width="8" height="4" fill="rgba(0,0,0,0.2)" rx="1"/>
            <rect x="4" y="5" width="8" height="1" fill="rgba(255,255,255,0.4)"/>
          </pattern>
        </defs>
        <rect width="16" height="16" fill="url(#buttonPattern)"/>
      </svg>
    `,
    pressure_plate_overlay: `
      <svg width="16" height="16" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="platePattern" x="0" y="0" width="16" height="16" patternUnits="userSpaceOnUse">
            <rect x="1" y="7" width="14" height="2" fill="rgba(0,0,0,0.2)" rx="1"/>
            <rect x="1" y="6" width="14" height="1" fill="rgba(255,255,255,0.3)"/>
          </pattern>
        </defs>
        <rect width="16" height="16" fill="url(#platePattern)"/>
      </svg>
    `,
    wall_overlay: `
      <svg width="16" height="16" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="wallPattern" x="0" y="0" width="16" height="16" patternUnits="userSpaceOnUse">
            <rect x="4" y="0" width="8" height="16" fill="rgba(0,0,0,0.15)"/>
            <rect x="0" y="4" width="16" height="8" fill="rgba(0,0,0,0.15)"/>
            <rect x="3" y="0" width="1" height="16" fill="rgba(255,255,255,0.2)"/>
            <rect x="0" y="3" width="16" height="1" fill="rgba(255,255,255,0.2)"/>
          </pattern>
        </defs>
        <rect width="16" height="16" fill="url(#wallPattern)"/>
      </svg>
    `,
    fence_overlay: `
      <svg width="16" height="16" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="fencePattern" x="0" y="0" width="16" height="16" patternUnits="userSpaceOnUse">
            <rect x="6" y="0" width="4" height="16" fill="rgba(0,0,0,0.2)"/>
            <rect x="0" y="4" width="16" height="2" fill="rgba(0,0,0,0.15)"/>
            <rect x="0" y="10" width="16" height="2" fill="rgba(0,0,0,0.15)"/>
            <rect x="5" y="0" width="1" height="16" fill="rgba(255,255,255,0.3)"/>
          </pattern>
        </defs>
        <rect width="16" height="16" fill="url(#fencePattern)"/>
      </svg>
    `,
    fence_gate_overlay: `
      <svg width="16" height="16" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="gatePattern" x="0" y="0" width="16" height="16" patternUnits="userSpaceOnUse">
            <rect x="2" y="0" width="2" height="16" fill="rgba(0,0,0,0.2)"/>
            <rect x="12" y="0" width="2" height="16" fill="rgba(0,0,0,0.2)"/>
            <rect x="4" y="4" width="8" height="2" fill="rgba(0,0,0,0.15)"/>
            <rect x="4" y="10" width="8" height="2" fill="rgba(0,0,0,0.15)"/>
            <rect x="1" y="0" width="1" height="16" fill="rgba(255,255,255,0.3)"/>
          </pattern>
        </defs>
        <rect width="16" height="16" fill="url(#gatePattern)"/>
      </svg>
    `
  };

  // Write overlay files
  Object.entries(overlays).forEach(([name, svg]) => {
    const filePath = path.join(generatedTexturesPath, `${name}.svg`);
    fs.writeFileSync(filePath, svg.trim());
  });

  console.log(`✅ Created ${Object.keys(overlays).length} shape overlay templates`);
}

/**
 * Generate composite textures for missing items
 */
function generateCompositeTextures(manifest) {
  createShapeOverlays();
  
  const newItems = {};
  let generatedCount = 0;

  // Look for base materials and generate composite items
  BASE_MATERIALS.forEach(baseMaterial => {
    const baseTexturePath = path.join(blockTexturesPath, `${baseMaterial}.png`);
    
    if (fs.existsSync(baseTexturePath)) {
      console.log(`🧱 Found base material: ${baseMaterial}`);
      
      // Generate combinations with each shape
      Object.entries(SHAPE_TEMPLATES).forEach(([shapeKey, shapeInfo]) => {
        const compositeKey = `${baseMaterial}_${shapeKey}`;
        
        // Check if this composite doesn't already exist
        if (!manifest.items[compositeKey]) {
          const displayName = `${formatItemName(baseMaterial)} ${shapeInfo.name}`;
          
          // Create a virtual composite texture (we'll use the base texture with an indicator)
          newItems[compositeKey] = {
            name: displayName,
            image: `/assets/minecraft/textures/block/${baseMaterial}.png`, // Use base texture for now
            category: shapeInfo.category,
            description: `${displayName} - ${shapeInfo.description}`,
            composite: true,
            baseTexture: baseMaterial,
            shape: shapeKey,
            // Add a visual indicator that this is a composite
            style: getCompositeStyle(shapeKey)
          };
          
          generatedCount++;
          console.log(`   ➕ Generated: ${compositeKey}`);
        }
      });
    }
  });

  return { newItems, generatedCount };
}

/**
 * Get CSS style for composite items to distinguish them visually
 */
function getCompositeStyle(shapeKey) {
  const styles = {
    slab: 'filter: brightness(0.9) sepia(0.1);',
    stairs: 'filter: brightness(0.95) hue-rotate(5deg);',
    button: 'filter: brightness(1.1) saturate(1.1);',
    pressure_plate: 'filter: brightness(0.85) contrast(1.1);',
    wall: 'filter: brightness(0.9) contrast(1.05);',
    fence: 'filter: brightness(0.95) saturate(0.9);',
    fence_gate: 'filter: brightness(1.05) hue-rotate(-5deg);'
  };
  
  return styles[shapeKey] || '';
}

/**
 * Format item name for display
 */
function formatItemName(name) {
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Main function to enhance manifest with composite textures
 */
export function enhanceManifestWithComposites(manifestPath) {
  console.log('🎨 Enhancing manifest with composite textures...');
  
  // Read current manifest
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  
  // Generate composite textures
  const { newItems, generatedCount } = generateCompositeTextures(manifest);
  
  // Add new items to manifest
  Object.assign(manifest.items, newItems);
  
  // Update generation info
  manifest.enhanced = new Date().toISOString();
  manifest.compositeItems = generatedCount;
  
  // Write enhanced manifest
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  
  console.log(`✅ Enhanced manifest with ${generatedCount} composite textures`);
  console.log(`📝 Total items: ${Object.keys(manifest.items).length}`);
  
  return manifest;
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const manifestPath = path.join(__dirname, '../public/assets/manifest.json');
  enhanceManifestWithComposites(manifestPath);
}
