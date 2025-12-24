import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Define paths
const blockTexturesPath = path.join(__dirname, '../public/assets/minecraft/textures/block');
const itemTexturesPath = path.join(__dirname, '../public/assets/minecraft/textures/item');

// Ensure item textures directory exists
if (!fs.existsSync(itemTexturesPath)) {
  fs.mkdirSync(itemTexturesPath, { recursive: true });
  console.log('Created item textures directory');
}

// Define mapping of TextCraft items to textures
const textcraftItemMap = {
  // Items that need to be copied from block textures to item textures
  'acacia_planks': { source: 'block', filename: 'acacia_planks.png' },
  'acacia_log': { source: 'block', filename: 'acacia_log.png' },
  'oak_log': { source: 'block', filename: 'oak_log.png' },
  'oak_planks': { source: 'block', filename: 'oak_planks.png' },
  'birch_log': { source: 'block', filename: 'birch_log.png' },
  'birch_planks': { source: 'block', filename: 'birch_planks.png' },
  'spruce_log': { source: 'block', filename: 'spruce_log.png' },
  'spruce_planks': { source: 'block', filename: 'spruce_planks.png' },
  'jungle_log': { source: 'block', filename: 'jungle_log.png' },
  'jungle_planks': { source: 'block', filename: 'jungle_planks.png' },
  'dark_oak_log': { source: 'block', filename: 'dark_oak_log.png' },
  'dark_oak_planks': { source: 'block', filename: 'dark_oak_planks.png' },
  'crimson_stem': { source: 'block', filename: 'crimson_stem.png' },
  'crimson_planks': { source: 'block', filename: 'crimson_planks.png' },
  'warped_stem': { source: 'block', filename: 'warped_stem.png' },
  'warped_planks': { source: 'block', filename: 'warped_planks.png' },
  'stone': { source: 'block', filename: 'stone.png' },
  'cobblestone': { source: 'block', filename: 'cobblestone.png' },
  'iron_block': { source: 'block', filename: 'iron_block.png' },
  'gold_block': { source: 'block', filename: 'gold_block.png' },
  'diamond_block': { source: 'block', filename: 'diamond_block.png' },
  'emerald_block': { source: 'block', filename: 'emerald_block.png' },
  
  // Create minimal placeholders for items not available from block textures
  'minecraft:acacia_planks': { alias: 'acacia_planks' },
  'minecraft:acacia_log': { alias: 'acacia_log' },
};

// Copy textures from block folder to item folder
console.log('Starting texture generation...');
let copied = 0;
let failed = 0;

// Process all mappings
for (const [itemName, config] of Object.entries(textcraftItemMap)) {
  try {
    // Handle aliases - just create a symlink or copy the referenced file
    if (config.alias) {
      const targetFile = path.join(itemTexturesPath, `${itemName}.png`);
      const sourceFile = path.join(itemTexturesPath, `${config.alias}.png`);
      
      // If the target file already exists, skip
      if (fs.existsSync(targetFile)) {
        console.log(`Skipping ${itemName} - already exists`);
        continue;
      }
      
      // If the source exists, copy it
      if (fs.existsSync(sourceFile)) {
        fs.copyFileSync(sourceFile, targetFile);
        console.log(`Created ${itemName} (alias of ${config.alias})`);
        copied++;
      } else {
        console.warn(`Source for alias ${config.alias} not found, skipping ${itemName}`);
        failed++;
      }
      continue;
    }
    
    // Handle block to item copies
    if (config.source === 'block') {
      const sourceFile = path.join(blockTexturesPath, config.filename);
      const targetFile = path.join(itemTexturesPath, config.filename);
      
      // If the target file already exists, skip
      if (fs.existsSync(targetFile)) {
        console.log(`Skipping ${config.filename} - already exists`);
        continue;
      }
      
      // If the source exists, copy it
      if (fs.existsSync(sourceFile)) {
        fs.copyFileSync(sourceFile, targetFile);
        console.log(`Copied ${config.filename} from block to item textures`);
        copied++;
      } else {
        console.warn(`Block texture ${config.filename} not found, skipping`);
        failed++;
      }
    }
  } catch (error) {
    console.error(`Error processing ${itemName}:`, error.message);
    failed++;
  }
}

console.log(`Finished texture generation. Copied: ${copied}, Failed: ${failed}`); 