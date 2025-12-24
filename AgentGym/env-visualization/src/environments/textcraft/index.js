/**
 * TextCraft Environment Module
 * 
 * Exports all TextCraft-related components
 */

import TextcraftViewer from './components/TextcraftViewer.vue'
import InventoryViewer from './components/InventoryViewer.vue'
import ActionInput from './components/ActionInput.vue'

export {
  TextcraftViewer,
  InventoryViewer,
  ActionInput
}

export default {
  name: 'TextCraft',
  id: 'textcraft',
  icon: '⚒️',
  description: 'A text-based crafting and building environment where agents learn to create items and structures.',
  config: {
    apiUrl: 'http://localhost:36001'
  },
  viewer: TextcraftViewer,
  components: {
    InventoryViewer,
    ActionInput
  }
}