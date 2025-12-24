/**
 * BabyAI Environment Module
 * 
 * Exports all BabyAI-related components and client
 */

import BabyAIClient from './client/babyaiClient.js'
import BabyAIViewer from './components/BabyAIViewer.vue'

// Export a factory function to create new client instances
export const createBabyAIClient = () => new BabyAIClient()

export {
  BabyAIViewer
}

export default {
  name: 'BabyAI',
  id: 'babyai',
  icon: '🧸',
  description: 'Navigate and solve tasks in grid-world environments',
  // Factory function to create new client instances
  client: createBabyAIClient,
  viewer: BabyAIViewer,
  components: {
    BabyAIViewer
  }
}
