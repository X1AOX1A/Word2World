/**
 * SciWorld Environment Module
 * 
 * Exports all SciWorld-related components and client
 */

import SciWorldClient from './client/sciworldClient.js'
import SciWorldViewer from './components/SciWorldViewer.vue'

// Export a factory function to create new client instances
export const createSciWorldClient = () => new SciWorldClient()

export {
  SciWorldViewer
}

export default {
  name: 'SciWorld',
  id: 'sciworld',
  icon: '🧪',
  description: 'A scientific exploration environment with interactive experiments and tasks',
  // Factory function to create new client instances
  client: createSciWorldClient,
  viewer: SciWorldViewer,
  components: {
    SciWorldViewer
  }
}
