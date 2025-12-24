/**
 * WebArena Environment Module
 * 
 * Exports all WebArena-related components and client
 */

import WebArenaClient from './client/webarenaClient.js'
import WebArenaViewer from './components/WebArenaViewer.vue'

// Export a factory function to create new client instances
export const createWebArenaClient = () => new WebArenaClient()

export {
  WebArenaViewer
}

export default {
  name: 'WebArena',
  id: 'webarena',
  icon: '🌐',
  description: 'Navigate and interact with web pages',
  // Factory function to create new client instances
  client: createWebArenaClient,
  viewer: WebArenaViewer,
  components: {
    WebArenaViewer
  }
}
