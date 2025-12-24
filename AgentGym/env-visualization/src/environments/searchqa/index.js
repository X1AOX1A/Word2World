/**
 * SearchQA Environment Module
 * 
 * Exports all SearchQA-related components and client
 */

import SearchQAClient from './client/searchqaClient.js'
import SearchQAViewer from './components/SearchQAViewer.vue'

// Export a factory function to create new client instances
export const createSearchQAClient = () => new SearchQAClient()

export {
  SearchQAViewer
}

export default {
  name: 'SearchQA',
  id: 'searchqa',
  icon: '🔍',
  description: 'Question answering environment with search capabilities. Agents can search for information and provide answers with a maximum of 5 rounds.',
  config: {
    apiUrl: 'http://localhost:36005',
    maxRounds: 5,
    supportedActions: ['search', 'answer', 'think'],
    datasets: ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']
  },
  // Factory function to create new client instances
  client: createSearchQAClient,
  viewer: SearchQAViewer,
  components: {
    SearchQAViewer
  },
  features: {
    hasSearch: true,
    hasAnswering: true,
    hasRounds: true,
    maxRounds: 5,
    hasRewards: true,
    hasHistory: true
  }
}
