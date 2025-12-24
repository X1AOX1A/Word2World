import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import { getAvailableEnvironments } from '../environments/index.js'

// Define routes for the application
const routes = [
  // Home route - Environment Selection
  {
    path: '/',
    name: 'home',
    component: HomeView,
    meta: {
      title: 'AgentGym - Select Environment'
    }
  },
  
  // Dynamic environment route
  {
    path: '/env/:envName',
    name: 'environment',
    // Use dynamic import for code splitting
    component: () => import('../views/EnvironmentView.vue'),
    meta: {
      title: 'AgentGym - Environment'
    },
    // Validate that the environment name is valid
    beforeEnter: (to, from, next) => {
      const envName = to.params.envName
      const availableEnvironments = getAvailableEnvironments()
      
      if (availableEnvironments.includes(envName)) {
        next()
      } else {
        // Redirect to home if environment not found
        next({ name: 'home' })
      }
    }
  },
  
  // About page
  {
    path: '/about',
    name: 'about',
    // route level code-splitting
    component: () => import('../views/AboutView.vue'),
    meta: {
      title: 'AgentGym - About'
    }
  },
  
  // 404 Page
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: () => import('../views/NotFoundView.vue'),
    meta: {
      title: 'Page Not Found'
    }
  }
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes
})

// Update page title on route change
router.beforeEach((to, from, next) => {
  document.title = to.meta.title || 'AgentGym'
  next()
})

export default router
