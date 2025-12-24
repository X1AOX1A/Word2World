<script>
import { useRouter } from 'vue-router'
import EnvironmentSelector from '../shared/components/EnvironmentSelector.vue'

export default {
  name: 'HomeView',
  components: {
    EnvironmentSelector
  },
  setup() {
    const router = useRouter()
    
    const onEnvironmentSelected = (env) => {
      if (env && env.id) {
        router.push(`/env/${env.id}`)
      }
    }
    
    return {
      onEnvironmentSelected
    }
  }
}
</script>

<template>
  <div class="home-view">
    <EnvironmentSelector 
      @environment-selected="onEnvironmentSelected" 
    />
  </div>
</template>

<style scoped>
.home-view {
  width: 100%;
  height: 100vh;
  overflow: hidden;
  margin: 0;
  padding: 0;
}
</style>
