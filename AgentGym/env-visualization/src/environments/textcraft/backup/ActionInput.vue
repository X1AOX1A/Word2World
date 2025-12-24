<template>
  <div class="action-input">
    <div v-if="error" class="error-message">
      {{ error }}
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'

export default {
  name: 'ActionInput',
  props: {
    disabled: {
      type: Boolean,
      default: false
    },
    environmentId: {
      type: [String, Number],
      default: null
    },
    environmentState: {
      type: Object,
      default: null
    },
    suggestedAction: {
      type: String,
      default: ''
    }
  },
  emits: ['action-sent', 'response-received'],
  setup(props, { emit }) {
    const loading = ref(false)
    const error = ref('')

    return {
      loading,
      error
    }
  }
}
</script>

<style scoped>
.action-input {
  width: 100%;
}

.error-message {
  color: #e74c3c;
  padding: 0.5rem;
  background: #fdf2f2;
  border: 1px solid #fecaca;
  border-radius: 4px;
  font-size: 0.9rem;
}
</style>
