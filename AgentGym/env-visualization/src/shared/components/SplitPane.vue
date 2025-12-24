<template>
  <div class="splitpanes-container">
    <splitpanes class="default-theme" :horizontal="horizontal">
      <pane :size="leftSize">
        <slot name="left"></slot>
      </pane>
      <pane :size="100 - leftSize">
        <slot name="right"></slot>
      </pane>
    </splitpanes>
  </div>
</template>

<script>
import { Splitpanes, Pane } from 'splitpanes'
import 'splitpanes/dist/splitpanes.css'

export default {
  name: 'SplitPane',
  components: {
    Splitpanes,
    Pane
  },
  props: {
    defaultSplit: {
      type: Number,
      default: 50
    },
    horizontal: {
      type: Boolean,
      default: false
    }
  },
  computed: {
    leftSize() {
      return this.defaultSplit
    }
  }
}
</script>

<style scoped>
.splitpanes-container {
  height: 100%;
  width: 100%;
}

.splitpanes {
  height: 100%;
}

.splitpanes__pane {
  display: flex;
  flex-direction: column;
}
</style>
