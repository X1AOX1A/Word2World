<template>
  <div class="babyai-config">
    <div class="config-header">
      <h3>BabyAI Configuration</h3>
    </div>
    
    <div class="config-form">
      <div class="form-group">
        <label for="level-select">Game Level:</label>
        <select id="level-select" v-model="selectedLevel" class="form-control">
          <option v-for="(levelName, levelId) in gameLevels" :key="levelId" :value="levelId">
            {{ levelName }}
          </option>
        </select>
        <div class="level-description">
          {{ getLevelDescription(selectedLevel) }}
        </div>
      </div>
      
      <div class="form-group">
        <label for="random-seed">Random Seed:</label>
        <input 
          id="random-seed" 
          type="number" 
          v-model.number="randomSeed" 
          class="form-control"
          min="0"
          :max="9999"
        >
      </div>
      
      <div class="action-buttons">
        <button @click="applyConfig" class="btn primary">Apply</button>
        <button @click="resetConfig" class="btn secondary">Reset</button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'BabyAIConfig',
  props: {
    environmentId: {
      type: [String, Number],
      default: null
    },
    client: {
      type: Object,
      default: null
    }
  },
  data() {
    return {
      selectedLevel: 2,
      randomSeed: 1234,
      gameLevels: {
        1: 'Go To Red Ball Grey',
        2: 'Go To Red Ball',
        3: 'Go To Red Ball No Dists',
        4: 'Go To Object',
        5: 'Go To Local',
        6: 'Go To Object Maze',
        7: 'Go To Imp Unlock',
        8: 'Go To Sequence',
        9: 'Go To Red Blue Ball',
        10: 'Go To Door',
        11: 'Go To Object Door',
        12: 'Open',
        13: 'Open Red Door',
        14: 'Open Door Location',
        15: 'Open Red Blue Doors',
        16: 'Open Doors Order',
        17: 'Pickup',
        18: 'Unblock Pickup',
        19: 'Pickup Location',
        20: 'Pickup Distance',
        21: 'Pickup Above',
        22: 'Put Next Local',
        23: 'Put Next Carrying',
        24: 'Unlock',
        25: 'Unlock Local Distance',
        26: 'Key In Box',
        27: 'Unlock Pickup Distance',
        28: 'Blocked Unlock Pickup',
        29: 'Unlock To Unlock',
        30: 'Action Object Door',
        31: 'Find Object',
        32: 'Key Corridor',
        33: 'One Room',
        34: 'Move Two Across',
        35: 'Synthesis Simple',
        36: 'Synthesis Location',
        37: 'Synthesis Sequence',
        38: 'Mini Boss Level',
        39: 'Boss Level',
        40: 'Boss Level No Unlock'
      }
    }
  },
  methods: {
    getLevelDescription(level) {
      const descriptions = {
        1: 'Simple task to find a red ball in a gray environment.',
        2: 'Find a red ball in a colorful environment.',
        3: 'Find a red ball without distance information.',
        4: 'Find a specific object in a simple room.',
        5: 'Navigate to a local target in a small space.',
        6: 'Find an object in a maze environment.',
        7: 'Unlock and navigate to a target.',
        8: 'Complete a sequence of navigation tasks.',
        9: 'Choose between red and blue balls.',
        10: 'Find and go to a door.',
        11: 'Find a door with an object.',
        12: 'Open a door.',
        13: 'Open a specific red door.',
        14: 'Open a door at a specific location.',
        15: 'Open red and blue doors in order.',
        16: 'Open multiple doors in a specific order.',
        17: 'Pick up an object.',
        18: 'Remove obstacles to pick up an object.',
        19: 'Pick up an object at a specific location.',
        20: 'Pick up an object from a distance.',
        21: 'Pick up an object from above.',
        22: 'Put an object next to another locally.',
        23: 'Put an object next to another while carrying it.',
        24: 'Unlock a door.',
        25: 'Unlock a door at a local distance.',
        26: 'Find a key inside a box.',
        27: 'Unlock and pick up from a distance.',
        28: 'Clear obstacles, unlock, and pick up an object.',
        29: 'Unlock multiple doors in sequence.',
        30: 'Perform an action on an object behind a door.',
        31: 'Find a specific object in a complex environment.',
        32: 'Navigate a key corridor.',
        33: 'Navigate a large single room with many objects.',
        34: 'Move two objects across a space.',
        35: 'Simple synthesized tasks.',
        36: 'Synthesized location-based tasks.',
        37: 'Synthesized sequence of tasks.',
        38: 'Mini-boss level with multiple challenges.',
        39: 'Final boss level with complex challenges.',
        40: 'Boss level without unlock mechanics.'
      };
      
      return descriptions[level] || 'A BabyAI environment challenge.';
    },
    
    applyConfig() {
      if (!this.client || !this.environmentId) {
        console.warn('Cannot apply config: missing client or environment ID');
        return;
      }
      
      // Create data index by combining level and seed
      const dataIdx = (this.randomSeed % 1000) * 40 + (this.selectedLevel - 1);
      
      this.$emit('config-updating', true);
      
      // Reset environment with new configuration
      this.client.reset(this.environmentId, dataIdx)
        .then(result => {
          if (result && result.success) {
            console.log('BabyAI configuration applied successfully');
            this.$emit('config-updated', {
              level: this.selectedLevel,
              levelName: this.gameLevels[this.selectedLevel],
              seed: this.randomSeed,
              dataIdx
            });
          } else {
            console.error('Failed to apply BabyAI configuration:', result?.error);
          }
        })
        .catch(error => {
          console.error('Error applying BabyAI configuration:', error);
        })
        .finally(() => {
          this.$emit('config-updating', false);
        });
    },
    
    resetConfig() {
      this.selectedLevel = 2;
      this.randomSeed = 1234;
    }
  }
}
</script>

<style scoped>
.babyai-config {
  background: white;
  border-radius: 8px;
  padding: 1.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  max-width: 500px;
  margin: 0 auto;
}

.config-header {
  margin-bottom: 1.5rem;
}

.config-header h3 {
  font-size: 1.2rem;
  color: #333;
  border-bottom: 2px solid #2196F3;
  padding-bottom: 0.5rem;
}

.form-group {
  margin-bottom: 1.25rem;
}

.form-group label {
  display: block;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: #555;
}

.form-control {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
}

.level-description {
  margin-top: 0.5rem;
  font-size: 0.85rem;
  color: #666;
  font-style: italic;
}

.action-buttons {
  display: flex;
  gap: 1rem;
  margin-top: 1.5rem;
}

.btn {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.2s ease;
}

.btn.primary {
  background: #2196F3;
  color: white;
}

.btn.primary:hover {
  background: #1976D2;
}

.btn.secondary {
  background: #e0e0e0;
  color: #333;
}

.btn.secondary:hover {
  background: #d5d5d5;
}
</style> 