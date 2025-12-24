<template>
  <!-- SciWorld Viewer - Event-driven updates (no polling) -->
  <div class="sciworld-viewer">
    <!-- Task Description -->
    <div class="task-section">
      <h3>Task Description</h3>
      <div class="task-text">{{ taskDescription || "Loading task..." }}</div>
    </div>
    
    <!-- Environment Visualization -->
    <div class="visualization-container">
      <div class="visualization-panel">
        <h3>Science World Environment</h3>
        <div class="env-visualization">
          <!-- 2D Game-style World Map -->
          <div class="world-map" ref="worldMap">
            <svg 
              :width="mapWidth" 
              :height="mapHeight" 
              viewBox="0 0 720 450"
              class="world-svg"
            >
              <!-- Background -->
              <rect width="720" height="450" fill="#4a6741" class="world-background"/>
              
              <!-- Grid lines for better organization -->
              <defs>
                <pattern id="grid" width="140" height="140" patternUnits="userSpaceOnUse">
                  <path d="M 140 0 L 0 0 0 140" fill="none" stroke="#5a7751" stroke-width="1" opacity="0.3"/>
                </pattern>
              </defs>
              <rect width="720" height="450" fill="url(#grid)" opacity="0.5"/>
              
              <!-- Walkable paths between rooms -->
              <g class="paths-group">
                <!-- Horizontal paths -->
                <rect x="140" y="75" width="20" height="10" fill="#8B7355" opacity="0.8"/>
                <rect x="280" y="75" width="20" height="10" fill="#8B7355" opacity="0.8"/>
                <rect x="420" y="75" width="20" height="10" fill="#8B7355" opacity="0.8"/>
                <rect x="560" y="75" width="20" height="10" fill="#8B7355" opacity="0.8"/>
                
                <rect x="140" y="355" width="20" height="10" fill="#8B7355" opacity="0.8"/>
                <rect x="280" y="355" width="20" height="10" fill="#8B7355" opacity="0.8"/>
                <rect x="420" y="355" width="20" height="10" fill="#8B7355" opacity="0.8"/>
                <rect x="560" y="355" width="20" height="10" fill="#8B7355" opacity="0.8"/>
                
                <!-- Vertical paths -->
                <rect x="75" y="140" width="10" height="20" fill="#8B7355" opacity="0.8"/>
                <rect x="215" y="140" width="10" height="20" fill="#8B7355" opacity="0.8"/>
                <rect x="355" y="140" width="10" height="20" fill="#8B7355" opacity="0.8"/>
                <rect x="495" y="140" width="10" height="20" fill="#8B7355" opacity="0.8"/>
                <rect x="635" y="140" width="10" height="20" fill="#8B7355" opacity="0.8"/>
                
                <rect x="75" y="280" width="10" height="20" fill="#8B7355" opacity="0.8"/>
                <rect x="215" y="280" width="10" height="20" fill="#8B7355" opacity="0.8"/>
                <rect x="355" y="280" width="10" height="20" fill="#8B7355" opacity="0.8"/>
                <rect x="495" y="280" width="10" height="20" fill="#8B7355" opacity="0.8"/>
                <rect x="635" y="280" width="10" height="20" fill="#8B7355" opacity="0.8"/>
              </g>
              
              <!-- Rooms -->
              <g v-for="(room, roomId) in worldRooms" :key="roomId" class="room-group">
                <!-- Room rectangle -->
                <rect 
                  :x="room.x" 
                  :y="room.y" 
                  :width="room.width" 
                  :height="room.height"
                  :fill="room.color"
                  :stroke="room.id === currentRoom ? '#ffff00' : '#8B4513'"
                  :stroke-width="room.id === currentRoom ? 4 : 2"
                  class="room-rect"
                  @click="onRoomClick(room)"
                />
                
                <!-- Room label -->
                <text 
                  :x="room.x + room.width/2" 
                  :y="room.y + 20"
                  text-anchor="middle"
                  class="room-label"
                  fill="#000"
                  font-weight="bold"
                  font-size="14"
                >
                  {{ room.name.toUpperCase() }}
                </text>
                
                <!-- Objects in room -->
                <g v-for="(obj, objIndex) in room.objects" :key="objIndex" class="object-group">
                  <circle 
                    :cx="obj.x" 
                    :cy="obj.y" 
                    :r="8"
                    :fill="obj.color"
                    stroke="#333"
                    stroke-width="1"
                    class="object-circle"
                  />
                  <text 
                    :x="obj.x" 
                    :y="obj.y + 3"
                    text-anchor="middle"
                    font-size="10"
                    class="object-icon-text"
                  >
                    {{ obj.icon }}
                  </text>
                </g>
              </g>
              
              <!-- Player/Agent -->
              <g v-if="playerPosition" class="player-group">
                <circle 
                  :cx="playerPosition.x" 
                  :cy="playerPosition.y" 
                  r="12"
                  fill="#ff6b6b"
                  stroke="#fff"
                  stroke-width="2"
                  class="player-circle"
                />
                <text 
                  :x="playerPosition.x" 
                  :y="playerPosition.y + 4"
                  text-anchor="middle"
                  font-size="12"
                  fill="#fff"
                  font-weight="bold"
                  class="player-icon"
                >
                  🤖
                </text>
              </g>
              
              <!-- Connections between rooms -->
              <g class="connections-group">
                <line 
                  v-for="(connection, index) in roomConnections" 
                  :key="index"
                  :x1="connection.x1" 
                  :y1="connection.y1"
                  :x2="connection.x2" 
                  :y2="connection.y2"
                  stroke="#8B4513"
                  stroke-width="3"
                  class="room-connection"
                />
              </g>
            </svg>
          </div>
          
          <!-- Fallback text view -->
          <div v-if="showTextFallback" class="text-visualization">
            <div class="observation-display">
              <h4>Current Observation</h4>
              <pre class="observation-text">{{ observation }}</pre>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Side Panel -->
      <div class="info-panel">
        <!-- Current Status -->
        <div class="status-section">
          <h4>Status</h4>
          <div class="status-info">
            <div>Score: {{ score || 0 }}</div>
            <div>Moves: {{ moves || 0 }}</div>
            <div v-if="isDone" class="completion-badge">✅ Completed</div>
          </div>
        </div>
        
        <!-- Inventory -->
        <div class="inventory-section">
          <h4>Inventory</h4>
          <div class="inventory-content">
            <div v-if="inventory && inventory.length > 0" class="inventory-items">
              <div v-for="(item, index) in inventoryItems" :key="index" class="inventory-item">
                <span class="item-icon">📦</span>
                <span class="item-name">{{ item }}</span>
              </div>
            </div>
            <div v-else class="empty-inventory">Empty</div>
          </div>
        </div>
        
        <!-- Possible Actions -->
        <div class="actions-section">
          <h4>Available Actions</h4>
          <div class="actions-content">
            <div v-if="possibleActions && possibleActions.length > 0" class="action-list">
              <div v-for="(action, index) in possibleActions.slice(0, 5)" :key="index" class="action-item">
                {{ action }}
              </div>
              <div v-if="possibleActions.length > 5" class="more-actions">
                ... and {{ possibleActions.length - 5 }} more
              </div>
            </div>
            <div v-else class="no-actions">No actions available</div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Loading Overlay -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner"></div>
      <div>Loading...</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'SciWorldViewer',
  emits: ['state-updated', 'room-selected', 'agent-moved'],
  props: {
    environmentId: {
      type: [String, Number],
      default: null
    },
    client: {
      type: Object,
      default: null
    },
    envState: {
      type: Object,
      default: () => null
    }
  },
  data() {
    return {
      loading: false,
      currentState: null,
      objectTree: null,
      inventory: null,
      taskDescription: null,
      possibleActions: [],
      observation: null,
      score: 0,
      moves: 0,
      isDone: false,
      // 2D World Map Data
      worldRooms: {},
      playerPosition: null,
      currentRoom: null,
      roomConnections: [],
      mapWidth: 720,
      mapHeight: 450,
      showTextFallback: false
    }
  },
  computed: {
    inventoryItems() {
      if (!this.inventory) return [];
      
      if (typeof this.inventory === 'string') {
        // Parse inventory string
        return this.inventory.split('\n')
          .filter(line => line.trim() && !line.includes('Inventory'))
          .map(line => line.trim())
          .slice(0, 5); // Limit to 5 items
      }
      
      return Array.isArray(this.inventory) ? this.inventory.slice(0, 5) : [];
    }
  },
  watch: {
    environmentId: {
      immediate: true,
      handler(newId) {
        if (newId && this.client) {
          this.loadEnvironmentData();
        }
      }
    },
    
    envState: {
      immediate: true,
      deep: true,
      handler(newState, oldState) {
        if (newState) {
          this.updateFromState(newState);
          // Only load additional data if this is a new state update (after action or reset)
          if (!oldState || newState !== oldState) {
            this.loadEnvironmentData();
          }
        }
      }
    }
  },
  mounted() {
    // Load initial data when component is mounted
    if (this.environmentId && this.client) {
      this.loadEnvironmentData();
    } else {
      // Create default layout even without data
      this.buildWorldMap();
    }
  },
  
  methods: {
    async loadEnvironmentData() {
      if (!this.environmentId || !this.client) return;
      
      this.loading = true;
      
      try {
        // Only load data that's not already available in envState
        const promises = [];
        
        // Always get current state for the most up-to-date info
        promises.push(this.client.getState(this.environmentId));
        
        // Load object tree if not available
        if (!this.objectTree) {
          promises.push(this.client.getObjectTree(this.environmentId));
        }
        
        // Load task description if not available  
        if (!this.taskDescription) {
          promises.push(this.client.getTaskDescription(this.environmentId));
        }
        
        const results = await Promise.all(promises);
        
        // Update state data (always first result)
        if (results[0] && results[0].success) {
          this.updateFromState(results[0].data, true); // Pass true to indicate this is from load
          // updateFromState will emit 'state-updated' if needed
        }
        
        // Update additional data based on what was requested
        let resultIndex = 1;
        
        if (!this.objectTree && results[resultIndex]) {
          if (results[resultIndex].success) {
            this.objectTree = results[resultIndex].data.object_tree;
          }
          resultIndex++;
        }
        
        if (!this.taskDescription && results[resultIndex]) {
          if (results[resultIndex].success) {
            this.taskDescription = results[resultIndex].data.task_description;
          }
          resultIndex++;
        }
        
        // Rebuild world map with new data
        this.buildWorldMap();
        
      } catch (error) {
        console.error('Error loading environment data:', error);
        // Even on error, try to build default map
        this.buildWorldMap();
      } finally {
        this.loading = false;
      }
    },

    updateFromState(state, from_load = false) {
      if (!state) return;
      
      const previousState = {
        observation: this.observation,
        score: this.score,
        moves: this.moves,
        isDone: this.isDone
      };
      
      this.observation = state.observation || state.current_observation;
      this.score = state.score || state.current_score || 0;
      this.moves = state.moves || state.current_moves || 0;
      this.isDone = state.done || state.finished || false;
      
      // Update other properties if available
      if (state.object_tree) this.objectTree = state.object_tree;
      if (state.inventory) this.inventory = state.inventory;
      if (state.task_description) this.taskDescription = state.task_description;
      if (state.possible_actions) this.possibleActions = state.possible_actions;
      
      // Rebuild the world map with new data
      this.buildWorldMap();
      
      // Force agent position update after state change
      this.$nextTick(() => {
        this.updatePlayerPosition();
      });
      
      // Emit state update if something meaningful changed
      // const hasChanged = 
      //   previousState.observation !== this.observation ||
      //   previousState.score !== this.score ||
      //   previousState.moves !== this.moves ||
      //   previousState.isDone !== this.isDone;
        
      // if (hasChanged) {
        // console.log('SciWorldViewer: emitting state update');
        // this.$emit('state-updated', state);
      // }
      if (!from_load) {
        console.log('SciWorldViewer: emitting state update');
        this.$emit('state-updated', state);
      }
    },
    
    // Parse ScienceWorld data and build 2D world map
    buildWorldMap() {
      try {
        // Parse current location from observation
        this.parseCurrentLocation();
        
        // Parse object tree for rooms and objects
        this.parseObjectTree();
        
        // If no structured data available, show text fallback
        if (Object.keys(this.worldRooms).length === 0) {
          this.showTextFallback = true;
          this.createDefaultRoomLayout();
        } else {
          this.showTextFallback = false;
        }
        
      } catch (error) {
        console.error('Error building world map:', error);
        this.showTextFallback = true;
      }
    },
    
    parseCurrentLocation() {
      if (!this.observation) return;
      
      console.log('🗺️ Parsing location from observation:', this.observation.substring(0, 200) + '...');
      
      // ScienceWorld specific patterns - prioritize the most reliable ones
      const roomPatterns = [
        // PRIMARY: ScienceWorld standard format - highest priority
        /This room is called the (.+?)\./i,
        /This room is called (.+?)\./i,
        
        // SECONDARY: Other common patterns
        /You are (?:in|at|inside|within) (?:the )?(.+?)(?:\.|,|\n|$)/i,
        /Current location:?\s*(?:the )?(.+?)(?:\.|,|\n|$)/i,
        /Location:?\s*(?:the )?(.+?)(?:\.|,|\n|$)/i,
        
        // MOVEMENT: Action-based patterns
        /(?:moved to|entered|went to|arrived at)\s*(?:the )?(.+?)(?:\.|,|\n|$)/i,
        /Now in (?:the )?(.+?)(?:\.|,|\n|$)/i,
        
        // FALLBACK: Basic patterns
        /Looking around,? you see:?\s*(?:the )?(.+?)(?:\.|,|\n|$)/i,
        /^(.+?)\s*(?:room|area|studio|hall|way)(?:\.|,|\n|$)/im
      ];
      
      let detectedRoom = null;
      
      for (const pattern of roomPatterns) {
        const match = this.observation.match(pattern);
        if (match && match[1]) {
          detectedRoom = match[1].toLowerCase().trim();
          console.log(`Detected room: "${detectedRoom}" using pattern:`, pattern);
          
          // Validate the room name - filter out common words that aren't rooms
          const nonRoomWords = ['you', 'are', 'the', 'see', 'can', 'here', 'there', 'this', 'that', 'around', 'look'];
          if (!nonRoomWords.includes(detectedRoom) && detectedRoom.length > 2) {
            break;
          } else {
            detectedRoom = null;
          }
        }
      }
      
      // Also check inventory and task description for context
      if (!detectedRoom && this.inventory) {
        const invMatch = this.inventory.match(/(?:in|at) (?:the )?(\w+)/i);
        if (invMatch) {
          detectedRoom = invMatch[1].toLowerCase();
        }
      }
      
      // Map ScienceWorld room names to our layout room names
      const roomMappings = {
        // Direct mappings
        'laboratory': 'laboratory',
        'kitchen': 'kitchen', 
        'bathroom': 'bathroom',
        'bedroom': 'bedroom',
        'workshop': 'workshop',
        'foundry': 'foundry',
        'greenhouse': 'greenhouse',
        'storage': 'storage',
        'outside': 'outside',
        'living room': 'living',
        'living': 'living',
        
        // ScienceWorld specific names
        'art studio': 'art',
        'art room': 'art', 
        'art': 'art',
        'hallway': 'hallway',
        'hall': 'hallway',
        'corridor': 'hallway',
        
        // Common variations
        'lab': 'laboratory',
        'labs': 'laboratory',
        'science lab': 'laboratory',
        'science': 'laboratory',
        'cook': 'kitchen',
        'bath': 'bathroom',
        'toilet': 'bathroom',
        'washroom': 'bathroom',
        'bed': 'bedroom',
        'sleep': 'bedroom',
        'work': 'workshop',
        'craft': 'workshop',
        'workroom': 'workshop',
        'forge': 'foundry',
        'foundary': 'foundry',
        'green': 'greenhouse',
        'garden': 'greenhouse',
        'plant': 'greenhouse',
        'store': 'storage',
        'storeroom': 'storage',
        'outdoor': 'outside',
        'lounge': 'living',
        'sitting room': 'living',
        'paint': 'art',
        'painting': 'art',
        'draw': 'art',
        'drawing': 'art'
      };
      
      if (detectedRoom && roomMappings[detectedRoom]) {
        detectedRoom = roomMappings[detectedRoom];
      }
      
      // Update current room if we found a valid one
      if (detectedRoom && this.worldRooms[detectedRoom]) {
        const previousRoom = this.currentRoom;
        this.currentRoom = detectedRoom;
        console.log(`🤖 Agent moved from "${previousRoom}" to "${this.currentRoom}"`);
        
        // Emit movement event
        this.$emit('agent-moved', {
          from: previousRoom,
          to: this.currentRoom,
          timestamp: Date.now()
        });
      } else if (detectedRoom) {
        // Room detected but not in our layout - add it dynamically
        console.log(`🆕 New room detected: "${detectedRoom}", adding to layout`);
        this.addDynamicRoom(detectedRoom);
        this.currentRoom = detectedRoom;
        
        // Also emit movement event for new rooms
        this.$emit('agent-moved', {
          from: this.currentRoom,
          to: detectedRoom,
          timestamp: Date.now()
        });
      } else if (!this.currentRoom) {
        // Fallback to first available room
        const roomKeys = Object.keys(this.worldRooms);
        this.currentRoom = roomKeys.length > 0 ? roomKeys[0] : 'hallway';
        console.log(`🏠 No room detected, defaulting to: "${this.currentRoom}"`);
      }
      
      // Parse objects in current room from observation
      this.parseRoomObjects();
    },
    
    parseRoomObjects() {
      if (!this.observation || !this.currentRoom || !this.worldRooms[this.currentRoom]) return;
      
      try {
        const room = this.worldRooms[this.currentRoom];
        const newObjects = [];
        
        // Extract objects from "In it, you see:" section
        const objectsSection = this.observation.match(/In it, you see:(.*?)(?:You also see:|$)/s);
        if (objectsSection) {
          const objectsText = objectsSection[1];
          const lines = objectsText.split('\n').map(line => line.trim()).filter(line => line);
          
          let objectIndex = 0;
          for (const line of lines) {
            // Skip "the agent" and "air"
            if (line.includes('the agent') || line.includes('substance called air')) continue;
            
            // Parse different object types
            let objectName = '';
            let icon = '📍';
            let color = '#CD853F';
            
            if (line.includes('cupboard') || line.includes('cabinet')) {
              objectName = 'cupboard';
              icon = '🗄️';
              color = '#8B4513';
            } else if (line.includes('table')) {
              objectName = 'table';
              icon = '🪑';
              color = '#8B4513';
            } else if (line.includes('jug')) {
              objectName = 'jug';
              icon = '🫗';
              color = '#4682B4';
            } else if (line.includes('wood cup') && line.includes('paint')) {
              if (line.includes('yellow')) {
                objectName = 'yellow_paint';
                icon = '🎨';
                color = '#FFD700';
              } else if (line.includes('blue')) {
                objectName = 'blue_paint';
                icon = '🎨';
                color = '#0000FF';
              } else if (line.includes('red')) {
                objectName = 'red_paint';
                icon = '🎨';
                color = '#FF0000';
              } else {
                objectName = 'paint_cup';
                icon = '🎨';
                color = '#FF1493';
              }
            } else if (line.includes('picture')) {
              objectName = 'picture';
              icon = '🖼️';
              color = '#8B4513';
            } else if (line.includes('door')) {
              continue; // Skip doors, they're connections
            } else {
              // Generic object
              const match = line.match(/(?:a|an|the)\s+([^.,(]+)/);
              if (match) {
                objectName = match[1].trim();
                icon = this.getObjectIcon(objectName);
                color = this.getObjectColor(objectName);
              }
            }
            
            if (objectName) {
              // Position objects within the room
              const col = objectIndex % 3;
              const row = Math.floor(objectIndex / 3);
              newObjects.push({
                name: objectName,
                x: room.x + 30 + col * 30,
                y: room.y + 40 + row * 25,
                icon: icon,
                color: color
              });
              objectIndex++;
            }
          }
        }
        
        // Update room objects if we found any
        if (newObjects.length > 0) {
          room.objects = newObjects;
          console.log(`Updated ${this.currentRoom} with ${newObjects.length} objects:`, newObjects.map(obj => obj.name));
        }
        
      } catch (error) {
        console.error('Error parsing room objects:', error);
      }
    },
    
    parseObjectTree() {
      if (!this.objectTree) return;
      
      try {
        let treeData = this.objectTree;
        
        // If it's a string, try to parse as JSON or text
        if (typeof treeData === 'string') {
          try {
            treeData = JSON.parse(treeData);
          } catch {
            // Not JSON, parse as text
            this.parseObjectTreeText(treeData);
            return;
          }
        }
        
        // If it's an object/array, parse structured data
        this.parseObjectTreeStructured(treeData);
        
      } catch (error) {
        console.error('Error parsing object tree:', error);
      }
    },
    
    parseObjectTreeText(treeText) {
      const lines = treeText.split('\n').filter(line => line.trim());
      const rooms = new Set();
      const objects = [];
      
      for (const line of lines) {
        // Extract room names
        if (line.includes('room') || line.includes('location')) {
          const roomMatch = line.match(/(\w+)(?:\s+room|\s+location)/i);
          if (roomMatch) {
            rooms.add(roomMatch[1].toLowerCase());
          }
        }
        
        // Extract objects
        const objectMatch = line.match(/(\w+)\s*\(([^)]*)\)/);
        if (objectMatch) {
          objects.push({
            name: objectMatch[1],
            properties: objectMatch[2]
          });
        }
      }
      
      this.createRoomsFromData(Array.from(rooms), objects);
    },
    
    parseObjectTreeStructured(treeData) {
      // Handle structured object tree data
      const rooms = new Set();
      const objects = [];
      
      const extractFromNode = (node) => {
        if (typeof node === 'object') {
          if (node.type && node.type.includes('room')) {
            rooms.add(node.name || 'room');
          }
          
          if (node.name) {
            objects.push({
              name: node.name,
              type: node.type || 'object',
              properties: node.properties || {}
            });
          }
          
          // Recursively process children
          if (node.children) {
            node.children.forEach(extractFromNode);
          }
        }
      };
      
      if (Array.isArray(treeData)) {
        treeData.forEach(extractFromNode);
      } else {
        extractFromNode(treeData);
      }
      
      this.createRoomsFromData(Array.from(rooms), objects);
    },
    
    createRoomsFromData(roomNames, objects) {
      const newRooms = {};
      const roomColors = [
        '#DEB887', '#F5DEB3', '#D2B48C', '#BC8F8F', 
        '#F4A460', '#CD853F', '#DAA520', '#B8860B'
      ];
      
      // Create default room layout
      const roomLayouts = this.getDefaultRoomLayouts();
      
      roomNames.forEach((roomName, index) => {
        const layout = roomLayouts[index] || roomLayouts[0];
        newRooms[roomName] = {
          id: roomName,
          name: roomName,
          x: layout.x,
          y: layout.y,
          width: layout.width,
          height: layout.height,
          color: roomColors[index % roomColors.length],
          objects: []
        };
      });
      
      // If no rooms detected, create default layout
      if (Object.keys(newRooms).length === 0) {
        this.createDefaultRoomLayout();
        return;
      }
      
      // Distribute objects among rooms
      objects.forEach((obj, index) => {
        const roomKeys = Object.keys(newRooms);
        const roomKey = roomKeys[index % roomKeys.length];
        const room = newRooms[roomKey];
        
        const objX = room.x + 30 + (index % 3) * 40;
        const objY = room.y + 40 + Math.floor(index / 3) * 30;
        
        room.objects.push({
          name: obj.name,
          x: objX,
          y: objY,
          icon: this.getObjectIcon(obj.name),
          color: this.getObjectColor(obj.name)
        });
      });
      
      this.worldRooms = newRooms;
      this.updatePlayerPosition();
    },
    
    createDefaultRoomLayout() {
      // Create a layout with hallway as central hub, exactly like ScienceWorld
      this.worldRooms = {
        'outside': {
          id: 'outside',
          name: 'outside',
          x: 20, y: 20, width: 120, height: 120,
          color: '#4a6741',
          objects: [
            {name: 'tree', x: 60, y: 60, icon: '🌳', color: '#228B22'},
            {name: 'pond', x: 100, y: 100, icon: '💧', color: '#4682B4'}
          ]
        },
        'foundry': {
          id: 'foundry',
          name: 'foundry',
          x: 20, y: 160, width: 120, height: 120,
          color: '#CD853F',
          objects: [
            {name: 'furnace', x: 60, y: 200, icon: '🔥', color: '#FF4500'},
            {name: 'anvil', x: 100, y: 220, icon: '⚒️', color: '#696969'}
          ]
        },
        'hallway': {
          id: 'hallway',
          name: 'hallway',
          x: 160, y: 160, width: 120, height: 120,
          color: '#F5F5DC',
          objects: [
            {name: 'picture', x: 200, y: 200, icon: '🖼️', color: '#8B4513'},
            {name: 'doors', x: 240, y: 220, icon: '🚪', color: '#8B4513'}
          ]
        },
        'greenhouse': {
          id: 'greenhouse',
          name: 'greenhouse',
          x: 300, y: 160, width: 120, height: 120,
          color: '#90EE90',
          objects: [
            {name: 'plant', x: 340, y: 200, icon: '🌱', color: '#32CD32'},
            {name: 'watering_can', x: 380, y: 220, icon: '🪣', color: '#4682B4'}
          ]
        },
        'workshop': {
          id: 'workshop',
          name: 'workshop',
          x: 440, y: 20, width: 120, height: 120,
          color: '#DEB887',
          objects: [
            {name: 'workbench', x: 480, y: 60, icon: '🪑', color: '#8B4513'},
            {name: 'tools', x: 520, y: 80, icon: '🔧', color: '#696969'}
          ]
        },
        'art': {
          id: 'art',
          name: 'art',
          x: 300, y: 20, width: 120, height: 120,
          color: '#FFB6C1',
          objects: [
            {name: 'easel', x: 340, y: 60, icon: '🎨', color: '#FF1493'},
            {name: 'wood_cup', x: 380, y: 80, icon: '🎨', color: '#FF1493'},
            {name: 'paint', x: 360, y: 100, icon: '🌈', color: '#FF6347'}
          ]
        },
        'living': {
          id: 'living',
          name: 'living',
          x: 580, y: 20, width: 120, height: 120,
          color: '#F0E68C',
          objects: [
            {name: 'sofa', x: 620, y: 60, icon: '🛋️', color: '#8B4513'},
            {name: 'tv', x: 660, y: 80, icon: '📺', color: '#000000'}
          ]
        },
        'kitchen': {
          id: 'kitchen',
          name: 'kitchen',
          x: 440, y: 300, width: 120, height: 120,
          color: '#F5DEB3',
          objects: [
            {name: 'stove', x: 480, y: 340, icon: '🔥', color: '#FF4500'},
            {name: 'refrigerator', x: 520, y: 360, icon: '🧊', color: '#4682B4'}
          ]
        },
        'bathroom': {
          id: 'bathroom',
          name: 'bathroom',
          x: 580, y: 160, width: 120, height: 120,
          color: '#E0FFFF',
          objects: [
            {name: 'sink', x: 620, y: 200, icon: '🚿', color: '#4682B4'},
            {name: 'toilet', x: 660, y: 220, icon: '🚽', color: '#FFFFFF'}
          ]
        },
        'bedroom': {
          id: 'bedroom',
          name: 'bedroom',
          x: 580, y: 300, width: 120, height: 120,
          color: '#DDA0DD',
          objects: [
            {name: 'bed', x: 620, y: 340, icon: '🛏️', color: '#8B4513'},
            {name: 'dresser', x: 660, y: 360, icon: '🗄️', color: '#8B4513'}
          ]
        },
        'laboratory': {
          id: 'laboratory',
          name: 'laboratory',
          x: 300, y: 300, width: 120, height: 120,
          color: '#D2B48C',
          objects: [
            {name: 'beaker', x: 340, y: 340, icon: '🧪', color: '#32CD32'},
            {name: 'microscope', x: 380, y: 360, icon: '🔬', color: '#4169E1'}
          ]
        },
        'storage': {
          id: 'storage',
          name: 'storage',
          x: 20, y: 300, width: 120, height: 120,
          color: '#BC8F8F',
          objects: [
            {name: 'shelf', x: 60, y: 340, icon: '📚', color: '#8B4513'},
            {name: 'box', x: 100, y: 360, icon: '📦', color: '#CD853F'}
          ]
        }
      };
      
      // Add connections between adjacent rooms
      this.createRoomConnections();
      this.updatePlayerPosition();
    },
    
    getDefaultRoomLayouts() {
      return [
        {x: 50, y: 50, width: 120, height: 100},
        {x: 200, y: 50, width: 120, height: 100},
        {x: 350, y: 50, width: 120, height: 100},
        {x: 500, y: 50, width: 120, height: 100},
        {x: 50, y: 200, width: 120, height: 100},
        {x: 200, y: 200, width: 120, height: 100},
        {x: 350, y: 200, width: 120, height: 100},
        {x: 500, y: 200, width: 120, height: 100}
      ];
    },
    
    addDynamicRoom(roomName) {
      // Add a new room dynamically if not in the predefined layout
      const existingRooms = Object.keys(this.worldRooms).length;
      const row = Math.floor(existingRooms / 3);
      const col = existingRooms % 3;
      
      this.worldRooms[roomName] = {
        id: roomName,
        name: roomName,
        x: 20 + col * 140,
        y: 20 + row * 140,
        width: 120,
        height: 120,
        color: '#D3D3D3', // Light gray for unknown rooms
        objects: []
      };
      
      console.log(`Added dynamic room: ${roomName} at position (${this.worldRooms[roomName].x}, ${this.worldRooms[roomName].y})`);
    },
    
    updatePlayerPosition() {
      if (this.currentRoom && this.worldRooms[this.currentRoom]) {
        const room = this.worldRooms[this.currentRoom];
        
        // Position player slightly offset from center for better visibility
        const offsetX = (Math.random() - 0.5) * 20; // Random offset ±10
        const offsetY = (Math.random() - 0.5) * 20;
        
        this.playerPosition = {
          x: room.x + room.width / 2 + offsetX,
          y: room.y + room.height / 2 + offsetY
        };
        
        console.log(`Agent positioned in ${this.currentRoom} at (${this.playerPosition.x}, ${this.playerPosition.y})`);
      } else {
        // Default position in center of map
        this.playerPosition = {x: 360, y: 225};
        console.log('Agent positioned at default location');
      }
    },
    
    getObjectIcon(objectName) {
      const iconMap = {
        'table': '🪑', 'chair': '🪑', 'desk': '🪑',
        'box': '📦', 'container': '📦', 'chest': '📦',
        'beaker': '🧪', 'flask': '🧪', 'vial': '🧪',
        'thermometer': '🌡️', 'gauge': '🌡️',
        'flame': '🔥', 'fire': '🔥', 'burner': '🔥', 'stove': '🔥',
        'water': '💧', 'sink': '💧', 'tap': '💧',
        'chemical': '⚗️', 'substance': '⚗️', 'compound': '⚗️',
        'instrument': '🔬', 'microscope': '🔬', 'scope': '🔬',
        'door': '🚪', 'entrance': '🚪', 'exit': '🚪',
        'key': '🗝️', 'card': '🗝️',
        'book': '📚', 'manual': '📚', 'guide': '📚',
        'note': '📝', 'paper': '📝', 'document': '📝',
        'tool': '🔧', 'wrench': '🔧', 'screwdriver': '🔧'
      };
      
      const name = objectName.toLowerCase();
      for (const [key, icon] of Object.entries(iconMap)) {
        if (name.includes(key)) {
          return icon;
        }
      }
      
      return '📍';
    },
    
    getObjectColor(objectName) {
      const colorMap = {
        'flame': '#FF4500', 'fire': '#FF4500', 'burner': '#FF4500',
        'water': '#4682B4', 'sink': '#4682B4',
        'chemical': '#32CD32', 'beaker': '#32CD32',
        'tool': '#696969', 'metal': '#696969',
        'wood': '#8B4513', 'table': '#8B4513',
        'book': '#4169E1', 'paper': '#4169E1'
      };
      
      const name = objectName.toLowerCase();
      for (const [key, color] of Object.entries(colorMap)) {
        if (name.includes(key)) {
          return color;
        }
      }
      
      return '#CD853F';
    },
    
    createRoomConnections() {
      // Create connections with hallway as central hub
      this.roomConnections = [
        // Hallway to surrounding rooms (star pattern)
        {x1: 220, y1: 160, x2: 220, y2: 140}, // hallway to outside (up)
        {x1: 220, y1: 280, x2: 220, y2: 300}, // hallway to laboratory (down)
        {x1: 160, y1: 220, x2: 140, y2: 220}, // hallway to foundry (left)
        {x1: 280, y1: 220, x2: 300, y2: 220}, // hallway to greenhouse (right)
        
        // Hallway to diagonal rooms
        {x1: 200, y1: 160, x2: 340, y2: 140}, // hallway to art (up-right)
        {x1: 240, y1: 160, x2: 480, y2: 140}, // hallway to workshop (up-far-right)
        {x1: 260, y1: 200, x2: 580, y2: 200}, // hallway to bathroom (right)
        {x1: 200, y1: 280, x2: 480, y2: 340}, // hallway to kitchen (down-right)
        {x1: 240, y1: 280, x2: 620, y2: 340}, // hallway to bedroom (down-far-right)
        {x1: 220, y1: 160, x2: 620, y2: 80},  // hallway to living (up-far-right)
        
        // Additional direct connections for convenience
        {x1: 140, y1: 80, x2: 340, y2: 80},   // outside to art
        {x1: 420, y1: 80, x2: 440, y2: 80},   // art to workshop  
        {x1: 560, y1: 80, x2: 580, y2: 80},   // workshop to living
        {x1: 140, y1: 360, x2: 300, y2: 360}, // storage to laboratory
        {x1: 420, y1: 360, x2: 440, y2: 360}, // laboratory to kitchen
        {x1: 560, y1: 360, x2: 580, y2: 360}, // kitchen to bedroom
      ];
    },
    
    onRoomClick(room) {
      console.log('🏠 Room clicked:', room.name);
      // Could emit event for room interaction
      this.$emit('room-selected', room);
    },
    
        // Test function for debugging location parsing
    testLocationParsing(testObservation) {
      console.log('🧪 Testing location parsing with:', testObservation);
      const oldObservation = this.observation;
      this.observation = testObservation;
      this.parseCurrentLocation();
      this.observation = oldObservation;
      console.log('🎯 Result: current room is now', this.currentRoom);
    },
     
    // Method to manually refresh data when needed (after reset or action)
    refreshEnvironmentData() {
      if (this.environmentId && this.client && !this.loading) {
        this.loadEnvironmentData();
      }
    },
    
    // Public method for parent components to trigger updates after actions
    onActionCompleted(actionResult) {
      if (actionResult) {
        this.updateFromState(actionResult);
      }
      this.refreshEnvironmentData();
    },
    
    // Public method for parent components to trigger updates after reset
    onEnvironmentReset(resetResult) {
      if (resetResult) {
        this.updateFromState(resetResult);
      }
      this.refreshEnvironmentData();
    }
  }
}
</script>

<style scoped>
.sciworld-viewer {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #f8f9fa;
  position: relative;
}

.task-section {
  background: white;
  padding: 0.5rem 0.8rem;
  border-bottom: 1px solid #dee2e6;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  flex-shrink: 0;
}

.task-section h3 {
  margin: 0 0 0.2rem 0;
  color: #495057;
  font-size: 0.9rem;
}

.task-text {
  color: #6c757d;
  line-height: 1.3;
  font-size: 0.8rem;
}

.visualization-container {
  flex: 1;
  display: flex;
  gap: 0.6rem;
  padding: 0.6rem;
  min-height: 0;
}

.visualization-panel {
  flex: 8;
  background: white;
  border-radius: 6px;
  padding: 0.5rem;
  box-shadow: 0 1px 4px rgba(0,0,0,0.1);
  overflow: hidden;
}

.visualization-panel h3 {
  margin: 0 0 0.3rem 0;
  color: #495057;
  font-size: 0.85rem;
}

.env-visualization {
  height: 100%;
  overflow: hidden;
}

/* 2D World Map Styles */
.world-map {
  width: 100%;
  height: 100%;
  min-height: 750px;
  border: 1px solid #8B4513;
  border-radius: 6px;
  overflow: hidden;
  background: #4a6741;
}

.world-svg {
  width: 100%;
  height: 100%;
  cursor: crosshair;
}

.world-background {
  filter: url(#grass-pattern);
}

.room-rect {
  transition: all 0.3s ease;
  cursor: pointer;
  filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.3));
  stroke-linejoin: round;
  stroke-linecap: round;
}

.room-rect:hover {
  filter: drop-shadow(3px 3px 6px rgba(0,0,0,0.5));
  transform: scale(1.02);
}

.room-label {
  pointer-events: none;
  text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
  font-family: 'Arial', sans-serif;
  font-weight: bold;
}

.object-circle {
  transition: all 0.2s ease;
  cursor: pointer;
  filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.3));
}

.object-circle:hover {
  transform: scale(1.2);
  filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.5));
}

.object-icon-text {
  pointer-events: none;
  font-family: 'Arial', sans-serif;
}

.player-circle {
  animation: playerBlink 1.5s ease-in-out infinite;
  filter: drop-shadow(2px 2px 6px rgba(0,0,0,0.4));
}

@keyframes playerBlink {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.3;
  }
}

.player-icon {
  pointer-events: none;
  text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
}

.room-connection {
  opacity: 0.6;
  filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.2));
}



.text-visualization {
  height: 100%;
}

.observation-display h4 {
  color: #495057;
  margin-bottom: 1rem;
}

.observation-text {
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 4px;
  padding: 1rem;
  white-space: pre-wrap;
  font-family: 'Courier New', monospace;
  font-size: 0.85rem;
  line-height: 1.4;
  color: #495057;
  max-height: 400px;
  overflow-y: auto;
}

.loading-viz {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 200px;
  color: #6c757d;
}

.info-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  min-width: 180px;
  max-width: 200px;
}

.status-section,
.inventory-section,
.actions-section {
  background: white;
  border-radius: 6px;
  padding: 0.5rem;
  box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}

.status-section h4,
.inventory-section h4,
.actions-section h4 {
  margin: 0 0 0.4rem 0;
  color: #495057;
  font-size: 0.8rem;
}

.status-info {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.status-info > div {
  color: #6c757d;
  font-size: 0.75rem;
}

.completion-badge {
  background: #d4edda;
  color: #155724;
  padding: 0.5rem;
  border-radius: 4px;
  text-align: center;
  font-weight: bold;
}

.inventory-content,
.actions-content {
  max-height: 120px;
  overflow-y: auto;
}

.inventory-items,
.action-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.inventory-item {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.4rem;
  background: #f8f9fa;
  border-radius: 4px;
  margin-bottom: 0.3rem;
}

.item-icon {
  font-size: 1rem;
}

.item-name {
  color: #495057;
  font-size: 0.75rem;
}

.action-item {
  padding: 0.3rem;
  background: #f8f9fa;
  border-radius: 4px;
  color: #495057;
  font-size: 0.7rem;
  border-left: 2px solid #007bff;
  margin-bottom: 0.25rem;
}

.empty-inventory,
.no-actions {
  color: #6c757d;
  font-style: italic;
  text-align: center;
  padding: 1rem;
}

.more-actions {
  color: #6c757d;
  font-style: italic;
  text-align: center;
  padding: 0.5rem;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.9);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 10;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #007bff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Responsive design */
@media (max-width: 768px) {
  .visualization-container {
    flex-direction: column;
  }
  
  .scene-graph {
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  }
}
</style> 