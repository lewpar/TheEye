<template>
  <div class="p-4">
    <h1>Live Stream</h1>

    <div v-if="imgUrl">
      <img :src="imgUrl" class="border rounded mt-4 max-w-full" />
    </div>

    <div v-else>
      <p>Waiting for stream...</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'

const imgUrl = ref(null)
let ws = null

onMounted(() => {
  // Connect to the Python server
  ws = new WebSocket("ws://localhost:8765")
  ws.binaryType = "arraybuffer"

  ws.onmessage = (event) => {
    const bytes = new Uint8Array(event.data)
    const blob = new Blob([bytes], { type: "image/jpeg" })
    imgUrl.value = URL.createObjectURL(blob)
  }

  ws.onopen = () => console.log("WebSocket connected")
  ws.onclose = () => console.log("WebSocket disconnected")
})

onBeforeUnmount(() => {
  if (ws) ws.close()
})
</script>

<style scoped>
img {
  max-width: 100%;
  display: block;
}
</style>
