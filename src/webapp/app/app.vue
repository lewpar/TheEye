<script setup lang="ts">
import { connect, isConnecting, isConnected } from "~/lib/websocket";

connect();
</script>

<template>
  <UApp>
    <UMain>
      <NuxtPage/>
    </UMain>
  </UApp>
</template>
