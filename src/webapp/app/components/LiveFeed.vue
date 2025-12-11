<script setup lang="ts">
import { isConnected, stream } from "~/lib/websocket";
</script>

<template>
    <template v-if="isConnected">
        <img v-if="stream" :src="stream" alt="Live Camera Feed"/>
        <div v-else>Failed to get feed.</div>
    </template>
    <template v-else>
        <div>No connection</div>
    </template>
</template>