let websocket: WebSocket | null = null;

export const isConnecting = ref<boolean>(false);
export const isConnected = ref<boolean>(false);
export let stream = ref<string | null>(null);

export function connect(maxAttempts: number = 5) {
    if(isConnected.value) {
        return;
    }

    try {
        console.log(`Connecting to websocket..`);

        isConnecting.value = true;

        websocket = new WebSocket("ws://localhost:8765");
        websocket.binaryType = "arraybuffer";

        websocket.onmessage = messageReceived;

        websocket.onopen = connected;
        websocket.onclose = disconnected;

        isConnecting.value = false;
    }
    catch(error) {
        isConnecting.value = false;
    }
}

export function disconnect() {
    if(!websocket) {
        return;
    }

    websocket.close();
}

function connected() {
    isConnected.value = true;
    console.log("Connected to websocket.");
}

function disconnected() {
    isConnected.value = false;
    console.log("Disconnected from websocket.");

    setTimeout(() => {
        connect();
    }, 2500);
}

function messageReceived(event: MessageEvent<any>) {
    const bytes = new Uint8Array(event.data)
    const blob = new Blob([bytes], { type: "image/jpeg" })
    stream.value = URL.createObjectURL(blob)
}