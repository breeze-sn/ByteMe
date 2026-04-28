# ByteME

ByteME is a browser-based ASCII camera app that turns a live webcam feed into a real-time terminal-style portrait. It detects the human subject, emphasizes visible body parts, and renders the result in a green-on-black ASCII display that feels like a modern photobooth with a retro console aesthetic.

## What It Does

- Streams live webcam video directly in the browser
- Converts the camera feed into ASCII art in real time
- Uses machine vision to focus on the person instead of the background
- Highlights visible body parts, including hands, even when the face is not centered
- Lets you capture the current ASCII frame as an image and save it locally
- Uses a minimal camera UI with a shutter button and download flow

## Features

- Live camera preview with mirrored selfie behavior
- Real-time ASCII rendering on a black full-screen stage
- Human-focused segmentation so the background stays out of the output
- Hand-aware detection to improve body-part visibility
- Local image export with a save dialog or download fallback
- Clean, minimal controls styled for a camera-app feel

## Getting Started

### Prerequisites

- Node.js 18 or newer
- A browser that supports `getUserMedia`
- Camera permission enabled in the browser

### Install

```bash
npm install
```

### Run the app

```bash
npm run dev
```

Open the local URL shown in the terminal, allow camera access, and press **Start camera**.

### Build for production

```bash
npm run build
```

### Preview the production build

```bash
npm run preview
```

## Scripts

- `npm run dev` - start the development server
- `npm run start` - alias for `npm run dev`
- `npm run build` - create a production build
- `npm run preview` - preview the production build locally

## Tech Stack

- React 19 for the UI and app state
- Rsbuild for development, bundling, and production builds
- TensorFlow.js for running camera vision models in the browser
- BodyPix for real-time body-part segmentation
- @tensorflow-models/pose-detection for modern pose workflows
- @tensorflow-models/hand-pose-detection for hand-aware recognition
- @mediapipe/pose and @mediapipe/hands for optimized browser inference support

## Notes

- The app works best in a secure browser context such as `localhost` or HTTPS.
- If the camera output looks blank, check that camera permissions are granted and that the browser tab is active.
- Captured images are exported as PNG files and include the ByteME branding in the saved image.

## License

No license has been specified yet.
