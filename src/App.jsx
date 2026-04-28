import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as bodyPix from "@tensorflow-models/body-pix";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const frameRef = useRef(0);
  const isProcessingRef = useRef(false);
  const frameTickRef = useRef(0);
  const modelRef = useRef(null);
  const [cameraState, setCameraState] = useState("idle");
  const [cameraError, setCameraError] = useState("");
  const [modelReady, setModelReady] = useState(false);
  const [asciiArt, setAsciiArt] = useState("Loading AI model...");

  const asciiPalette = "@#S%?*+;:,. ";

  const stopCamera = () => {
    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = 0;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraState("idle");
    setAsciiArt("Camera stopped. Start again to resume ASCII art.");
  };

  const capturePicture = async () => {
    if (cameraState !== "live" || !asciiArt.trim()) {
      setCameraError("Start the camera and wait for ASCII art before capturing.");
      return;
    }

    const asciiView = document.querySelector(".ascii-view");
    const computedStyle = asciiView ? window.getComputedStyle(asciiView) : null;
    const rootStyle = window.getComputedStyle(document.documentElement);
    const fontSize = computedStyle ? Number.parseFloat(computedStyle.fontSize) : 16;
    const lineHeight = computedStyle ? Number.parseFloat(computedStyle.lineHeight) : fontSize;
    const fontFamily = computedStyle?.fontFamily || "'Courier New', Courier, monospace";
    const textColor = computedStyle?.color || "#009B77";
    const backgroundColor = computedStyle?.backgroundColor || "#000000";
    const footerColor = "#ffffff";
    const footerFontFamily = rootStyle.fontFamily || "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

    const captureCanvas = document.createElement("canvas");
    const pixelRatio = window.devicePixelRatio || 1;
    captureCanvas.width = Math.floor(window.innerWidth * pixelRatio);
    captureCanvas.height = Math.floor(window.innerHeight * pixelRatio);
    captureCanvas.style.width = `${window.innerWidth}px`;
    captureCanvas.style.height = `${window.innerHeight}px`;

    const context = captureCanvas.getContext("2d");

    if (!context) {
      setCameraError("Unable to create a capture canvas.");
      return;
    }

    context.scale(pixelRatio, pixelRatio);
    context.fillStyle = backgroundColor;
    context.fillRect(0, 0, window.innerWidth, window.innerHeight);
    context.fillStyle = textColor;
    context.font = `${fontSize}px ${fontFamily}`;
    context.textBaseline = "top";
    context.letterSpacing = "-0.05em";

    const lines = asciiArt.replace(/\n$/, "").split("\n");
    const textWidths = lines.map((line) => context.measureText(line).width);
    const maxTextWidth = textWidths.length ? Math.max(...textWidths) : 0;
    const totalTextHeight = lineHeight * lines.length;
    const offsetX = Math.max(0, (window.innerWidth - maxTextWidth) / 2);
    const offsetY = Math.max(0, (window.innerHeight - totalTextHeight) / 2);

    let y = offsetY;
    for (const line of lines) {
      context.fillText(line, offsetX, y);
      y += lineHeight;
    }

    const footerTitle = "ByteME";
    const footerCaption = "Made with care (and characters)";
    const footerFontSize = Math.max(14, Math.floor(fontSize * 1.05));
    const footerLineGap = Math.max(8, Math.floor(footerFontSize * 0.4));
    const footerPadding = Math.max(24, Math.floor(window.innerWidth * 0.025));

    context.fillStyle = footerColor;
    context.font = `700 ${footerFontSize}px ${footerFontFamily}`;
    context.textAlign = "right";
    context.textBaseline = "alphabetic";

    const footerRightEdge = window.innerWidth - footerPadding;
    const footerCaptionY = window.innerHeight - footerPadding;
    const footerTitleY = footerCaptionY - footerFontSize - footerLineGap;

    context.fillText(footerTitle, footerRightEdge, footerTitleY);
    context.font = `${footerFontSize}px ${footerFontFamily}`;
    context.fillText(footerCaption, footerRightEdge, footerCaptionY);

    captureCanvas.toBlob(async (blob) => {
      if (!blob) {
        setCameraError("Unable to capture the ASCII image.");
        return;
      }

      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const fileName = `byteme-capture-${timestamp}.png`;

      try {
        if (window.showSaveFilePicker) {
          const fileHandle = await window.showSaveFilePicker({
            suggestedName: fileName,
            types: [
              {
                description: "PNG Image",
                accept: { "image/png": [".png"] },
              },
            ],
          });

          const writable = await fileHandle.createWritable();
          await writable.write(blob);
          await writable.close();
          return;
        }

        const downloadUrl = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = downloadUrl;
        link.download = fileName;
        link.click();
        URL.revokeObjectURL(downloadUrl);
      } catch (error) {
        console.error("Capture save failed:", error);
        setCameraError("Could not save the captured image.");
      }
    }, "image/png");
  };

  const renderAsciiFrame = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const model = modelRef.current;

    if (!video || !canvas || video.readyState < 2 || !model) {
      frameRef.current = requestAnimationFrame(renderAsciiFrame);
      return;
    }

    if (isProcessingRef.current) {
      frameRef.current = requestAnimationFrame(renderAsciiFrame);
      return;
    }

    isProcessingRef.current = true;
    frameTickRef.current += 1;

    const context = canvas.getContext("2d", { willReadFrequently: true });

    if (!context) {
      isProcessingRef.current = false;
      frameRef.current = requestAnimationFrame(renderAsciiFrame);
      return;
    }

    // Calculate columns to fill screen width
    const targetColumns = 200; // wider ASCII for full screen
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    context.drawImage(video, 0, 0);
    const fullPixelData = context.getImageData(0, 0, video.videoWidth, video.videoHeight).data;

    try {
      const segmentationPromise = model.segmentPersonParts(video, {
        flipHorizontal: false,
        internalResolution: "low",
        segmentationThreshold: 0.5,
      });

      const segmentation = await segmentationPromise;

      if (!segmentation) {
        console.warn("No body-part segmentation object");
        frameRef.current = requestAnimationFrame(renderAsciiFrame);
        return;
      }

      const segmentationData = segmentation.data;
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;
      const aspectRatio = videoHeight / videoWidth;
      
      // Calculate rows based on fixed columns and camera aspect ratio
      const targetRows = Math.round(targetColumns * aspectRatio * 0.5);

      if (!segmentationData || segmentationData.length === 0) {
        console.warn("Empty segmentation data", {
          dataLength: segmentationData?.length,
          videoSize: `${videoWidth}x${videoHeight}`,
        });
        frameRef.current = requestAnimationFrame(renderAsciiFrame);
        return;
      }

      // Calculate mask dimensions based on data length
      const maskArea = segmentationData.length;
      const maskWidth = Math.sqrt(maskArea / aspectRatio);
      const maskHeight = maskArea / maskWidth;
      let ascii = "";
      let personPixelCount = 0;

      for (let row = 0; row < targetRows; row += 1) {
        let line = "";

        for (let col = 0; col < targetColumns; col += 1) {
          // Map from target grid to mask dimensions (mirrored)
          const maskRow = Math.floor((row / targetRows) * maskHeight);
          const mirroredCol = targetColumns - 1 - col;
          const maskCol = Math.floor((mirroredCol / targetColumns) * maskWidth);
          const maskIndex = maskRow * Math.floor(maskWidth) + maskCol;

          // Map from target grid to video dimensions for pixel data (mirrored)
          const videoRow = Math.floor((row / targetRows) * videoHeight);
          const videoCol = Math.floor((mirroredCol / targetColumns) * videoWidth);
          const videoIndex = videoRow * videoWidth + videoCol;

          const isHumanPixel = segmentationData[maskIndex] !== -1;

          if (isHumanPixel) personPixelCount += 1;

          if (!isHumanPixel) {
            line += " ";
          } else {
            const pixelIndex = videoIndex * 4;
            const red = fullPixelData[pixelIndex];
            const green = fullPixelData[pixelIndex + 1];
            const blue = fullPixelData[pixelIndex + 2];
            const brightness = (red * 0.299 + green * 0.587 + blue * 0.114) / 255;
            const paletteIndex = Math.min(
              asciiPalette.length - 1,
              Math.floor((1 - brightness) * (asciiPalette.length - 1)),
            );

            line += asciiPalette[paletteIndex];
          }
        }

        ascii += `${line}\n`;
      }

      if (ascii.trim()) {
        setAsciiArt(ascii);
      }
    } catch (error) {
      console.error("Segmentation error:", error);
    } finally {
      isProcessingRef.current = false;
    }

    frameRef.current = requestAnimationFrame(renderAsciiFrame);
  };

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("This browser does not support live camera access.");
      setCameraState("unsupported");
      return;
    }

    try {
      setCameraError("");
      setCameraState("starting");

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
        },
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraState("live");
      setAsciiArt("");
      frameRef.current = requestAnimationFrame(renderAsciiFrame);
    } catch (error) {
      setCameraState("error");
      setCameraError(
        error instanceof Error
          ? error.message
          : "Camera access was blocked or could not be started.",
      );
      stopCamera();
    }
  };

  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log("Loading body-pix model...");
        const model = await bodyPix.load({
          architecture: "MobileNetV1",
          outputStride: 16,
          multiplier: 0.75,
          quantBytes: 2,
        });

        modelRef.current = model;
        setModelReady(true);
        setAsciiArt("Model loaded! Click START CAMERA to begin.");
        console.log("Model loaded successfully");
      } catch (error) {
        console.error("Failed to load model:", error);
        setAsciiArt("Error loading AI model. Check console.");
      }
    };

    loadModel();
    return stopCamera;
  }, []);

  return (
    <main className="camera-shell">
      <section className="camera-stage">
        <video ref={videoRef} autoPlay muted playsInline className="camera-source" />
        <canvas ref={canvasRef} className="camera-canvas" aria-hidden="true" />
        <pre className="ascii-view" aria-live="polite" aria-label="Live ASCII camera feed">
          {asciiArt}
        </pre>
        {cameraState !== "live" && <div className="camera-placeholder">Camera Disabled</div>}

        <header className="camera-topbar">
          <div>
            <p className="eyebrow">ByteME</p>
            <p className="camera-title">ASCII Photobooth</p>
          </div>

          <div className="camera-actions">
            <button className="button primary" type="button" onClick={startCamera}>
              {cameraState === "live" ? "Restart" : "Start camera"}
            </button>
            {cameraState === "live" && (
              <button className="button secondary" type="button" onClick={stopCamera}>
                Stop
              </button>
            )}
          </div>
        </header>

        <div className="camera-bottom">
          <div className="camera-bottom-row">
            <p className="camera-status" data-state={cameraState}>
              {cameraState === "live" && "Camera is live."}
              {cameraState === "starting" && "Requesting camera access..."}
              {cameraState === "unsupported" && "Live camera is not supported in this browser."}
              {cameraState === "error" && cameraError}
              {cameraState === "idle" && "Camera is idle."}
            </p>

            <button
              className="button capture-button"
              type="button"
              onClick={capturePicture}
              aria-label="Capture picture"
              title="Capture picture"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                <path d="M9 3.75 7.59 5.5H5.5A2.5 2.5 0 0 0 3 8v8A2.5 2.5 0 0 0 5.5 18.5h13A2.5 2.5 0 0 0 21 16V8a2.5 2.5 0 0 0-2.5-2.5h-2.09L15 3.75H9Zm3 11.5A3.25 3.25 0 1 1 12 8a3.25 3.25 0 0 1 0 6.5Zm0-1.75A1.5 1.5 0 1 0 12 10.5a1.5 1.5 0 0 0 0 3Z" />
              </svg>
            </button>

            <p className="camera-hint">Made with care (and characters)</p>
          </div>
        </div>
      </section>
    </main>
  );
}
