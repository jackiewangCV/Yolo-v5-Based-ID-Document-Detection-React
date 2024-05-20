import React, { useCallback, useRef, useState, useEffect } from "react";
import cv from "@techstark/opencv-js";
// import Webcam from "react-webcam";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import { download } from "./utils/download";
import "./style/App.css";

function CameraIDCapture() {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({
    text: "Loading OpenCV.js",
    progress: null,
  });
  const [image, setImage] = useState(null);
  const [screenshot, setScreenshot] = useState(null);
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  let streaming = null;
  let processVideoInterval;

  const modelName = "id_detect.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const topk = 100;
  const iouThreshold = 0.45;
  const confThreshold = 0.2;
  const classThreshold = 0.2;

  const onClickVideoStream = () => {
    let video = document.getElementById("vid");
    let canvas = document.getElementById("canvas");
    let timeElement = document.getElementById("time");
    let buttonWebcamElement = document.getElementById("btn-webcam");

    if (streaming == null) {
      streaming = "camera";

      video.style.display = "block";
      canvas.style.display = "block";

      video.width = 640;
      video.height = 640;
      navigator.mediaDevices
        .getUserMedia({ video: true, audio: false })
        .then(function (stream) {
          video.srcObject = stream;
          video.play();

          let src = new cv.Mat(640, 640, cv.CV_8UC4);
          let cap = new cv.VideoCapture(video);

          async function processVideo() {
            try {
              if (!streaming) {
                src.delete();
                return;
              }
              let start = Date.now();
              cap.read(src);

              const result = await detectImage(
                src,
                canvas,
                session,
                topk,
                iouThreshold,
                confThreshold,
                classThreshold,
                modelInputShape,
                true
              );

              if (result && result.length > 0) {
                const screenshotCanvas = document.createElement("canvas");
                screenshotCanvas.width = video.videoWidth;
                screenshotCanvas.height = video.videoHeight;
                const screenshotCtx = screenshotCanvas.getContext("2d");
                screenshotCtx.drawImage(
                  video,
                  0,
                  0,
                  screenshotCanvas.width,
                  screenshotCanvas.height
                );
                const screenshotData = screenshotCanvas.toDataURL("image/png");
                console.log("result:::", result);
                setScreenshot(screenshotData);
              }

              let end = Date.now();
              let time = end - start;
              timeElement.innerHTML = "Time: " + time + "ms";
            } catch (err) {
              alert(err);
            }
          }

          processVideoInterval = setInterval(processVideo, 100); // Adjust interval to reduce CPU load
        })
        .catch(function (err) {
          console.log("An error occurred! " + err);
        });
    } else {
      streaming = null;
      video.style.display = "none";
      clearInterval(processVideoInterval);
      video.srcObject.getTracks().forEach(function (track) {
        track.stop();
      });
      timeElement.innerHTML = "Time: 0ms";
    }

    buttonWebcamElement.innerHTML =
      (streaming === "camera" ? "Close" : "Open") + " Webcam";
  };

  cv["onRuntimeInitialized"] = async () => {
    const baseModelURL = `${process.env.PUBLIC_URL}/model`;

    // create session
    const arrBufNet = await download(
      `${baseModelURL}/${modelName}`, // url
      ["Loading model", setLoading] // logger
    ); // get model arraybuffer
    let model = await InferenceSession.create(arrBufNet);
    const arrBufNMS = await download(
      `${baseModelURL}/nms-weight.onnx`, // url
      ["Loading NMS model", setLoading] // logger
    ); // get nms model arraybuffer
    const nms = await InferenceSession.create(arrBufNMS);

    // warmup model
    setLoading({ text: "Warming up model...", progress: null });
    const tensor = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    const config = new Tensor(
      "float32",
      new Float32Array([topk, iouThreshold, confThreshold])
    );
    const { output0 } = await model.run({ images: tensor });
    await nms.run({ detection: output0, config: config });

    setSession({ net: model, nms: nms });
    setLoading(null);
  };

  return (
    <div className="App">
      {loading && (
        <Loader>
          {loading.progress
            ? `${loading.text} - ${loading.progress}%`
            : loading.text}
        </Loader>
      )}

      <div className="header">
        <h1>ID Document Capture Web App</h1>
        <h4 id="time">0</h4>
        <p>
          ID document capture application live on browser powered by{" "}
          <code>onnxruntime-web</code>
        </p>
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
      </div>

      <div className="content">
        {screenshot ? (
          <img src={screenshot} alt="Screenshot" />
        ) : (
          <>
            <img
              src="#"
              alt=""
              style={{ display: image ? "block" : "none" }}
              ref={imageRef}
              onLoad={() => {
                detectImage(
                  imageRef.current,
                  canvasRef.current,
                  session,
                  topk,
                  iouThreshold,
                  confThreshold,
                  classThreshold,
                  modelInputShape
                );
              }}
            />
            <video
              id="vid"
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ inlineSize: "fit-content" }}
            />
            <canvas
              id="canvas"
              width={modelInputShape[2]}
              height={modelInputShape[3]}
              ref={canvasRef}
            />
          </>
        )}
      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
          }

          const url = URL.createObjectURL(e.target.files[0]);
          imageRef.current.src = url;
          setImage(url);
        }}
      />

      <div className="btn-container">
        <button
          onClick={() => {
            inputImage.current.click();
          }}
        >
          Open local image
        </button>
        {image && (
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
              const ctx = canvasRef.current.getContext("2d");
              ctx.clearRect(
                0,
                0,
                canvasRef.current.width,
                canvasRef.current.height
              );
            }}
          >
            Close image
          </button>
        )}
        <button id="btn-webcam" onClick={onClickVideoStream}>
          Open Webcam
        </button>
      </div>
    </div>
  );
}

export default CameraIDCapture;
