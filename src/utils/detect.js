import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox";

/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session onnxruntime session
 * @param {Number} topk Integer representing the maximum number of boxes to be selected per class
 * @param {Number} iouThreshold Float representing the threshold for deciding whether boxes overlap too much with respect to IOU
 * @param {Number} confThreshold Float representing the threshold for deciding when to remove boxes based on confidence score
 * @param {Number} classThreshold class threshold
 * @param {Number[]} inputShape model input shape. [batch, channels, width, height]
 */
export const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  confThreshold,
  classThreshold,
  inputShape,
  isVideo,
  callback = () => { },
) => {
  const [modelWidth] = inputShape.slice(2);
  const [modelHeight] = inputShape.slice(3);
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight, isVideo);

  const tensor = new Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const config = new Tensor("float32", new Float32Array([topk, iouThreshold, confThreshold])); // nms config tensor
  const start = Date.now();
  const { output0 } = await session.net.run({ images: tensor }); // run session and get output layer
  const { selected_idx } = await session.nms.run({ detection: output0, config: config }); // get selected idx from nms
  // console.log(Date.now() - start);

  const boxes = [];

  // looping through output
  selected_idx.data.forEach((idx) => {
    const data = output0.data.slice(idx * output0.dims[2], (idx + 1) * output0.dims[2]); // get rows
    const [x, y, w, h] = data.slice(0, 4);
    const confidence = data[4]; // detection confidence
    const scores = data.slice(5); // classes probability scores
    let score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores
    score *= confidence; // multiply score by conf
    
    // filtering by score thresholds
    if ((score >= classThreshold) && (label == 3)){
      if (Math.floor(w * xRatio) / Math.floor(h * yRatio) > 0.8) // width/height > 0.8
        boxes.push({
          label: label,
          probability: score,
          bounding: [
            Math.floor((x - 0.5 * w) * xRatio), // left
            Math.floor((y - 0.5 * h) * yRatio), //top
            Math.floor(w * xRatio), // width
            Math.floor(h * yRatio), // height
          ],
        });
    }

  });

  renderBoxes(canvas, boxes); // Draw boxes

  callback();
  // release mat opencv
  input.delete();
};

/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @return preprocessed image and configs
 */
const preprocessing = (source, modelWidth, modelHeight, isVideo) => {
  const mat = isVideo ? source : cv.imread(source); // read from img tag
  
  // padding image to [n x n] dim
  const maxSize = Math.max(mat.rows, mat.cols); // get max size from width and height
  const xPad = maxSize - mat.cols, // set xPadding
    xRatio = maxSize / mat.cols; // set xRatio
    const yPad = maxSize - mat.rows, // set yPadding
    yRatio = maxSize / mat.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image

  cv.copyMakeBorder(mat, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // padding black

  cv.cvtColor(matPad, matPad, cv.COLOR_BGRA2BGR); // RGBA to BGR
  const input = cv.blobFromImage(
    matPad,
    1 / 255.0, // normalize
    new cv.Size(modelWidth, modelHeight), // resize to model input size
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  ); // preprocessing image matrix
  
  // release mat opencv
  // mat.delete();
  matPad.delete();
  return [input, xRatio, yRatio];
};


