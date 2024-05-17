import React from "react";
import StaticIDCapture from "./StaticIDCapture";
import CameraIDCapture from "./CameraIDCapture";
import "./style/App.css";

function App() {

  return (
    <div className="App">
      {/* <StaticIDCapture/> */}
      <CameraIDCapture/>
    </div>
  );
}

export default App;