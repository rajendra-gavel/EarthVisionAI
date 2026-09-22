import { useState } from "react";
import axios from "axios";
import {
  Satellite,
  Search,
  ArrowRight,
  Activity,
  Bot,
  ScanSearch,
  MapPin,
  Layers,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";
import "./App.css";

const API_BASE = "http://localhost:9090/api/v1";
const API_SERVER = "http://localhost:9090";

const beforeImage =
  `${API_SERVER}/data/raw/india/iccd/sample/ICCD_Sample/Agra/labeled/im1/Agra_0_2022_r00_c01.png`;

const afterImage =
  `${API_SERVER}/data/raw/india/iccd/sample/ICCD_Sample/Agra/labeled/im2/Agra_0_2023_r00_c01.png`;


/* =========================================================
   MODEL INFORMATION
   ========================================================= */

const modelInfo = {
  aerial: {
    name: "YOLO26n-OBB",
    description:
      "Optimized for supported aerial and geospatial feature classes.",

    classes: [
      "Plane",
      "Ship",
      "Storage Tank",
      "Baseball Diamond",
      "Tennis Court",
      "Basketball Court",
      "Ground Track Field",
      "Harbor",
      "Bridge",
      "Large Vehicle",
      "Small Vehicle",
      "Helicopter",
      "Roundabout",
      "Soccer Ball Field",
      "Swimming Pool",
    ],
  },

  general: {
    name: "YOLO11n",
    description:
      "General-purpose object detection using supported COCO categories.",

    classes: [
      "Person",
      "Bicycle",
      "Car",
      "Motorcycle",
      "Airplane",
      "Bus",
      "Train",
      "Truck",
      "Boat",
      "Animals",
      "Sports Objects",
      "Furniture",
      "Electronics",
      "Kitchen Objects",
      "And other COCO classes",
    ],
  },
};


/* =========================================================
   DEMO IMAGES
   ========================================================= */

const demoImages = {
  aerial: [
    {
      name: "DOTA8 — Aerial Demo 1",
      path:
        "/data/demo/aerial/dota8/images/train/P0861__1024__0___1648.jpg",
    },
    {
      name: "DOTA8 — Aerial Demo 2",
      path:
        "/data/demo/aerial/dota8/images/train/P1053__1024__0___90.jpg",
    },
    {
      name: "DOTA8 — Aerial Demo 3",
      path:
        "/data/demo/aerial/dota8/images/train/P1142__1024__0___824.jpg",
    },
    {
      name: "DOTA8 — Aerial Demo 4",
      path:
        "/data/demo/aerial/dota8/images/train/P1161__1024__3296___1648.jpg",
    },
    {
      name: "DOTA8 — Aerial Demo 5",
      path:
        "/data/demo/aerial/dota8/images/val/P1470__1024__3296___1648.jpg",
    },
    {
      name: "DOTA8 — Aerial Demo 6",
      path:
        "/data/demo/aerial/dota8/images/val/P1571__1024__2976___0.jpg",
    },
    {
      name: "DOTA8 — Aerial Demo 7",
      path:
        "/data/demo/aerial/dota8/images/val/P1580__1024__824___824.jpg",
    },
    {
      name: "DOTA8 — Aerial Demo 8",
      path:
        "/data/demo/aerial/dota8/images/val/P1724__1024__0___824.jpg",
    },
  ],

  general: [
    {
      name: "COCO8 — General Demo 1",
      path:
        "/data/demo/general/coco8/images/train/000000000009.jpg",
    },
    {
      name: "COCO8 — General Demo 2",
      path:
        "/data/demo/general/coco8/images/train/000000000025.jpg",
    },
    {
      name: "COCO8 — General Demo 3",
      path:
        "/data/demo/general/coco8/images/train/000000000030.jpg",
    },
    {
      name: "COCO8 — General Demo 4",
      path:
        "/data/demo/general/coco8/images/train/000000000034.jpg",
    },
    {
      name: "COCO8 — General Demo 5",
      path:
        "/data/demo/general/coco8/images/val/000000000036.jpg",
    },
    {
      name: "COCO8 — General Demo 6",
      path:
        "/data/demo/general/coco8/images/val/000000000042.jpg",
    },
    {
      name: "COCO8 — General Demo 7",
      path:
        "/data/demo/general/coco8/images/val/000000000049.jpg",
    },
    {
      name: "COCO8 — General Demo 8",
      path:
        "/data/demo/general/coco8/images/val/000000000061.jpg",
    },
  ],
};


/* =========================================================
   SMALL HELPERS
   ========================================================= */

function formatPercent(value) {
  if (value === undefined || value === null) {
    return "—";
  }

  const number = Number(value);

  if (Number.isNaN(number)) {
    return "—";
  }

  return `${number.toFixed(2)}%`;
}


function formatConfidence(value) {
  if (value === undefined || value === null) {
    return "—";
  }

  return `${(Number(value) * 100).toFixed(1)}%`;
}


/* =========================================================
   APP
   ========================================================= */

function App() {

  /* -------------------------------------------------------
     CHANGE ANALYSIS STATE
     ------------------------------------------------------- */

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");


  /* -------------------------------------------------------
     OBJECT DETECTION STATE
     ------------------------------------------------------- */

  const [objectResult, setObjectResult] = useState(null);
  const [objectLoading, setObjectLoading] = useState(false);
  const [objectError, setObjectError] = useState("");

  const [selectedObjectFile, setSelectedObjectFile] =
    useState(null);

  const [objectPreview, setObjectPreview] =
    useState("");

  const [objectModel, setObjectModel] =
    useState("aerial");


  /* -------------------------------------------------------
     AI ASSISTANT STATE
     ------------------------------------------------------- */

  const [question, setQuestion] = useState("");
  const [assistantAnswer, setAssistantAnswer] =
    useState("");

  const [assistantLoading, setAssistantLoading] =
    useState(false);

  const [assistantError, setAssistantError] =
    useState("");


  /* =======================================================
     CHANGE ANALYSIS
     ======================================================= */

  const analyzeChanges = async () => {

    setLoading(true);
    setError("");

    try {

      const response = await axios.post(
        `${API_BASE}/change-detection`,
        {
          before:
            "/mnt/d/pocs/EarthVisionAI/data/raw/india/iccd/sample/ICCD_Sample/Agra/labeled/im1/Agra_0_2022_r00_c01.png",

          after:
            "/mnt/d/pocs/EarthVisionAI/data/raw/india/iccd/sample/ICCD_Sample/Agra/labeled/im2/Agra_0_2023_r00_c01.png",
        }
      );

      setResult(response.data);

    } catch (err) {

      console.error(err);

      setError(
        err.response?.data?.detail ||
        "Unable to connect to EarthVision AI backend."
      );

    } finally {

      setLoading(false);

    }
  };


  /* =======================================================
     OBJECT DETECTION
     ======================================================= */

  const detectObjects = async (
    file,
    model = objectModel
  ) => {

    if (!file) {

      setObjectError(
        "Please select an image first."
      );

      return;
    }

    setObjectLoading(true);
    setObjectError("");
    setObjectResult(null);

    try {

      const formData = new FormData();

      formData.append("image", file);
      formData.append("model", model);

      const response = await axios.post(
        `${API_BASE}/object-detection/upload`,
        formData
      );

      setObjectResult(response.data);

    } catch (err) {

      console.error(
        "Object detection error:",
        err
      );

      setObjectError(
        err.response?.data?.detail ||
        "Unable to run object detection."
      );

    } finally {

      setObjectLoading(false);

    }
  };


  /* =======================================================
     UPLOAD IMAGE
     ======================================================= */

  const handleObjectImageChange =
    async (event) => {

      const file =
        event.target.files?.[0];

      if (!file) {
        return;
      }

      if (
        ![
          "image/jpeg",
          "image/png",
        ].includes(file.type)
      ) {

        setObjectError(
          "Please select a JPG or PNG image."
        );

        return;
      }

      if (
        file.size >
        15 * 1024 * 1024
      ) {

        setObjectError(
          "Image size must be 15 MB or smaller."
        );

        return;
      }

      setSelectedObjectFile(file);
      setObjectError("");
      setObjectResult(null);

      const previewUrl =
        URL.createObjectURL(file);

      setObjectPreview(previewUrl);

      /*
       * Automatic inference is intentionally preserved.
       */
      await detectObjects(
        file,
        objectModel
      );
    };


  /* =======================================================
     LOAD DEMO IMAGE
     ======================================================= */

  const loadDemoImage = async (
    demoPath,
    demoName,
    demoModel
  ) => {

    try {

      setObjectError("");
      setObjectResult(null);
      setObjectModel(demoModel);
      setObjectLoading(true);

      const response =
        await fetch(
          `${API_SERVER}${demoPath}`
        );

      if (!response.ok) {

        throw new Error(
          "Unable to load demo image."
        );
      }

      const blob =
        await response.blob();

      const extension =
        blob.type === "image/png"
          ? ".png"
          : ".jpg";

      const file =
        new File(
          [blob],
          `${demoName}${extension}`,
          {
            type: blob.type,
          }
        );

      setSelectedObjectFile(file);

      const previewUrl =
        URL.createObjectURL(file);

      setObjectPreview(previewUrl);

      await detectObjects(
        file,
        demoModel
      );

    } catch (err) {

      console.error(err);

      setObjectError(
        "Unable to load the selected demo image."
      );

    } finally {

      setObjectLoading(false);

    }
  };


  /* =======================================================
     MODEL CHANGE
     ======================================================= */

  const handleModelChange =
    async (event) => {

      const newModel =
        event.target.value;

      setObjectModel(newModel);
      setObjectError("");
      setObjectResult(null);

      if (selectedObjectFile) {

        await detectObjects(
          selectedObjectFile,
          newModel
        );
      }
    };


  /* =======================================================
     AI ASSISTANT
     ======================================================= */

  const askAssistant = async () => {

    if (!question.trim()) {
      return;
    }

    setAssistantLoading(true);
    setAssistantError("");
    setAssistantAnswer("");

    const payload = {
      question: question.trim(),
      change_result: result,
      object_result: objectResult,
    };

    try {

      const response =
        await axios.post(
          `${API_BASE}/assistant`,
          payload
        );

      setAssistantAnswer(
        response.data.answer ||
        "No answer returned."
      );

    } catch (err) {

      console.error(
        "EarthVision Assistant error:",
        err
      );

      setAssistantError(
        err.response?.data?.detail ||
        "Unable to get a response from EarthVision AI Assistant."
      );

    } finally {

      setAssistantLoading(false);

    }
  };


  /* =======================================================
     SUGGESTED QUESTION
     ======================================================= */

  const useSuggestedQuestion = (text) => {

    setQuestion(text);

  };


  /* =======================================================
     RENDER
     ======================================================= */

  return (

    <div className="app">

      {/* =================================================
          HEADER
          ================================================= */}

      <header className="header">

        <div className="brand">

          <div className="brand-icon">
            <Satellite size={25} />
          </div>

          <div>
            <h1>EarthVision AI</h1>

            <p>
              Satellite Image Intelligence
            </p>
          </div>

        </div>


        <div className="status">

          <span className="status-dot" />

          AI Engine Online

        </div>

      </header>


      <main className="container">

        {/* =================================================
            HERO
            ================================================= */}

        <section className="hero">

          <div className="hero-eyebrow">
            <Sparkles size={14} />
            EARTH OBSERVATION AI
          </div>

          <h2>
            Satellite Image Analysis
          </h2>

          <div className="hero-subtitle">
            Analyze satellite imagery, identify
            potential visual differences, and
            detect supported geographic features using AI-powered computer vision.
          </div>

        </section>


        {/* =================================================
            SATELLITE ANALYSIS
            ================================================= */}

        <section className="analysis-section">

          <div className="section-heading">

            <div>

              <span className="label">
                SATELLITE ANALYSIS
              </span>

              <h2>
                Before / After Imagery
              </h2>

              <p>
                Compare imagery captured across
                two time periods.
              </p>

            </div>

            <div className="location-badge">

              <MapPin size={15} />

              Agra, India

            </div>

          </div>


          <section className="comparison">

            {/* BEFORE */}

            <div className="image-card">

              <div className="image-header">

                <span className="label">
                  BEFORE
                </span>

                <h3>
                  Agra — 2022
                </h3>

              </div>

              <img
                src={beforeImage}
                alt="Agra satellite imagery from 2022"
              />

            </div>


            {/* ARROW */}

            <div className="arrow">

              <ArrowRight size={23} />

            </div>


            {/* AFTER */}

            <div className="image-card">

              <div className="image-header">

                <span className="label">
                  AFTER
                </span>

                <h3>
                  Agra — 2023
                </h3>

              </div>

              <img
                src={afterImage}
                alt="Agra satellite imagery from 2023"
              />

            </div>

          </section>


          {/* ACTION */}

          <div className="action-area">

            <button
              className="analyze-button"
              onClick={analyzeChanges}
              disabled={loading}
            >

              <Search size={18} />

              {loading
                ? "Analyzing imagery..."
                : "Analyze Imagery"}

            </button>

          </div>


          {error && (
            <div className="error">
              {error}
            </div>
          )}


          {loading && (

            <div className="detection-status">

              <Activity size={18} />

              <span>
                EarthVision is analyzing
                the satellite imagery...
              </span>

            </div>

          )}


          {/* =================================================
              EXPERIMENTAL CHANGE ANALYSIS
              ================================================= */}

          {result && (

            <section className="results">

              <div className="section-title">

                <Layers size={21} />

                <div>

                  <span className="label">
                    EXPERIMENTAL ANALYSIS
                  </span>

                  <h2>
                    Potential Visual Differences
                  </h2>

                </div>

              </div>


              <div className="metrics">

                <div className="metric">

                  <span>
                    Potential Difference
                  </span>

                  <strong>
                    {formatPercent(
                      result.change_percentage
                    )}
                  </strong>

                </div>


                <div className="metric">

                  <span>
                    Candidate Regions
                  </span>

                  <strong>
                    {result.regions?.length ??
                      result.candidate_regions ??
                      "—"}
                  </strong>

                </div>


                <div className="metric">

                  <span>
                    Alignment
                  </span>

                  <strong className="metric-status">

                    {result.alignment ? (
                      <>
                        <CheckCircle2
                          size={17}
                        />
                        Registered
                      </>
                    ) : (
                      "Not available"
                    )}

                  </strong>

                </div>


                <div className="metric">

                  <span>
                    Image Size
                  </span>

                  <strong>
                    {result.width && result.height
                      ? `${result.width} × ${result.height}`
                      : "—"}
                  </strong>

                </div>

              </div>


              {/* METHOD */}

              <div className="method-card">

                <div>

                  <span className="label">
                    ANALYSIS METHOD
                  </span>

                  <strong>
                    {result.method_label ||
                      result.method ||
                      "Structural + Spectral Analysis"}
                  </strong>

                </div>

                {result.experimental && (

                  <span className="experimental-badge">
                    Experimental
                  </span>

                )}

              </div>


              {/* OVERLAY */}

              {(result.change_overlay ||
                result.change_map) && (

                <div className="change-map-section">

                  <div className="change-map-header">

                    <div>

                      <span className="label">
                        POTENTIAL DIFFERENCES
                      </span>

                      <h3>
                        Analysis Overlay
                      </h3>

                    </div>

                  </div>


                  <div className="change-map-container">

                    <img
                      src={`${API_SERVER}${
                        result.change_overlay ||
                        result.change_map
                      }`}
                      alt="Potential visual difference analysis overlay"
                    />

                  </div>


                  <p className="change-map-description">

                    Highlighted regions indicate
                    areas of potential visual difference
                    identified by the experimental
                    structural and spectral analysis.

                  </p>

                </div>

              )}


              {/* CHANGE INTENSITY */}

              {result.change_score && (

                <div className="change-map-section">

                  <div className="change-map-header">

                    <div>

                      <span className="label">
                        ANALYSIS SIGNAL
                      </span>

                      <h3>
                        Change Intensity
                      </h3>

                    </div>

                  </div>


                  <div className="change-map-container">

                    <img
                      src={`${API_SERVER}${result.change_score}`}
                      alt="Change intensity analysis"
                    />

                  </div>

                </div>

              )}


              {/* REGIONS */}

              {result.regions?.length > 0 && (

                <div className="regions-section">

                  <div className="section-subtitle">

                    <span className="label">
                      CANDIDATE REGIONS
                    </span>

                    <span>
                      {result.regions.length} regions
                    </span>

                  </div>


                  <div className="region-grid">

                    {result.regions
                      .slice(0, 12)
                      .map((region) => (

                        <div
                          className="region-card"
                          key={region.id}
                        >

                          <div>

                            <strong>
                              Region {region.id}
                            </strong>

                            <span>
                              {region.area_pixels
                                ?.toLocaleString() ||
                                "—"}{" "}
                              pixels
                            </span>

                          </div>

                          <span
                            className={`severity ${(
                              region.severity ||
                              "low"
                            ).toLowerCase()}`}
                          >
                            {region.severity ||
                              "low"}
                          </span>

                        </div>

                      ))}

                  </div>

                </div>

              )}


              {/* DISCLAIMER */}

              <div className="baseline-warning">

                <div className="warning-heading">

                  <AlertTriangle size={18} />

                  <strong>
                    Experimental Analysis
                  </strong>

                </div>

                <p>
                  Highlighted regions represent
                  potential visual differences.
                  They may include effects from
                  illumination, seasonal conditions,
                  acquisition differences, or residual
                  image-registration errors. They should
                  not be interpreted as confirmed
                  geographic change.
                </p>

                {result.note && (
                  <p>
                    {result.note}
                  </p>
                )}

              </div>

            </section>

          )}

        </section>


        {/* =================================================
            OBJECT / FEATURE DETECTION
            ================================================= */}

        <section className="analysis-section object-section">

          <div className="section-heading">

            <div>

              <span className="label">
                AI VISION
              </span>

              <h2>
                Dynamic Object / Feature Detection
              </h2>

              <p>
                Upload imagery or select a prepared
                demo and run the appropriate computer
                vision model automatically.
              </p>

            </div>

          </div>


          {/* CONTROLS */}

          <div className="upload-panel">

            <div className="upload-controls">

              <label className="upload-label">
                Upload Image
              </label>

              <input
                className="file-input"
                type="file"
                accept="image/png,image/jpeg"
                onChange={
                  handleObjectImageChange
                }
              />

            </div>


            <div className="upload-controls">

              <label className="upload-label">
                Detection Model
              </label>

              <select
                value={objectModel}
                onChange={
                  handleModelChange
                }
              >

                <option value="aerial">
                  Aerial / Geospatial — YOLO26n-OBB
                </option>

                <option value="general">
                  General Objects — YOLO11n
                </option>

              </select>

            </div>


            <div className="upload-controls">

              <label className="upload-label">
                Demo Image
              </label>

              <select
                value=""
                onChange={async (e) => {

                  const selectedPath =
                    e.target.value;

                  if (!selectedPath) {
                    return;
                  }

                  const selectedDemo =
                    demoImages[objectModel].find(
                      (item) =>
                        item.path === selectedPath
                    );

                  if (selectedDemo) {

                    await loadDemoImage(
                      selectedDemo.path,
                      selectedDemo.name,
                      objectModel
                    );

                  }

                }}
              >

                <option value="">
                  Select a demo image
                </option>

                {demoImages[objectModel].map(
                  (demo) => (

                    <option
                      key={demo.path}
                      value={demo.path}
                    >
                      {demo.name}
                    </option>

                  )
                )}

              </select>

            </div>


            {/* MODEL INFO */}

            <div className="model-info">

              <div className="model-info-header">

                <strong>
                  {modelInfo[objectModel].name}
                </strong>

              </div>

              <div className="model-description">

                {modelInfo[
                  objectModel
                ].description}

              </div>

              <div className="class-tags">

                {modelInfo[
                  objectModel
                ].classes.map(
                  (className) => (

                    <span
                      className="class-tag"
                      key={className}
                    >
                      {className}
                    </span>

                  )
                )}

              </div>


              {objectModel === "aerial" && (

                <div className="model-note">

                  Building detection is not included
                  in this model's trained classes.

                </div>

              )}

            </div>


            {selectedObjectFile && (

              <div className="selected-file">

                <span>
                  Selected:{" "}
                  {selectedObjectFile.name}
                </span>

                <span>

                  {(
                    selectedObjectFile.size /
                    1024 /
                    1024
                  ).toFixed(2)} MB

                </span>

              </div>

            )}

          </div>


          {/* IMAGE COMPARISON */}

          <div className="object-comparison">

            {/* INPUT */}

            <div className="image-card">

              <div className="image-header">

                <span className="label">
                  INPUT IMAGE
                </span>

                <h3>
                  Source Imagery
                </h3>

              </div>

              {objectPreview ? (

                <img
                  src={objectPreview}
                  alt="Uploaded source image"
                />

              ) : (

                <div className="detection-placeholder">

                  <ScanSearch size={38} />

                  <span>
                    Select a JPG or PNG image
                    to begin detection.
                  </span>

                </div>

              )}

            </div>


            {/* AI RESULT */}

            <div className="image-card">

              <div className="image-header">

                <span className="label">
                  AI DETECTION
                </span>

                <h3>
                  {objectResult
                    ? objectResult.model
                    : "Detection Result"}
                </h3>

              </div>


              {objectLoading ? (

                <div className="detection-placeholder">

                  <ScanSearch size={38} />

                  <span>
                    AI is analyzing the image...
                  </span>

                </div>

              ) : objectResult ? (

                <img
                  src={`${API_SERVER}${objectResult.annotated_image}`}
                  alt="AI object detection result"
                />

              ) : (

                <div className="detection-placeholder">

                  <ScanSearch size={38} />

                  <span>
                    AI detection result will appear here.
                  </span>

                </div>

              )}

            </div>

          </div>


          {objectLoading && (

            <div className="detection-status">

              <ScanSearch size={18} />

              <span>
                AI is analyzing the image...
              </span>

            </div>

          )}


          {objectResult &&
            !objectLoading && (

              <div className="detection-status success">

                <CheckCircle2 size={18} />

                <span>
                  Analysis completed using{" "}
                  {objectResult.model}
                </span>

              </div>

            )}


          {objectError && (

            <div className="error">
              {objectError}
            </div>

          )}


          {/* RESULTS */}

          {objectResult && (

            <div className="object-results">

              <div className="section-title">

                <ScanSearch size={21} />

                <div>

                  <span className="label">
                    OBJECT INTELLIGENCE
                  </span>

                  <h2>
                    Detection Results
                  </h2>

                </div>

              </div>


              <div className="metrics">

                <div className="metric">

                  <span>
                    Total Detections
                  </span>

                  <strong>
                    {objectResult.total_detections}
                  </strong>

                </div>


                <div className="metric">

                  <span>
                    Model
                  </span>

                  <strong>
                    {objectResult.model}
                  </strong>

                </div>


                <div className="metric">

                  <span>
                    Detected Classes
                  </span>

                  <strong>
                    {Object.keys(
                      objectResult.counts || {}
                    ).length}
                  </strong>

                </div>


                <div className="metric">

                  <span>
                    Image Size
                  </span>

                  <strong>
                    {objectResult.width} ×{" "}
                    {objectResult.height}
                  </strong>

                </div>

              </div>


              <div className="detection-list">

                <div className="section-subtitle">

                  <span className="label">
                    {objectModel === "aerial"
                      ? "DETECTED GEOGRAPHIC FEATURES"
                      : "DETECTED OBJECTS"}
                  </span>

                </div>


                {objectResult.detections?.length > 0 ? (

                  objectResult.detections.map(
                    (detection, index) => (

                      <div
                        className="detection-item"
                        key={index}
                      >

                        <div>

                          <strong>
                            {detection.class}
                          </strong>

                          <span>
                            Confidence
                          </span>

                        </div>

                        <strong>
                          {formatConfidence(
                            detection.confidence
                          )}
                        </strong>

                      </div>

                    )
                  )

                ) : (

                  <div className="detection-placeholder">

                    <span>
                      {objectModel === "aerial"
                        ? "No supported geographic features were detected."
                        : "No supported objects were detected."}
                    </span>

                  </div>

                )}

              </div>


              <div className="baseline-warning">

                <strong>
                  Model Capability
                </strong>

                <p>

                  {objectModel === "aerial"

                    ? "YOLO26n-OBB detects supported aerial and geospatial classes. Building detection is not included in this model."

                    : "YOLO11n provides general object detection using supported COCO categories. Results depend on image quality and model coverage."}

                </p>

              </div>

            </div>

          )}

        </section>


        {/* =================================================
            AI ASSISTANT
            ================================================= */}

        <section className="assistant">

          <div className="section-title">

            <Bot size={21} />

            <div>

              <span className="label">
                AI ASSISTANT
              </span>

              <h2>
                Ask EarthVision
              </h2>

            </div>

          </div>


          <p>
            Ask questions about the imagery and
            the analysis results currently available
            in this session.
          </p>


          {/* SUGGESTED QUESTIONS */}

          <div className="suggested-questions">

            {[
              "What objects were detected?",
              "What potential differences were identified?",
              "Which regions have high severity?",
              "Explain the analysis method.",
            ].map((text) => (

              <button
                key={text}
                className="suggested-question"
                onClick={() =>
                  useSuggestedQuestion(text)
                }
              >
                {text}
              </button>

            ))}

          </div>


          <div className="assistant-input">

            <input
              type="text"
              value={question}
              onChange={(e) =>
                setQuestion(e.target.value)
              }
              onKeyDown={(e) => {

                if (e.key === "Enter") {
                  askAssistant();
                }

              }}
              placeholder="Ask: What did EarthVision detect?"
            />

            <button
              onClick={askAssistant}
              disabled={assistantLoading}
            >

              {assistantLoading
                ? "Thinking..."
                : "Ask AI"}

            </button>

          </div>


          {assistantError && (

            <div className="error">
              {assistantError}
            </div>

          )}


          {assistantAnswer && (

            <div className="assistant-answer">

              <div className="assistant-answer-header">

                <Bot size={18} />

                <strong>
                  EarthVision AI
                </strong>

              </div>

              <p>
                {assistantAnswer}
              </p>

            </div>

          )}

        </section>

      </main>

    </div>
  );
}


export default App;