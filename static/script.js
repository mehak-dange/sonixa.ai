document.getElementById("processBtn").addEventListener("click", async () => {
  console.log("Process button clicked");

  const fileInput = document.getElementById("audioFile");
  const status = document.getElementById("status");
  const transcription = document.getElementById("transcription");
  const report = document.getElementById("report");

  if (!fileInput.files[0]) {
    status.textContent = "Please select an audio file first.";
    console.log("No file selected");
    return;
  }

  const file = fileInput.files[0];
  console.log("Selected file:", file.name);

  const formData = new FormData();
  formData.append("file", file);

  try {
    status.textContent = "Processing audio...";

    console.log("Sending request to backend...");

    const response = await fetch("http://127.0.0.1:8000/process-audio", {
      method: "POST",
      body: formData,
    });

    console.log("Response received:", response);

    const data = await response.json();
    console.log("Backend data:", data);

    transcription.value = data.transcription || "No transcription found";
    report.value = JSON.stringify(data.report, null, 2);

    status.textContent = "Processing complete ✅";
  } catch (error) {
    console.error("Fetch error:", error);
    status.textContent = "Error connecting to backend ❌";
  }
});