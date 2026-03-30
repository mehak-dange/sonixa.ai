document.getElementById("processBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("audioFile");
  const status = document.getElementById("status");
  const transcription = document.getElementById("transcription");
  const report = document.getElementById("report");

  if (!fileInput.files[0]) {
    status.textContent = "Please select an audio file first.";
    return;
  }

  const file = fileInput.files[0];

  // create form data
  const formData = new FormData();
  formData.append("file", file);

  try {
    status.textContent = "Processing audio... please wait ⏳";

    const response = await fetch("http://127.0.0.1:8000/process-audio", {
      method: "POST",
      body: formData,
    });


   const data = await response.json();
console.log("Backend response:", data);

// show raw response for testing
transcription.value = JSON.stringify(data, null, 2);
report.value = JSON.stringify(data, null, 2);
    status.textContent = "Processing complete ✅";

  } catch (error) {
    console.error(error);
    status.textContent = "Error connecting to backend ❌";
  }
});