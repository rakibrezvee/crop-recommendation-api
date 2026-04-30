const form = document.getElementById("prediction-form");
const submitBtn = document.getElementById("submit-btn");
const resultSection = document.getElementById("result");
const cropName = document.getElementById("crop-name");
const statusText = document.getElementById("status");
const errorSection = document.getElementById("error");
const debugDetails = document.getElementById("debug");
const debugContent = document.getElementById("debug-content");

const setHidden = (element, hidden) => {
  if (hidden) {
    element.classList.add("hidden");
  } else {
    element.classList.remove("hidden");
  }
};

const setLoading = (loading) => {
  submitBtn.disabled = loading;
  submitBtn.textContent = loading ? "Predicting..." : "Get recommendation";
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoading(true);
  setHidden(errorSection, true);
  setHidden(resultSection, true);
  setHidden(debugDetails, true);

  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    payload[key] = Number(value);
  }

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Request failed (${response.status})`);
    }

    const data = await response.json();
    cropName.textContent = data.recommended_crop ?? "Unknown";
    statusText.textContent = data.status
      ? `Status: ${data.status} • Accuracy: ${data.accuracy_used ?? "N/A"}`
      : "Prediction complete.";

    if (data.debug_info) {
      debugContent.textContent = JSON.stringify(data.debug_info, null, 2);
      setHidden(debugDetails, false);
    }

    setHidden(resultSection, false);
  } catch (error) {
    errorSection.textContent =
      error?.message ?? "Something went wrong while calling the API.";
    setHidden(errorSection, false);
  } finally {
    setLoading(false);
  }
});
