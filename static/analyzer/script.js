
const micBtn = document.getElementById('mic-btn');
const recordingStatus = document.getElementById('recording-status');
const form = document.getElementById('voice-form');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('result');
const audioPreview = document.getElementById('audio-preview');

let mediaRecorder;
let audioChunks = [];
let audioBlob = null;
let isRecording = false;

// Microphone recording setup
micBtn.addEventListener('click', async () => {
  if (!isRecording) {
    // Start recording
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) audioChunks.push(event.data);
    };

    mediaRecorder.onstart = () => {
      isRecording = true;
      micBtn.classList.add('recording');
      recordingStatus.textContent = 'Recording...';
      if (audioPreview) audioPreview.src = '';
    };

    mediaRecorder.onstop = () => {
      isRecording = false;
      micBtn.classList.remove('recording');
      recordingStatus.textContent = '';
      audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      // Optional: Preview the recorded audio
      if (audioPreview) {
        audioPreview.src = URL.createObjectURL(audioBlob);
        audioPreview.classList.remove('hidden');
      }
    };

    mediaRecorder.start();
  } else {
    // Stop recording
    mediaRecorder.stop();
  }
});

// Form submission: send .wav to backend and handle CSV response
form.addEventListener('submit', async function (e) {
  e.preventDefault();
  if (!audioBlob) {
    alert('Please record your voice input first.');
    return;
  }
  form.classList.add('hidden');
  loading.classList.remove('hidden');

  // Prepare form data
  const formData = new FormData();
  formData.append('audio', audioBlob, 'input.wav');

  try {
    // Send audio to backend and expect CSV file in response
    const response = await fetch('/analyze', { // <-- Set your backend endpoint here
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Failed to get analysis from backend.');

    // Get CSV text
    const csvText = await response.text();

    // Parse CSV (using PapaParse for robustness)
    Papa.parse(csvText, {
      header: true,
      complete: function(results) {
        loading.classList.add('hidden');
        showCSVResults(results.data);
      }
    });

  } catch (error) {
    loading.classList.add('hidden');
    resultSection.innerHTML = `<div class="result-card"><p style="color:red;">Error: ${error.message}</p></div>`;
    resultSection.classList.remove('hidden');
  }
});

function showCSVResults(dataRows) {
  if (!dataRows || dataRows.length === 0) {
    resultSection.innerHTML = `<div class="result-card"><p>No results found in CSV.</p></div>`;
    resultSection.classList.remove('hidden');
    return;
  }

  // Create a table from CSV data
  let table = '<div class="result-card"><h2>Analysis Report</h2><table><thead><tr>';
  for (const key of Object.keys(dataRows[0])) {
    table += `<th>${key}</th>`;
  }
  table += '</tr></thead><tbody>';
  for (const row of dataRows) {
    table += '<tr>';
    for (const val of Object.values(row)) {
      table += `<td>${val}</td>`;
    }
    table += '</tr>';
  }
  table += '</tbody></table></div>';

  resultSection.innerHTML = table;
  resultSection.classList.remove('hidden');
}


