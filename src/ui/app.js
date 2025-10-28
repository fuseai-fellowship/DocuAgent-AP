const uploadBtn = document.getElementById("uploadBtn");
const fileinput = document.getElementById("fileinput");
const uploadStatus = document.getElementById("uploadStatus");
const filenameEl = document.getElementById("filename");
const analyzeBtn = document.getElementById("analyzeBtn");
const summaryBox = document.getElementById("summaryBox");
const chatSection = document.getElementById("chatSection");
const chatLog = document.getElementById("chatLog");
const userMsg = document.getElementById("userMsg");
const sendBtn = document.getElementById("sendBtn");

let currentFilename = null;

uploadBtn.addEventListener("click", async () => {
  const file = fileinput.files[0];
  if (!file) return alert("Choose a file first");
  const fd = new FormData();
  fd.append("file", file);
  uploadStatus.textContent = "Uploading...";
  const res = await fetch("/upload", {method:"POST", body:fd});
  const data = await res.json();
  uploadStatus.textContent = "Uploaded: " + data.filename;
  currentFilename = data.filename;
  filenameEl.textContent = currentFilename;
  analyzeBtn.disabled = false;
});

analyzeBtn.addEventListener("click", async () => {
  if (!currentFilename) return;
  analyzeBtn.disabled = true;
  summaryBox.classList.remove("hidden");
  summaryBox.textContent = "Analyzing...";
  const fd = new FormData();
  fd.append("filename", currentFilename);
  const res = await fetch("/analyze", {method:"POST", body:fd});
  const data = await res.json();
  summaryBox.textContent = data.summary || "No summary";
  chatSection.classList.remove("hidden");
});

sendBtn.addEventListener("click", async () => {
  const msg = userMsg.value.trim();
  if (!msg || !currentFilename) return;
  appendChat("You", msg, "user");
  userMsg.value = "";
  const fd = new FormData();
  fd.append("filename", currentFilename);
  fd.append("message", msg);
  appendChat("Bot", "thinking...", "bot", true);
  const res = await fetch("/chat", {method:"POST", body:fd});
  const data = await res.json();
  replaceLastBot(data.answer || "No answer");
});

function appendChat(name, text, cls, isTemp=false){
  const el = document.createElement("div");
  el.className = cls;
  el.innerHTML = `<strong>${name}:</strong> ${text}`;
  if(isTemp) el.dataset.temp = "1";
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
}
function replaceLastBot(text){
  const els = chatLog.querySelectorAll("[data-temp='1']");
  if(els.length){
    const el = els[els.length-1];
    el.innerHTML = `<strong>Bot:</strong> ${text}`;
    delete el.dataset.temp;
  } else appendChat("Bot", text, "bot");
  chatLog.scrollTop = chatLog.scrollHeight;
}