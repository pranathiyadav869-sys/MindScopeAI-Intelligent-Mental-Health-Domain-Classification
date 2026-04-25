
import torch
import json
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================
# CONFIG
# =====================
MODEL_PATH = "model/model.pt"
TOKENIZER_PATH = "model/tokenizer"
labels = ['anxiety', 'depression', 'ocd', 'trauma']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# LOAD MODEL
# =====================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# =====================
# FLASK APP
# =====================
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    scores = {labels[i]: round(float(probs[i]) * 100, 2) for i in range(len(labels))}
    return jsonify(scores)

@app.route("/")
def index():
    return HTML_PAGE

# =====================
# HTML UI
# =====================
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Mental Health Classifier</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@600;700&display=swap" rel="stylesheet"/>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    background:#0a0a0a;
    min-height:100vh;
    font-family:'DM Sans',sans-serif;
    color:#f0f0f0;
    padding:2.5rem 1rem 4rem;
  }
  .app{max-width:780px;margin:0 auto;}

  /* HEADER */
  .header{text-align:center;margin-bottom:2.5rem;}
  .brain-circle{
    width:64px;height:64px;border-radius:50%;
    background:#1e1e1e;border:1px solid #333;
    display:flex;align-items:center;justify-content:center;
    font-size:30px;margin:0 auto 1.2rem;
  }
  .header h1{
    font-family:'Syne',sans-serif;
    font-size:28px;font-weight:700;
    color:#e8e8e8;letter-spacing:-0.5px;margin-bottom:6px;
  }
  .header p{font-size:14px;color:#888;line-height:1.6;}
  .badge-row{display:flex;gap:8px;justify-content:center;margin-top:12px;flex-wrap:wrap;}
  .badge{
    font-size:11px;font-weight:500;padding:4px 12px;
    border-radius:20px;border:1px solid #333;
    color:#aaa;background:#141414;
  }

  /* CARDS */
  .card{
    background:#141414;border:1px solid #252525;
    border-radius:16px;padding:1.5rem;margin-bottom:1rem;
  }
  .card-label{
    font-size:11px;font-weight:600;letter-spacing:0.1em;
    text-transform:uppercase;color:#666;margin-bottom:12px;
  }

  /* INPUT */
  textarea{
    width:100%;min-height:115px;
    background:#0f0f0f;border:1px solid #2e2e2e;
    border-radius:10px;padding:14px;
    font-family:'DM Sans',sans-serif;font-size:15px;
    color:#e0e0e0;resize:vertical;outline:none;line-height:1.6;
    transition:border-color 0.2s;
  }
  textarea:focus{border-color:#555;}
  textarea::placeholder{color:#444;}
  .char-counter{text-align:right;font-size:12px;color:#555;margin-top:6px;}

  .examples-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;align-items:center;}
  .try-label{font-size:12px;color:#555;white-space:nowrap;}
  .chip{
    font-size:12px;padding:5px 13px;border-radius:20px;
    border:1px solid #2a2a2a;background:#181818;
    cursor:pointer;color:#888;transition:all 0.15s;
  }
  .chip:hover{border-color:#555;color:#ccc;}

  .analyze-btn{
    width:100%;margin-top:14px;padding:14px;
    border-radius:10px;border:1px solid #444;
    background:#e8e8e8;color:#0a0a0a;
    font-family:'DM Sans',sans-serif;font-size:15px;font-weight:600;
    cursor:pointer;transition:background 0.2s,transform 0.1s;letter-spacing:0.01em;
  }
  .analyze-btn:hover{background:#fff;}
  .analyze-btn:active{transform:scale(0.99);}
  .analyze-btn:disabled{background:#2a2a2a;color:#555;border-color:#2a2a2a;cursor:not-allowed;transform:none;}

  /* TOP PREDICTION */
  .top-card{
    background:#141414;border:1px solid #252525;
    border-radius:16px;padding:1.4rem 1.5rem;
    margin-bottom:1rem;display:flex;align-items:center;gap:14px;
  }
  .top-icon{
    width:50px;height:50px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;font-size:24px;flex-shrink:0;
  }
  .top-label{
    font-family:'Syne',sans-serif;font-size:22px;
    font-weight:700;text-transform:capitalize;color:#e8e8e8;margin-bottom:3px;
  }
  .top-sub{font-size:13px;color:#666;}
  .top-pct{
    margin-left:auto;font-family:'Syne',sans-serif;
    font-size:26px;font-weight:700;color:#e8e8e8;
  }

  /* BARS */
  .bars-grid{display:grid;gap:16px;}
  .bar-meta{display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;}
  .bar-name{font-size:13px;font-weight:500;text-transform:capitalize;color:#ccc;display:flex;align-items:center;gap:7px;}
  .bar-icon{font-size:14px;}
  .bar-pct{font-size:13px;font-weight:500;color:#666;}
  .bar-track{height:8px;background:#1e1e1e;border-radius:20px;overflow:hidden;}
  .bar-fill{height:100%;border-radius:20px;width:0%;transition:width 0.75s cubic-bezier(.4,0,.2,1);}

  /* DONUT */
  .donut-wrapper{display:flex;align-items:center;gap:2.5rem;flex-wrap:wrap;}
  .legend-list{flex:1;min-width:140px;display:flex;flex-direction:column;gap:11px;}
  .legend-item{display:flex;align-items:center;gap:9px;}
  .legend-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
  .legend-text{font-size:13px;color:#888;text-transform:capitalize;flex:1;}
  .legend-val{font-size:13px;font-weight:500;color:#ccc;}

  /* EMPTY */
  .empty{text-align:center;padding:2.5rem 1rem;color:#444;font-size:14px;line-height:1.8;}

  /* LOADER */
  .dots{display:flex;gap:7px;justify-content:center;padding:2rem;}
  .dot{width:8px;height:8px;border-radius:50%;background:#444;animation:bounce 1.2s infinite;}
  .dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
  @keyframes bounce{0%,80%,100%{transform:scale(0.6);opacity:0.3}40%{transform:scale(1);opacity:1}}

  /* ERROR */
  .error-msg{
    text-align:center;padding:1rem;color:#e24b4a;
    font-size:14px;background:#1a0f0f;border-radius:10px;margin-top:1rem;
    border:1px solid #3a1f1f;
  }
</style>
</head>
<body>
<div class="app">

  <div class="header">
    <div class="brain-circle">&#129504;</div>
    <h1>Mental Health Classifier</h1>
    <p>Enter a description to identify the most likely mental health domain.</p>
    <div class="badge-row">
      <span class="badge">Anxiety</span>
      <span class="badge">Depression</span>
      <span class="badge">OCD</span>
      <span class="badge">Trauma</span>
    </div>
  </div>

  <div class="card">
    <div class="card-label">Input text</div>
    <textarea id="inputText" placeholder="Describe symptoms or feelings, e.g. &quot;I feel constant worry about everyday situations and can't seem to relax...&quot;" oninput="updateCounter()"></textarea>
    <div class="char-counter"><span id="charCount">0</span> characters</div>
    <div class="examples-row">
      <span class="try-label">Try:</span>
      <button class="chip" onclick="fillExample(0)">Constant worry & panic</button>
      <button class="chip" onclick="fillExample(1)">Feeling hopeless & empty</button>
      <button class="chip" onclick="fillExample(2)">Repetitive intrusive thoughts</button>
      <button class="chip" onclick="fillExample(3)">Flashbacks & nightmares</button>
    </div>
    <button class="analyze-btn" id="analyzeBtn" onclick="runAnalysis()">Analyze text &#8599;</button>
  </div>

  <div id="resultsArea">
    <div class="empty">Enter text above and click <strong>Analyze text</strong> to see predictions.</div>
  </div>

</div>

<script>
const COLORS = {anxiety:'#378ADD',depression:'#D4537E',ocd:'#E09B2D',trauma:'#1D9E75'};
const ICONS  = {anxiety:'&#128161;',depression:'&#128566;',ocd:'&#129504;',trauma:'&#129979;'};
const BGCOL  = {anxiety:'#0d1a2a',depression:'#1e0d14',ocd:'#1e1500',trauma:'#071a12'};

const EXAMPLES = [
  "I feel constant worry and unease about everyday situations. My heart races and I can't seem to relax even when nothing bad is happening. I avoid social situations because I fear judgment.",
  "I've been feeling completely empty and hopeless for weeks. I've lost interest in things I used to enjoy, I sleep too much but still feel exhausted, and I can't see a way forward.",
  "I keep having intrusive, unwanted thoughts that I can't control. I feel compelled to repeat certain actions or check things multiple times to reduce the anxiety, even though I know it's excessive.",
  "I keep reliving a traumatic event through nightmares and flashbacks. Certain triggers take me right back there. I feel numb and disconnected, and I avoid anything that reminds me of what happened."
];

function fillExample(i){ document.getElementById('inputText').value=EXAMPLES[i]; updateCounter(); }
function updateCounter(){ document.getElementById('charCount').textContent=document.getElementById('inputText').value.length; }

async function runAnalysis(){
  const text = document.getElementById('inputText').value.trim();
  if(!text){ showError("Please enter some text first."); return; }

  const btn = document.getElementById('analyzeBtn');
  btn.disabled=true; btn.textContent='Analyzing...';
  document.getElementById('resultsArea').innerHTML='<div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';

  try {
    const res  = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    const data = await res.json();
    if(data.error){ showError(data.error); btn.disabled=false; btn.textContent='Analyze text \u2197'; return; }
    renderResults(data);
  } catch(e){
    showError("Could not reach the model server. Make sure Flask is running.");
  }
  btn.disabled=false; btn.textContent='Analyze text \u2197';
}

function showError(msg){
  document.getElementById('resultsArea').innerHTML=`<div class="error-msg">${msg}</div>`;
}

function renderResults(scores){
  const sorted = Object.entries(scores).sort((a,b)=>b[1]-a[1]);
  const [topLabel, topScore] = sorted[0];

  let html = `
  <div class="top-card">
    <div class="top-icon" style="background:${BGCOL[topLabel]}">${ICONS[topLabel]}</div>
    <div>
      <div class="top-label">${topLabel}</div>
      <div class="top-sub">Top predicted domain</div>
    </div>
    <div class="top-pct">${topScore.toFixed(1)}%</div>
  </div>

  <div class="card">
    <div class="card-label">Confidence breakdown</div>
    <div class="bars-grid">`;

  sorted.forEach(([label,score])=>{
    html+=`
    <div>
      <div class="bar-meta">
        <div class="bar-name"><span class="bar-icon">${ICONS[label]}</span>${label}</div>
        <div class="bar-pct">${score.toFixed(1)}%</div>
      </div>
      <div class="bar-track">
        <div class="bar-fill" id="bar-${label}" style="background:${COLORS[label]}"></div>
      </div>
    </div>`;
  });
  html+=`</div></div>`;

  html+=`
  <div class="card">
    <div class="card-label">Distribution chart</div>
    <div class="donut-wrapper">
      <svg id="donutSvg" width="150" height="150" viewBox="0 0 150 150"></svg>
      <div class="legend-list">`;
  sorted.forEach(([label,score])=>{
    html+=`
    <div class="legend-item">
      <div class="legend-dot" style="background:${COLORS[label]}"></div>
      <span class="legend-text">${label}</span>
      <span class="legend-val">${score.toFixed(1)}%</span>
    </div>`;
  });
  html+=`</div></div></div>`;

  document.getElementById('resultsArea').innerHTML=html;

  // animate bars
  sorted.forEach(([label,score])=>{
    const el=document.getElementById(`bar-${label}`);
    if(el) requestAnimationFrame(()=>el.style.width=score+'%');
  });

  // draw donut
  const svg=document.getElementById('donutSvg');
  const cx=75,cy=75,r=55,sw=22,circ=2*Math.PI*r;
  let offset=0, paths='';
  const total=sorted.reduce((s,[,v])=>s+v,0);
  sorted.forEach(([label,score])=>{
    const frac=score/total;
    const dash=(frac*circ).toFixed(2);
    const gap=(circ-frac*circ).toFixed(2);
    paths+=`<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${COLORS[label]}" stroke-width="${sw}" stroke-dasharray="${dash} ${gap}" stroke-dashoffset="${-(offset*circ).toFixed(2)}" transform="rotate(-90,${cx},${cy})"/>`;
    offset+=frac;
  });
  paths+=`<text x="${cx}" y="${cy-6}" text-anchor="middle" font-family="DM Sans" font-size="11" fill="#555">top</text>`;
  paths+=`<text x="${cx}" y="${cy+14}" text-anchor="middle" font-family="Syne,sans-serif" font-size="17" font-weight="700" fill="#e0e0e0">${sorted[0][0].slice(0,4).toUpperCase()}</text>`;
  svg.innerHTML=paths;
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    print(f"Device: {device}")
    print("Starting server → http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
