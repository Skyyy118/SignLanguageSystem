// ==========================
// MODE CONTROLS
// ==========================

function startAlphabet(){

fetch("/alphabet")

document.getElementById("alphabetBtn").classList.add("active")
document.getElementById("wordBtn").classList.remove("active")

document.getElementById("alphabetBtn").disabled = true
document.getElementById("wordBtn").disabled = false

document.getElementById("modeIndicator").innerText =
"Mode: Alphabet Detection"

}


function startWords(){

fetch("/words")

document.getElementById("wordBtn").classList.add("active")
document.getElementById("alphabetBtn").classList.remove("active")

document.getElementById("wordBtn").disabled = true
document.getElementById("alphabetBtn").disabled = false

document.getElementById("modeIndicator").innerText =
"Mode: Word Detection"

}


function stopAll(){

fetch("/stop")

document.getElementById("alphabetBtn").classList.remove("active")
document.getElementById("wordBtn").classList.remove("active")

document.getElementById("alphabetBtn").disabled = false
document.getElementById("wordBtn").disabled = false

document.getElementById("modeIndicator").innerText =
"Mode: Stopped"

document.getElementById("translatedText").innerText =
"Waiting..."

}



// ==========================
// TRANSLATION UPDATE
// ==========================

function updateTranslation(){

fetch("/get_translation")
.then(res => res.json())
.then(data => {

let text = data.text || "Waiting..."

document.getElementById("translatedText").innerText = text

let camera = document.querySelector(".camera-feed")

if(text !== "Waiting..."){

camera.classList.add("active-detection")

}else{

camera.classList.remove("active-detection")

}

})

}



// ==========================
// SENTENCE BUILDER
// (WITH CORRECT SPACING)
// ==========================

let lastSentence = ""

function updateSentence(){

fetch("/get_sentence")
.then(res => res.json())
.then(data => {

let sentence = data.sentence || ""

if(sentence !== lastSentence){

let words = sentence.trim().split(/\s+/)

let container = document.getElementById("sentenceText")

container.innerHTML = ""

words.forEach((word,index)=>{

let span = document.createElement("span")

span.className = "word-animate"

span.textContent = word

container.appendChild(span)

// ADD REAL SPACE BETWEEN WORDS
if(index < words.length - 1){

let space = document.createTextNode(" ")

container.appendChild(space)

}

})

lastSentence = sentence

}

})

}



// ==========================
// CLEAR SENTENCE
// ==========================

function clearSentence(){

fetch("/clear_sentence")

document.getElementById("sentenceText").innerText =
"Start signing..."

}



// ==========================
// SPEECH
// ==========================

function speakSentence(){

fetch("/speak")

}



// ==========================
// AUTO REFRESH
// ==========================

setInterval(updateTranslation,500)
setInterval(updateSentence,700)