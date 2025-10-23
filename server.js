import express from "express";
import dotenv from "dotenv";
import OpenAI from "openai";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";
import multer from "multer";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

if (!process.env.OPENAI_API_KEY) {
  console.error("Missing OPENAI_API_KEY in .env");
  process.exit(1);
}

const app = express();
app.use(cors());
app.use(express.json());

// OpenAI
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const CHAT_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";

// Silence detection config
const SILENCE_MS = 3000;   // espera ~3s de silêncio antes de parar
const CHECK_EVERY = 150;   // checa menos freneticamente
const THRESHOLD = 0.02;    // mais tolerante ao volume baixo

// ===== Multer (memória) para uploads de áudio =====
const upload = multer({ storage: multer.memoryStorage() });

// Simple in-memory session store for conversation history
const sessions = new Map();
function genId() { return Math.random().toString(36).slice(2,10); }
function trimMessages(msgs, max = 20) {
  const system = msgs.length && msgs[0]?.role === "system" ? [msgs[0]] : [];
  const rest = msgs.slice(system.length);
  return system.concat(rest.slice(-max));
}

app.get("/health", (_req, res) => res.status(200).json({ ok: true }));

// ===== Prompt do professor =====
const SYSTEM_PROMPT = `
You are an Elementary English teacher AI that follows Fabiano Savella’s methodology.

GOALS
- Build fluency, confidence, and accuracy through natural conversation with elementary students.
- Keep English (UK) as the default language of the conversation.

STRICT LANGUAGE POLICY
- You may respond only in: English (UK), Portuguese (Brazil), or Italian.
- If the student writes in English → reply 100% in English (UK).

- If the student writes in Portuguese/Italian because they did NOT understand your last question, follow EXACTLY this 3-step sequence:

  (1) **Translate only your previous question** into the student’s language Prtuguese/Italian using NATURAL Portuguese if the student spoke portuguese or Italian if the student spoke italian — do NOT include any English words in this line.
      Format (no English inside the quotes):
      Portuguese example: A pergunta foi: "Como você está hoje?"
      Italian example: Traduzione: "Come stai oggi?"
      ✱ NUNCA escreva a pergunta em inglês nessa linha. É tradução REAL.

  (2) Immediately return to **ENGLISH (UK)** and repeat the **exact same question** (same meaning, no reformulation, no extra info).
      Example: How are you today?

  (3) Still in **ENGLISH (UK)**, add ONE short clarifying follow-up to make answering easier (e.g., a simple example or either/or).
      Example: Are you feeling good or a bit tired?

- If the student repeats Portuguese or italian again (“não entendi” / “non ho capito”), **repeat** the same 3 steps.
- Portuguese and Italian are ONLY for brief clarification. The conversation must ALWAYS return to ENGLISH (UK) after step (1).
- Never guess random languages; if the input is empty, unclear/gibberish, reply only in English asking to repeat in simple English.

CORRECTION POLICY (Savella Method)
- Be concise (3–6 short lines total).
- Correct only REAL mistakes (grammar, tense, word order, preposition, article, or vocabulary).
- Ignore punctuation/capitalisation mistakes if the meaning is clear.
- When correcting:
  (1) Provide a natural corrected model: “You could say (or something similar): …”
  (2) Explain WHY in one short sentence (in the same language the student is using at that moment).
- If the student’s sentence is already natural, do NOT correct — encourage and continue.

CONVERSATION FLOW
- Refer to what the student said and continue the same topic grading the language for elementary students.
- When the student talk about you aknowledge the comment and react as a human.
- ALWAYS end with ONE specific follow-up question in ENGLISH (UK), based on the student’s last message.
- If the student expresses confusion, apply the potuguese/italian clarification rule and then return to English.

TONE
- Warm, supportive, human, friendly.
- No long lectures. Short, clear communication that keeps the conversation moving.
`;

// ===== Rota de conversa (chat) =====
app.post("/ask", async (req, res) => {
  try {
    const userText = (req.body?.message || "").trim();
    if (!userText) return res.status(400).json({ error: "Empty message" });

    // sessionId pode vir no body ou query; se não houver, criamos novo
    let sessionId = req.body?.sessionId || req.query?.sessionId;
    let messages;
    if (sessionId && sessions.has(sessionId)) {
      messages = sessions.get(sessionId);
    } else {
      sessionId = sessionId || genId();
      messages = [{ role: "system", content: SYSTEM_PROMPT }];
    }

    // anexa a mensagem do usuário e limita histórico
    messages.push({ role: "user", content: userText });
    messages = trimMessages(messages, 20);
    sessions.set(sessionId, messages);

    const chatResponse = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages,
      temperature: 0.8,
      max_tokens: 500
    });

    const answer = chatResponse.choices?.[0]?.message?.content?.trim() || "";
    // armazena resposta no histórico
    messages.push({ role: "assistant", content: answer });
    messages = trimMessages(messages, 20);
    sessions.set(sessionId, messages);

    // retorna resposta e sessionId (cliente deve guardá-lo)
    res.json({ answer, sessionId });
  } catch (err) {
    console.error("OpenAI error:", err);
    const status = err?.response?.status || 500;
    const detail = err?.response?.data?.error?.message || err?.message || "Server error";
    res.status(status).json({ error: detail });
  }
});

// ===== Rota de STT (Whisper) =====
app.post("/stt", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file || !req.file.buffer) {
      return res.status(400).json({ error: "No audio uploaded" });
    }

    // MIME types suportados pelos endpoints de transcrição (OpenAI)
    // Referências comuns: mp3/mpeg, mp4, m4a, wav, webm, ogg, mpga
    const allowedMimes = new Set([
      "audio/mpeg", "audio/mp3", "audio/mpga", // mp3
      "audio/wav",                             // wav
      "audio/webm",                            // webm/opus
      "audio/ogg",                             // ogg/opus
      "audio/mp4", "audio/m4a"                 // mp4/m4a (aac/alac)
    ]);

    console.log("Uploaded file type:", req.file.mimetype);

    if (!allowedMimes.has(req.file.mimetype)) {
      return res.status(400).json({ error: "Unsupported file format" });
    }

    // Ajusta extensão do tmp conforme mimetype
    const extByMime = {
      "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/mpga": ".mp3",
      "audio/wav": ".wav",
      "audio/webm": ".webm",
      "audio/ogg": ".ogg",
      "audio/mp4": ".m4a", "audio/m4a": ".m4a"
    };
    const ext = extByMime[req.file.mimetype] || "";
    const tmpPath = path.join(__dirname, "tmp-upload" + ext);

    fs.writeFileSync(tmpPath, req.file.buffer);

    const tr = await openai.audio.transcriptions.create({
      file: fs.createReadStream(tmpPath),
      model: "gpt-4o-transcribe"
    });

    fs.unlink(tmpPath, () => {});
    res.json({ text: tr.text || "" });
  } catch (err) {
    console.error("Whisper error:", err?.response?.data || err);
    const status = err?.response?.status || 500;
    const detail = err?.response?.data?.error?.message || err?.message || "STT error";
    res.status(status).json({ error: detail });
  }
});

// ===== Rota de TTS (Neural) =====
app.get("/tts", async (req, res) => {
  try {
    const text = (req.query.text || "").toString().slice(0, 1200);
    if (!text) return res.status(400).json({ error: "Missing ?text=" });

    const allowedVoices = ["nova","shimmer","echo","onyx","fable","alloy","ash","sage","coral"];
    let voice = (process.env.OPENAI_TTS_VOICE || "alloy").toLowerCase();
    if (!allowedVoices.includes(voice)) voice = "alloy";

    const tts = await openai.audio.speech.create({
      model: "tts-1",
      voice,
      input: text,
      format: "mp3"
    });

    const buf = Buffer.from(await tts.arrayBuffer());
    res.setHeader("Content-Type", "audio/mpeg");
    res.setHeader("Cache-Control", "no-store, max-age=0");
    res.send(buf);
  } catch (err) {
    console.error("TTS error:", err?.response?.data || err);
    const status = err?.response?.status || 500;
    const detail = err?.response?.data?.error?.message || err?.message || "TTS error";
    res.status(status).json({ error: detail });
  }
});

// ===== Frontend estático =====
app.use(express.static(path.join(__dirname, "public")));
app.get("/", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// ===== Start =====
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log("Conversation bot running on http://localhost:" + PORT);
});
