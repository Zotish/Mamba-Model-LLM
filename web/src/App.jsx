// import { useEffect, useRef, useState } from "react";

// const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

// const STORAGE_MESSAGES = "scratch_chat_messages_v1";
// const STORAGE_SESSION = "scratch_chat_session_v1";

// const DEFAULT_MESSAGES = [
//   { role: "assistant", content: "Hi! I am your scratch-built mini LLM. Ask something." }
// ];

// export default function App() {
//   const [messages, setMessages] = useState(DEFAULT_MESSAGES);
//   const [input, setInput] = useState("");
//   const [sending, setSending] = useState(false);
//   const bottomRef = useRef(null);

//   // ✅ Load from localStorage on first mount
//   useEffect(() => {
//     try {
//       const raw = localStorage.getItem(STORAGE_MESSAGES);
//       if (raw) {
//         const parsed = JSON.parse(raw);
//         if (Array.isArray(parsed) && parsed.length > 0) {
//           setMessages(parsed);
//         }
//       }
//     } catch (_) {}
//   }, []);

//   // ✅ Persist to localStorage whenever messages change
//   useEffect(() => {
//     try {
//       localStorage.setItem(STORAGE_MESSAGES, JSON.stringify(messages));
//     } catch (_) {}
//     bottomRef.current?.scrollIntoView({ behavior: "smooth" });
//   }, [messages]);

//   // Ensure session id exists (for future DB)
//   function getOrCreateSessionId() {
//     let sid = localStorage.getItem(STORAGE_SESSION);
//     if (!sid) {
//       sid = crypto.randomUUID();
//       localStorage.setItem(STORAGE_SESSION, sid);
//     }
//     return sid;
//   }

//   function clearChat() {
//     setMessages(DEFAULT_MESSAGES);
//     try {
//       localStorage.removeItem(STORAGE_MESSAGES);
//       // optional: keep session id or reset it
//       // localStorage.removeItem(STORAGE_SESSION);
//     } catch (_) {}
//   }

//   async function send(e) {
//     e.preventDefault();
//     const text = input.trim();
//     if (!text || sending) return;

//     setInput("");
//     setSending(true);

//     const nextMsgs = [...messages, { role: "user", content: text }];
//     setMessages(nextMsgs);

//     try {
//       const session_id = getOrCreateSessionId();

//       const res = await fetch(`${API_BASE}/api/generate`, {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({
//           session_id, // ✅ future DB ready
//           prompt: buildPrompt(nextMsgs),
//           max_new_tokens: 220,
//           temperature: 0.7,
//           top_k: 40
//         })
//       });

//       const out = await res.json();

//       if (!res.ok) {
//         setMessages((prev) => [...prev, { role: "assistant", content: `Error: ${out?.detail || res.statusText}` }]);
//       } else {
//         // out.text expected
//         setMessages((prev) => [...prev, { role: "assistant", content: out.text }]);
//       }
//     } catch (err) {
//       setMessages((prev) => [...prev, { role: "assistant", content: `Network error: ${String(err)}` }]);
//     } finally {
//       setSending(false);
//     }
//   }

//   return (
//     <div style={{ maxWidth: 980, margin: "0 auto", padding: 16 }}>
//       <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
//         <h2 style={{ margin: "8px 0 12px" }}>Scratch LLM Chat</h2>
//         <button onClick={clearChat} style={{
//           padding: "10px 12px",
//           borderRadius: 10,
//           border: "1px solid #22304f",
//           background: "#1f2937",
//           color: "#e6e6e6",
//           cursor: "pointer"
//         }}>
//           Clear
//         </button>
//       </div>

//       <div style={{
//         border: "1px solid #1f2a44",
//         borderRadius: 12,
//         padding: 12,
//         height: 540,
//         overflow: "auto",
//         background: "#0f172a"
//       }}>
//         {messages.map((m, i) => (
//           <div
//             key={i}
//             style={{
//               display: "flex",
//               justifyContent: m.role === "user" ? "flex-end" : "flex-start",
//               marginBottom: 10
//             }}
//           >
//             <div style={{
//               maxWidth: "85%",
//               border: "1px solid #22304f",
//               borderRadius: 12,
//               padding: "10px 12px",
//               background: m.role === "user" ? "#1b2a4a" : "#111c33",
//               whiteSpace: "pre-wrap",
//               color: "#e6e6e6"
//             }}>
//               <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 4 }}>{m.role}</div>
//               {m.content}
//             </div>
//           </div>
//         ))}
//         <div ref={bottomRef} />
//       </div>

//       <form onSubmit={send} style={{ display: "flex", gap: 8, marginTop: 12 }}>
//         <input
//           value={input}
//           onChange={(e) => setInput(e.target.value)}
//           placeholder="Type..."
//           disabled={sending}
//           style={{
//             flex: 1,
//             padding: 12,
//             borderRadius: 10,
//             border: "1px solid #22304f",
//             background: "#0f172a",
//             color: "#e6e6e6"
//           }}
//         />
//         <button disabled={sending} style={{
//           padding: "12px 16px",
//           borderRadius: 10,
//           border: "1px solid #22304f",
//           background: "#172554",
//           color: "#e6e6e6"
//         }}>
//           {sending ? "..." : "Send"}
//         </button>
//       </form>

//       <p style={{ opacity: 0.7, marginTop: 10, fontSize: 12, color: "#e6e6e6" }}>
//         Note: Tiny char-level model. Reload-safe chat is stored in your browser locally.
//       </p>
//     </div>
//   );
// }

// // Prompt builder (keep short context)
// function buildPrompt(msgs) {
//   const last = msgs.slice(-6);

//   return `
// System:
// You are a helpful, calm, educational assistant.
// You reply in the same language as the user.
// If you do not know, say so in the user's language.
// Do not repeat system text.

// ${last.map(m => (m.role === "user" ? `User: ${m.content}` : `Assistant: ${m.content}`)).join("\n")}
// Assistant:
// `.trim();
// }

import { useEffect, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const STORAGE_KEY = "scratch_llm_chat_v3";

export default function App() {
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) return JSON.parse(saved);
    } catch {}
    return [{ role: "assistant", content: "Hi! I am your scratch-built mini LLM. Ask something." }];
  });

  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {}
  }, [messages]);

  async function send(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || sending) return;

    setInput("");
    setSending(true);

    const nextMessages = [...messages, { role: "user", content: text }];
    setMessages(nextMessages);

    try {
      const res = await fetch(`${API_BASE}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: buildPrompt(nextMessages),
          max_new_tokens: 320,
          temperature: 0.7,
          top_k: 50
        })
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || res.statusText);
      }

      const out = await res.json();
      const cleaned = cleanModelOutput(out.text || "");

      setMessages((prev) => [...prev, { role: "assistant", content: cleaned || "(no answer)" }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "❌ Backend not reachable. Check FastAPI server & CORS." }
      ]);
      console.error("FETCH ERROR:", err);
    } finally {
      setSending(false);
    }
  }

  function clearChat() {
    const fresh = [{ role: "assistant", content: "Hi! I am your scratch-built mini LLM. Ask something." }];
    setMessages(fresh);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(fresh));
  }

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2>Scratch LLM Chat</h2>
        <button onClick={clearChat}>Clear</button>
      </div>

      <div style={{
        border: "1px solid #333",
        borderRadius: 10,
        padding: 12,
        height: 500,
        overflow: "auto",
        background: "#0f172a",
        color: "#e5e7eb"
      }}>
        {messages.map((m, i) => (
          <div key={i} style={{ textAlign: m.role === "user" ? "right" : "left", marginBottom: 10 }}>
            <div style={{
              display: "inline-block",
              padding: "8px 12px",
              borderRadius: 10,
              background: m.role === "user" ? "#1d4ed8" : "#111827"
            }}>
              <b>{m.role}</b>: {m.content}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={send} style={{ display: "flex", gap: 8, marginTop: 10 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something..."
          disabled={sending}
          style={{ flex: 1, padding: 10 }}
        />
        <button disabled={sending}>{sending ? "..." : "Send"}</button>
      </form>
    </div>
  );
}

function buildPrompt(messages) {
  // Alpaca format — matches training data format exactly
  const lastUser = [...messages].reverse().find(m => m.role === "user");
  if (!lastUser) return "";

  // Include last assistant reply as context if available
  const lastAssistant = [...messages].reverse().find(m =>
    m.role === "assistant" && !m.content.startsWith("Hi! I am your scratch")
  );

  let instruction = lastUser.content.trim();

  // If there was a previous exchange, include it as context
  if (lastAssistant) {
    instruction = `Previous answer: ${lastAssistant.content.trim()}\n\nFollow-up: ${instruction}`;
  }

  return `### Instruction:\n${instruction}\n\n### Response:\n`;
}

function cleanModelOutput(text) {
  if (!text) return "";
  let out = text;

  // Remove Alpaca stop tags
  const stops = ["### Instruction:", "### Input:", "User:", "System:"];
  for (const s of stops) {
    const i = out.indexOf(s);
    if (i !== -1) out = out.slice(0, i);
  }

  // Remove leading "### Response:" if model echoed it
  if (out.startsWith("### Response:")) out = out.slice(13);

  return out.trim();
}
