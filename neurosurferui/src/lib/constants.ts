import { json } from "stream/consumers";

export const FOLLOWUP_QUESTIONS = [
    "What if time is not linear but a dimension we can navigate?",
    "How might ancient civilizations have understood the universe differently?",
    "Could AI one day create art that feels more human than humans?",
    "What if consciousness is not a byproduct of matter but a fundamental aspect of reality?",
    "How might we create a society that values and nurtures creativity and innovation?",
    "What if there is no objective reality, and everything is subjective?",
    "How might we create a world where everyone has access to education and resources?",
    "What if we could communicate with extraterrestrial life?",
    "How might we create a world where everyone has access to healthcare?",
    "What if we could cure all diseases?",
    "How might we create a world where everyone has access to clean water?",
    "What if we could eliminate poverty?",
    "How might we create a world where everyone has access to food?",
    "What if we could eliminate hunger?",
    "How might we create a world where everyone has access to shelter?",
    "What if we could eliminate homelessness?",
    "How might we create a world where everyone has access to transportation?",
    "What if we could eliminate traffic?",
    "How might we create a world where everyone has access to energy?",
    "What if we could eliminate energy poverty?",
    "How might we create a world where everyone has access to water?",
    "What if we could eliminate water scarcity?",
    "How might we create a world where everyone has access to healthcare?",
    "What if we could cure all diseases?",
]

export const getRandomFollowUps = (count: number = 3): string[] => {
  const shuffled = [...FOLLOWUP_QUESTIONS].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, count);
};

export const TITLE_SYSTEM_PROMPT = `
You are a terse title generator for chat transcripts.

Rules:
- Output JSON only: {"title":"..."}
- 3-7 words, descriptive, no emojis.
- Do NOT include quotes, brackets, or markdown fences in the value.
- No trailing punctuation. No code.
- Use the existing messages as context. Summarize the main topic.

Examples:
User+Assistant discuss fixing a React event JSON error -> {"title":"Fix React Circular JSON Error"}
Building IPTV proxy with FastAPI and HLS -> {"title":"FastAPI HLS Proxy Setup"}
`