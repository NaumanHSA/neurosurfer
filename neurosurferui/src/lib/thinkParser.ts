// reasoningParser.ts

export type ReasoningMode = 'normal' | 'in_think' | 'in_final';

export interface ReasoningParseState {
  mode: ReasoningMode;
  buffer: string;       // carry partial tags between chunks
  thinking: string;     // between <__think__>...</__think__>
  finalAnswer: string;  // between <__final_answer__>...</__final_answer__>
  visible: string;      // what the user sees
  hasFinalTag: boolean;
  done: boolean;        // </__final_answer__> seen
}

export function createReasoningState(): ReasoningParseState {
  return {
    mode: 'normal',
    buffer: '',
    thinking: '',
    finalAnswer: '',
    visible: '',
    hasFinalTag: false,
    done: false,
  };
}

const OPEN_THINK = '<__think__>';
const CLOSE_THINK = '</__think__>';
const OPEN_FINAL = '<__final_answer__>';
const CLOSE_FINAL = '</__final_answer__>';

export function consumeReasoningDelta(
  prev: ReasoningParseState,
  delta: string
): ReasoningParseState {
  let state: ReasoningParseState = { ...prev };

  state.buffer += delta;
  const buf = state.buffer;
  const n = buf.length;
  let i = 0;

  while (i < n) {
    const ch = buf[i];

    if (ch === '<') {
      const closeIdx = buf.indexOf('>', i + 1);
      if (closeIdx === -1) {
        // Partial tag; keep in buffer for next chunk
        break;
      }

      const tag = buf.slice(i, closeIdx + 1);

      if (tag === OPEN_THINK) {
        state.mode = 'in_think';
        i = closeIdx + 1;
        continue;
      }
      if (tag === CLOSE_THINK) {
        state.mode = 'normal';
        i = closeIdx + 1;
        continue;
      }
      if (tag === OPEN_FINAL) {
        state.mode = 'in_final';
        state.hasFinalTag = true;
        i = closeIdx + 1;
        continue;
      }
      if (tag === CLOSE_FINAL) {
        state.mode = 'normal';
        state.done = true;
        i = closeIdx + 1;
        continue;
      }

      // Unknown <...> → treat as literal '<'
      dispatchChar(state, '<');
      i += 1;
      continue;
    }

    // Normal character
    dispatchChar(state, ch);
    i += 1;
  }

  // Keep unprocessed tail (e.g. "<__thi")
  state.buffer = buf.slice(i);
  return state;
}

function dispatchChar(state: ReasoningParseState, ch: string) {
  switch (state.mode) {
    case 'in_think':
      state.thinking += ch;
      break;

    case 'in_final':
      state.finalAnswer += ch;
      state.visible += ch;
      break;

    case 'normal':
    default:
      if (!state.hasFinalTag) {
        // No <__final_answer__> seen → treat as visible and finalAnswer
        state.visible += ch;
        state.finalAnswer += ch;
      } else {
        // After we saw <__final_answer__>, but outside tags – just show it
        state.visible += ch;
      }
      break;
  }
}
