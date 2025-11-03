// Incrementally splits out <think>...</think> from a streaming buffer
export function extractThinking(acc: string): { thinking: string; visible: string } {
  const start = acc.indexOf('<think>');
  if (start === -1) return { thinking: '', visible: acc };
  const end = acc.indexOf('</think>', start + 7);
  if (end === -1) {
    return { thinking: acc.slice(start + 7), visible: acc.slice(0, start) };
  }
  const thinking = acc.slice(start + 7, end);
  const visible = acc.slice(0, start) + acc.slice(end + 8);
  return { thinking, visible };
}
