import { Bot, User } from './Icons'
import ThinkingAccordion from './ThinkingAccordion'
import Markdown from './Markdown';

export default function MessageBubble({
  role, content, thinking
}: { role: 'user' | 'assistant', content: string, thinking?: string }) {
  const isUser = role === 'user';
  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'} my-2 w-full`}>
      {!isUser && <div className="flex-shrink-0 w-9 h-9 rounded-full bg-primary/20 flex items-center justify-center"><Bot size={18}/></div>}

      <div className={`${isUser ? 'max-w-[50%]' : 'w-full'}`}>
        {!isUser && <div className="w-full"><ThinkingAccordion content={thinking || ''} /></div>}
        <div className={`${isUser ? 'bg-muted px-3 py-1 rounded-lg whitespace-pre-wrap' : ''}`}>
          {isUser ? content : <Markdown source={content || (isUser ? '' : '…')} />}
        </div>
      </div>
      {isUser && <div className="flex-shrink-0 w-9 h-9 rounded-full bg-accent/20 flex items-center justify-center"><User size={18}/></div>}
    </div>
  );
}
