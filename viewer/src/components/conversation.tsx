"use client";

import { useState } from "react";
import type { ConversationMessage } from "@/lib/types";

function parseAssistantContent(content: string) {
  const parts: { type: "text" | "think" | "tool_call"; content: string; toolName?: string }[] = [];
  let remaining = content;

  while (remaining.length > 0) {
    const thinkStart = remaining.indexOf("<think>");
    const actionMatch = remaining.match(/^Action:\s*(\{[\s\S]*?\})\s*$/m);

    if (thinkStart === 0) {
      const thinkEnd = remaining.indexOf("</think>");
      if (thinkEnd !== -1) {
        parts.push({ type: "think", content: remaining.slice(7, thinkEnd).trim() });
        remaining = remaining.slice(thinkEnd + 8);
        continue;
      }
    }

    if (thinkStart > 0) {
      parts.push({ type: "text", content: remaining.slice(0, thinkStart) });
      remaining = remaining.slice(thinkStart);
      continue;
    }

    if (actionMatch) {
      const idx = remaining.indexOf(actionMatch[0]);
      if (idx > 0) {
        parts.push({ type: "text", content: remaining.slice(0, idx) });
      }
      let toolName = "tool";
      try {
        const parsed = JSON.parse(actionMatch[1]);
        if (parsed.action) toolName = parsed.action;
        if (parsed.tool) toolName = parsed.tool;
        if (parsed.name) toolName = parsed.name;
      } catch {}
      parts.push({ type: "tool_call", content: actionMatch[1], toolName });
      remaining = remaining.slice(idx + actionMatch[0].length);
      continue;
    }

    parts.push({ type: "text", content: remaining });
    break;
  }

  return parts.length ? parts : [{ type: "text" as const, content }];
}

function CollapsibleBlock({
  label,
  children,
  defaultOpen = false,
}: {
  label: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="text-xs text-text-tertiary hover:text-text-secondary transition-colors"
      >
        {open ? "Hide" : "Show"} {label}
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  );
}

function ToolResponseContent({ content }: { content: string }) {
  const lines = content.split("\n");
  const [expanded, setExpanded] = useState(false);

  if (lines.length <= 15) {
    return <pre className="whitespace-pre-wrap text-xs font-mono">{content}</pre>;
  }

  return (
    <div>
      <pre className="whitespace-pre-wrap text-xs font-mono">
        {expanded ? content : lines.slice(0, 15).join("\n")}
      </pre>
      <button
        onClick={() => setExpanded(!expanded)}
        className="mt-1 text-xs text-accent hover:text-accent-hover transition-colors"
      >
        {expanded ? "Show less" : `Show ${lines.length - 15} more lines`}
      </button>
    </div>
  );
}

function MessageCard({ message, index }: { message: ConversationMessage; index: number }) {
  const role = message.role;

  if (role === "system") {
    return (
      <CollapsibleBlock label="[System message]">
        <div className="rounded-lg border border-border bg-bg-card p-4">
          <pre className="whitespace-pre-wrap text-xs text-text-tertiary font-mono">
            {message.content}
          </pre>
        </div>
      </CollapsibleBlock>
    );
  }

  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-lg border border-accent/20 bg-accent/5 p-4">
          <p className="mb-1 text-[11px] font-medium text-accent/70">User</p>
          <div className="text-sm whitespace-pre-wrap">{message.content}</div>
        </div>
      </div>
    );
  }

  if (role === "tool-response" || role === "tool" || role === "tool_response") {
    return (
      <div className="rounded-lg border border-success/20 bg-success/5 p-4">
        <p className="mb-2 text-[11px] font-medium text-success/70">Tool Response</p>
        <ToolResponseContent content={message.content} />
      </div>
    );
  }

  // Assistant
  const parts = parseAssistantContent(message.content);
  const toolCalls = message.tool_calls;

  return (
    <div className="rounded-lg border border-border bg-bg-card p-4">
      <p className="mb-2 text-[11px] font-medium text-text-tertiary">Assistant</p>
      <div className="space-y-3">
        {parts.map((part, i) => {
          if (part.type === "think") {
            return (
              <CollapsibleBlock key={i} label="thinking">
                <div className="rounded border border-border bg-bg p-3 text-sm italic text-text-secondary">
                  {part.content}
                </div>
              </CollapsibleBlock>
            );
          }
          if (part.type === "tool_call") {
            return (
              <div key={i} className="rounded border border-border bg-bg p-3">
                <p className="mb-1 text-[11px] font-medium text-warning">{part.toolName}</p>
                <pre className="whitespace-pre-wrap text-xs font-mono text-text-secondary">
                  {part.content}
                </pre>
              </div>
            );
          }
          return (
            <div key={i} className="text-sm whitespace-pre-wrap">
              {part.content}
            </div>
          );
        })}
        {toolCalls?.map((tc, i) => {
          const name = tc.function?.name || tc.name || "tool";
          const args = tc.function?.arguments || "";
          return (
            <div key={`tc-${i}`} className="rounded border border-border bg-bg p-3">
              <p className="mb-1 text-[11px] font-medium text-warning">{name}</p>
              <pre className="whitespace-pre-wrap text-xs font-mono text-text-secondary">
                {args}
              </pre>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ConversationTimeline({ messages }: { messages: ConversationMessage[] }) {
  return (
    <div className="space-y-3">
      {messages.map((msg, i) => (
        <MessageCard key={i} message={msg} index={i} />
      ))}
    </div>
  );
}
