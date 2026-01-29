import { mkdir, readFile, writeFile, readdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { resolveGeminiWebSessionsDir, resolveGeminiWebSessionPath } from './paths.js';

export interface SessionMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  error?: string;
}

export interface SessionData {
  id: string;
  metadata: unknown;
  messages: SessionMessage[];
  createdAt: string;
  updatedAt: string;
}

export interface SessionListItem {
  id: string;
  updatedAt: string;
}

export async function readSession(id: string): Promise<SessionData | null> {
  const sessionPath = resolveGeminiWebSessionPath(id);
  try {
    const content = await readFile(sessionPath, 'utf8');
    return JSON.parse(content) as SessionData;
  } catch {
    return null;
  }
}

export async function writeSession(
  id: string,
  metadata: unknown,
  userMessage: string,
  assistantMessage: string,
  error?: string,
): Promise<void> {
  const sessionPath = resolveGeminiWebSessionPath(id);
  const sessionsDir = resolveGeminiWebSessionsDir();
  await mkdir(sessionsDir, { recursive: true });

  const existing = await readSession(id);
  const now = new Date().toISOString();

  const newMessages: SessionMessage[] = [
    { role: 'user', content: userMessage, timestamp: now },
    { role: 'assistant', content: assistantMessage, timestamp: now, ...(error && { error }) },
  ];

  const data: SessionData = {
    id,
    metadata,
    messages: [...(existing?.messages ?? []), ...newMessages],
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
  };
  await writeFile(sessionPath, JSON.stringify(data, null, 2));
}

export async function listSessions(limit = 100): Promise<SessionListItem[]> {
  const sessionsDir = resolveGeminiWebSessionsDir();
  try {
    const files = await readdir(sessionsDir);
    const jsonFiles = files.filter((f) => f.endsWith('.json'));

    const items: { id: string; updatedAt: string; mtime: number }[] = [];
    for (const file of jsonFiles) {
      const filePath = path.join(sessionsDir, file);
      try {
        const stats = await stat(filePath);
        items.push({
          id: file.slice(0, -5),
          updatedAt: stats.mtime.toISOString(),
          mtime: stats.mtime.getTime(),
        });
      } catch {
        continue;
      }
    }

    items.sort((a, b) => b.mtime - a.mtime);
    return items.slice(0, limit).map(({ id, updatedAt }) => ({ id, updatedAt }));
  } catch {
    return [];
  }
}
